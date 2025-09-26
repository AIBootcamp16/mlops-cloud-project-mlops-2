from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Dict
import json

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
import joblib
import faiss
import pickle
import lightgbm as lgb
from sklearn.model_selection import train_test_split

from ..data.features import (
    normalize_columns, coerce_types, basic_feature_engineering,
    select_feature_columns, REQUIRED_ID_COL
)
from ..data.preprocessing import build_preprocess_pipeline

@dataclass
class RecommenderArtifacts:
    pipeline: object
    knn: NearestNeighbors
    id_index: pd.DataFrame
    feature_names: List[str]

class Recommender:
    def __init__(self, n_neighbors: int = 50, metric: str = "cosine"):
        self.n_neighbors = n_neighbors
        self.metric = metric
        self.artifacts: Optional[RecommenderArtifacts] = None

    def fit(self, df: pd.DataFrame, model_dir: Optional[str] = None) -> "Recommender":
        df = normalize_columns(df)
        assert REQUIRED_ID_COL in df.columns, f"CSV must contain '{REQUIRED_ID_COL}' column."
        if "popularity" in df.columns:
            df = df.sort_values("popularity", ascending=False)
        df = df.drop_duplicates(subset=[REQUIRED_ID_COL]).reset_index(drop=True)

        df = coerce_types(df)
        df = basic_feature_engineering(df)

        num_cols, cat_cols = select_feature_columns(df, max_genres=30)
        pipe = build_preprocess_pipeline(num_cols, cat_cols)
        X = pipe.fit_transform(df)

        knn = NearestNeighbors(n_neighbors=self.n_neighbors, metric=self.metric, algorithm="auto")
        knn.fit(X)

        name_col = "track_name" if "track_name" in df.columns else REQUIRED_ID_COL
        artist_col = "artist_name" if "artist_name" in df.columns else None

        id_index = pd.DataFrame({
            "track_id": df[REQUIRED_ID_COL].values,
            "track_name": df[name_col].values if name_col in df.columns else df[REQUIRED_ID_COL].values,
            "artist_name": df[artist_col].values if artist_col else [""]*len(df)
        })

        feature_names = []
        try:
            ohe = pipe.named_steps["preprocess"].named_transformers_["cat"]
            num_features = num_cols
            cat_features = list(ohe.get_feature_names_out(cat_cols))
            feature_names = num_features + cat_features
        except Exception:
            feature_names = num_cols + cat_cols

        self.artifacts = RecommenderArtifacts(
            pipeline=pipe, knn=knn, id_index=id_index, feature_names=feature_names
        )

        if model_dir:
            self.save(model_dir)
        return self

    def fit_from_csv(self, csv_path: str, model_dir: Optional[str] = None) -> "Recommender":
        df = pd.read_csv(csv_path)
        return self.fit(df, model_dir=model_dir)

    def save(self, model_dir: str) -> None:
        assert self.artifacts is not None, "Call fit() first."
        p = Path(model_dir)
        p.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.artifacts.pipeline, p / "preprocess.joblib")
        joblib.dump(self.artifacts.knn, p / "knn.joblib")
        self.artifacts.id_index.to_parquet(p / "id_index.parquet", index=False)
        (p / "meta.json").write_text(
            json.dumps({
                "feature_names": self.artifacts.feature_names,
                "n_neighbors": self.n_neighbors,
                "metric": self.metric
            }, ensure_ascii=False, indent=2),
            encoding="utf-8"
        )

    @classmethod
    def load(cls, model_dir: str) -> "Recommender":
        p = Path(model_dir)
        pipeline_path = p / "preprocess.joblib"
        knn_path = p / "knn.joblib"
        id_index_path = p / "id_index.parquet"
        meta_path = p / "meta.json"
        if not (pipeline_path.exists() and knn_path.exists() and id_index_path.exists() and meta_path.exists()):
            raise FileNotFoundError(f"Model artifacts not found in {model_dir}. Run training first.")
        pipe = joblib.load(pipeline_path)
        knn = joblib.load(knn_path)
        id_index = pd.read_parquet(id_index_path)
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        rec = cls(n_neighbors=meta.get("n_neighbors", 50), metric=meta.get("metric","cosine"))
        rec.artifacts = RecommenderArtifacts(
            pipeline=pipe, knn=knn, id_index=id_index,
            feature_names=meta.get("feature_names", [])
        )
        return rec

    def _lookup_indices(self, by: str, query: str, max_matches: int = 50) -> List[int]:
        assert self.artifacts is not None, "Model not fitted."
        idx = self.artifacts.id_index
        by = by.lower()
        if by not in idx.columns.str.lower():
            raise ValueError(f"'by' must be one of {list(idx.columns)}")
        col = idx.columns[idx.columns.str.lower() == by][0]
        mask = idx[col].astype(str).str.contains(str(query), case=False, na=False)
        matches = np.where(mask.values)[0].tolist()
        return matches[:max_matches]

    def recommend_by_track_ids(self, track_ids: List[str], top_k: int = 10) -> pd.DataFrame:
        assert self.artifacts is not None, "Model not fitted."
        idx = self.artifacts.id_index
        seed_indices = idx.index[idx["track_id"].isin(track_ids)].tolist()
        if not seed_indices:
            raise ValueError("None of the provided track_ids found in index.")
        knn = self.artifacts.knn
        results: Dict[int, float] = {}
        for si in seed_indices:
            distances, neighbors = knn.kneighbors(n_neighbors=self.n_neighbors, X=[knn._fit_X[si]])
            for d, n in zip(distances[0], neighbors[0]):
                results[n] = min(results.get(n, np.inf), float(d))
        for si in seed_indices:
            results.pop(si, None)
        top = sorted(results.items(), key=lambda kv: kv[1])[:top_k]
        out_idx = [i for i, _ in top]
        dists = [d for _, d in top]
        out = idx.iloc[out_idx].copy().reset_index(drop=True)
        out.insert(0, "rank", np.arange(1, len(out)+1))
        out["distance"] = dists
        return out

    def recommend(self, by: str, query: str, top_k: int = 10) -> pd.DataFrame:
        matches = self._lookup_indices(by=by, query=query)
        if not matches:
            raise ValueError(f"No matches for {by} contains '{query}'.")
        seed_ids = self.artifacts.id_index.iloc[matches]["track_id"].tolist()
        return self.recommend_by_track_ids(seed_ids[:10], top_k=top_k)



class FAISSRecommender:
    #초기설정
    def __init__(self, data: pd.DataFrame, features, nlist = 100):
        # 피처 벡터화
        self.df = data
        self.features = features
        self.X = data[features].astype("float32").values
        d = self.X.shape[1]

        # 3. IVF 인덱스 생성
        quantizer = faiss.IndexFlatL2(d)
        self.index = faiss.IndexIVFFlat(quantizer, d, nlist, faiss.METRIC_L2)
    #학습
    def fit(self):
        print("Training FAISS IVF index...")
        self.index.train(self.X)

        # 5. 벡터 추가
        self.index.add(self.X)
        print(f"Index built. Total vectors: {self.index.ntotal}")

        # 6. 검색 편의용 매핑 (track_name, artist_name → index)
        self.track_to_idx = {name.lower(): i for i, name in enumerate(self.df["track_name"])}
        self.artist_to_idx = {}
        for i, artists in enumerate(self.df["artist_name"]):
            for artist in str(artists).split(","):
                self.artist_to_idx[artist.strip().lower()] = i
        self.id_to_idx = {tid: i for i, tid in enumerate(self.df["track_id"])}
    #추천
    def recommend(self, by: str, query: str, top_k: int = 5):
        """track_id, track_name, artist_name 기준 추천"""
        query_lower = query.lower()
        if by == "track_id":
            idx = self.id_to_idx.get(query)
        elif by == "track_name":
            idx = self.track_to_idx.get(query_lower)
        elif by == "artist_name":
            idx = self.artist_to_idx.get(query_lower)
        else:
            raise ValueError(f"Unsupported search key: {by}")

        if idx is None:
            raise ValueError(f"No results found for {by}='{query}'")

        # FAISS 검색
        query_vec = self.X[idx].reshape(1, -1)
        distances, indices = self.index.search(query_vec, top_k)

        # 결과 DataFrame 생성
        results = []
        for i, r_idx in enumerate(indices[0]):
            results.append({
                "rank": i + 1,
                "track_id": self.df.loc[r_idx, "track_id"],
                "track_name": self.df.loc[r_idx, "track_name"],
                "artist_name": self.df.loc[r_idx, "artist_name"],
                "distance": float(distances[0][i]),
            })
        return pd.DataFrame(results)
    
    # 모델 저장
    def save(self, path: str):
        """FAISS 인덱스와 매핑을 함께 저장"""
        # 1. FAISS 인덱스 저장
        faiss.write_index(self.index, f"{path}/faiss.index")

        # 2. 매핑 객체 저장
        with open(path, "wb") as f:
            pickle.dump({
                "track_to_idx": self.track_to_idx,
                "artist_to_idx": self.artist_to_idx,
                "id_to_idx": self.id_to_idx,
                "features": self.features,
                "X": self.X,
                "df": self.df
            }, f)
        print(f"Model saved to {path}")

    # 모델 불러오기
    @classmethod
    def load(cls, path: str):
        """저장된 FAISS 인덱스와 매핑을 불러와 Recommender 객체 생성"""
        # 1. 매핑 불러오기
        with open(path, "rb") as f:
            data = pickle.load(f)

        # 2. 객체 생성
        obj = cls(data["df"], data["features"])
        obj.X = data["X"]

        # 3. FAISS 인덱스 불러오기
        obj.index = faiss.read_index(f"{path}/faiss.index")

        # 4. 매핑 복원
        obj.track_to_idx = data["track_to_idx"]
        obj.artist_to_idx = data["artist_to_idx"]
        obj.id_to_idx = data["id_to_idx"]

        print(f"Model loaded from {path}")
        return obj

class LGBMRecommender:
    def __init__(self):
        self.model = None
        self.features = None

    def fit(self, df: pd.DataFrame, features, target: str):
        self.features = features
        X = df[features]
        y = df[target]
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
        # LightGBM 회귀 모델 생성
        self.model = lgb.LGBMRegressor(
            n_estimators=1000,
            num_leaves=31,
            learning_rate=0.1,
            num_threads=4,       # CPU 4코어 사용
            random_state=42
        )
        # LGBMRegressor 학습 (sklearn API)
        self.model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            eval_metric='rmse',
            callbacks=[
                lgb.early_stopping(stopping_rounds=100),
                lgb.log_evaluation(period=100)  # verbose 대신 사용
            ]
        )
        print("LGBM model trained.")

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        if self.model is None or self.features is None:
            raise ValueError("Model is not trained yet.")
        X = df[self.features]
        return self.model.predict(X)

    def save(self, path: str):
        if self.model is None or self.features is None:
            raise ValueError("Model is not trained yet.")
        joblib.dump({
            "model": self.model,
            "features": self.features
        }, path)
        print(f"LGBM model saved to {path}")

    @classmethod
    def load(cls, path: str) -> "LGBMRecommender":
        data = joblib.load(path)
        obj = cls()
        obj.model = data["model"]
        obj.features = data["features"]
        print(f"LGBM model loaded from {path}")
        return obj