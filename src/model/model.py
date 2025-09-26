from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Dict
import json

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
import joblib

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
