
"""
spotify_recommender.py

Content-based music recommender using Spotify-style audio features.

Expected columns (case-insensitive, best effort mapping):
- "track_id" (unique id)          [required for dedup & lookup]
- "track_name"                     [recommended]
- "artist.name" or "artist_name"   [recommended]
- "genre"                          [optional, categorical]
- "year", "popularity"             [optional, numeric]
- Audio features (numeric):
  ["danceability","energy","loudness","speechiness","acousticness",
   "instrumentalness","liveness","valence","tempo","duration_ms",
   "time_signature","key","mode"]

USAGE (CLI):
    python spotify_recommender.py fit --csv ./spotify.csv --model_dir ./models
    python spotify_recommender.py rec --model_dir ./models --by "track_name" --q "I'm Yours" --top_k 10

USAGE (library):
    from spotify_recommender import Recommender
    rec = Recommender().fit_from_csv("spotify.csv")
    rec.recommend(by="track_name", query="I'm Yours", top_k=10)

Author: ChatGPT
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple, Dict

import argparse
import json
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.neighbors import NearestNeighbors
import joblib
import time

from mysql_logger import MySQLLogger

mysql_logger = MySQLLogger(
    host="114.203.195.166", user="root", password="root", database="mlops", port=3306
)

NUMERIC_CANDIDATES = [
    "danceability","energy","loudness","speechiness","acousticness",
    "instrumentalness","liveness","valence","tempo","duration_ms",
    "time_signature","popularity","year"
]
CATEGORICAL_CANDIDATES = ["genre","key","mode"]

REQUIRED_ID_COL = "track_id"
NAME_COLS = ["track_name", "artist.name", "artist_name"]  # best-effort


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    # Lowercase and replace spaces with underscores for robustness
    df = df.copy()
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
    # unify artist column name
    if "artist.name".lower() in df.columns and "artist_name" not in df.columns:
        df.rename(columns={"artist.name": "artist_name"}, inplace=True)
    return df


def _coerce_types(df: pd.DataFrame) -> pd.DataFrame:
    # Numeric coercion
    for col in NUMERIC_CANDIDATES:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    # Categorical coercion
    for col in CATEGORICAL_CANDIDATES:
        if col in df.columns:
            df[col] = df[col].astype("category")
    return df


def _basic_feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Duration in minutes (helps scale better than raw ms)
    if "duration_ms" in df.columns:
        df["duration_min"] = df["duration_ms"] / 60000.0

    # Loudness is negative dB; clip to a sensible range to reduce outliers’ influence
    if "loudness" in df.columns:
        df["loudness"] = df["loudness"].clip(lower=-40, upper=5)

    # Key + mode combo as categorical token (e.g., "4_major")
    if "key" in df.columns and "mode" in df.columns:
        # ensure numeric to string then categorical
        key_str = df["key"].astype("Int64").astype(str) if str(df["key"].dtype) != "category" else df["key"].astype(str)
        mode_str = df["mode"].astype("Int64").astype(str) if str(df["mode"].dtype) != "category" else df["mode"].astype(str)
        df["key_mode"] = (key_str + "_" + mode_str).astype("category")

    # Tempo buckets (optional) – can help a bit with rhythmic similarity
    if "tempo" in df.columns:
        df["tempo_bucket"] = pd.cut(df["tempo"].clip(lower=0, upper=250),
                                    bins=[0,60,90,110,130,160,250],
                                    labels=["slow","chill","mid","groove","up","fast"],
                                    include_lowest=True).astype("category")

    return df


def _select_feature_columns(df: pd.DataFrame, max_genres: int = 30):
    numeric = []
    categorical = []

    for col in NUMERIC_CANDIDATES + ["duration_min"]:
        if col in df.columns:
            numeric.append(col)

    for col in CATEGORICAL_CANDIDATES + ["key_mode","tempo_bucket"]:
        if col in df.columns:
            categorical.append(col)

    # Limit genre cardinality to top-N by frequency to keep OHE small
    if "genre" in categorical and "genre" in df.columns:
        # 상위 N개 장르 계산은 문자열 기준으로
        top_genres = df["genre"].astype(str).value_counts().nlargest(max_genres).index

        # Categorical이어도 안전하게 "__other__"를 추가한 뒤 치환
        if pd.api.types.is_categorical_dtype(df["genre"]):
            df["genre"] = df["genre"].cat.add_categories(["__other__"])
            mask = df["genre"].astype(str).isin(top_genres)
            df.loc[~mask, "genre"] = "__other__"
        else:
            mask = df["genre"].astype(str).isin(top_genres)
            df.loc[~mask, "genre"] = "__other__"
            df["genre"] = df["genre"].astype("category")

    return numeric, categorical

@dataclass
class RecommenderArtifacts:
    pipeline: Pipeline
    knn: NearestNeighbors
    id_index: pd.DataFrame  # columns: ["track_id","track_name","artist_name"]
    feature_names: List[str]


class Recommender:
    def __init__(self, n_neighbors: int = 50, metric: str = "cosine"):
        self.n_neighbors = n_neighbors
        self.metric = metric
        self.artifacts: Optional[RecommenderArtifacts] = None

    # ----------------------- Fit -----------------------
    def fit_from_csv(self, csv_path: str, model_dir: Optional[str] = None) -> "Recommender":
        df = pd.read_csv(csv_path)
        return self.fit(df, model_dir=model_dir)

    def fit(self, df: pd.DataFrame, model_dir: Optional[str] = None) -> "Recommender":
        df = _normalize_columns(df)
        assert REQUIRED_ID_COL in df.columns, f"CSV must contain '{REQUIRED_ID_COL}' column."

        # Deduplicate by track_id (keep the most popular entry if multiple)
        if "popularity" in df.columns:
            df = df.sort_values("popularity", ascending=False)
        df = df.drop_duplicates(subset=[REQUIRED_ID_COL]).reset_index(drop=True)

        # Coerce types, basic feature engineering
        df = _coerce_types(df)
        df = _basic_feature_engineering(df)

        # Select features present
        num_cols, cat_cols = _select_feature_columns(df, max_genres=30)

        # Compose preprocessing: standardize numeric + one-hot categorical
        preproc = ColumnTransformer(
            transformers=[
                ("num", StandardScaler(with_mean=True, with_std=True), num_cols),
                ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols),
            ],
            remainder="drop"
        )

        pipe = Pipeline([("preprocess", preproc)])
        X = pipe.fit_transform(df)

        # Fit kNN index on processed features
        knn = NearestNeighbors(n_neighbors=self.n_neighbors, metric=self.metric, algorithm="auto")
        knn.fit(X)

        # Build id index
        name_col = "track_name" if "track_name" in df.columns else REQUIRED_ID_COL
        artist_col = "artist_name" if "artist_name" in df.columns else None

        id_index = pd.DataFrame({
            "track_id": df[REQUIRED_ID_COL].values,
            "track_name": df[name_col].values if name_col in df.columns else df[REQUIRED_ID_COL].values,
            "artist_name": df[artist_col].values if artist_col else [""]*len(df)
        })

        feature_names = []
        # Try to get feature names for debugging
        try:
            ohe = pipe.named_steps["preprocess"].named_transformers_["cat"]
            num = pipe.named_steps["preprocess"].named_transformers_["num"]
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

    # ----------------------- Save / Load -----------------------
    def save(self, model_dir: str) -> None:
        assert self.artifacts is not None, "Call fit() first."
        model_path = Path(model_dir)
        model_path.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.artifacts.pipeline, model_path / "preprocess.joblib")
        joblib.dump(self.artifacts.knn, model_path / "knn.joblib")
        self.artifacts.id_index.to_parquet(model_path / "id_index.parquet", index=False)
        with open(model_path / "meta.json","w",encoding="utf-8") as f:
            json.dump({"feature_names": self.artifacts.feature_names,
                       "n_neighbors": self.n_neighbors,
                       "metric": self.metric}, f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, model_dir: str) -> "Recommender":
        model_path = Path(model_dir)
        pipe = joblib.load(model_path / "preprocess.joblib")
        knn = joblib.load(model_path / "knn.joblib")
        id_index = pd.read_parquet(model_path / "id_index.parquet")
        meta = json.loads(Path(model_path / "meta.json").read_text(encoding="utf-8"))
        rec = cls(n_neighbors=meta.get("n_neighbors", 50), metric=meta.get("metric","cosine"))
        rec.artifacts = RecommenderArtifacts(pipeline=pipe, knn=knn, id_index=id_index,
                                             feature_names=meta.get("feature_names", []))
        return rec

    # ----------------------- Recommend -----------------------
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

    def _embed(self, df: pd.DataFrame) -> np.ndarray:
        assert self.artifacts is not None, "Model not fitted."
        pipe = self.artifacts.pipeline
        return pipe.transform(df)

    def recommend_by_track_ids(self, track_ids: List[str], top_k: int = 10) -> pd.DataFrame:
        """Recommend similar tracks given 1+ seed track_ids.
        Returns a DataFrame with columns: rank, track_id, track_name, artist_name, distance
        """
        assert self.artifacts is not None, "Model not fitted."
        idx = self.artifacts.id_index
        # Gather seed indices
        seed_indices = idx.index[idx["track_id"].isin(track_ids)].tolist()
        if not seed_indices:
            raise ValueError("None of the provided track_ids found in index.")
        # Query the kNN index using the average of seed embeddings (centroid query)
        # To do that, we need to recover the original rows to transform
        # We'll simulate by constructing a small df from id_index lookups
        seed_df = pd.DataFrame({ "track_id": idx.iloc[seed_indices]["track_id"].values })
        # We don't have original raw features here; instead, we approximate by using
        # the fitted pipeline on the original training dataframe via its order.
        # An easier approach: query neighbors for each seed separately and merge.
        # We'll do the latter for robustness.
        knn = self.artifacts.knn
        results: Dict[int, float] = {}
        for si in seed_indices:
            distances, neighbors = knn.kneighbors(n_neighbors=self.n_neighbors, X=[knn._fit_X[si]])
            for d, n in zip(distances[0], neighbors[0]):
                results[n] = min(results.get(n, np.inf), float(d))

        # Remove seeds & sort
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
        """Find seed(s) by field (track_name | artist_name | track_id) and recommend."""
        matches = self._lookup_indices(by=by, query=query)
        if not matches:
            raise ValueError(f"No matches for {by} contains '{query}'.")
        seed_ids = self.artifacts.id_index.iloc[matches]["track_id"].tolist()
        return self.recommend_by_track_ids(seed_ids[:10], top_k=top_k)  # cap many matches

# --------------------------- CLI ---------------------------

def _cli_fit(args):
    rec = Recommender(n_neighbors=args.n_neighbors, metric=args.metric).fit_from_csv(args.csv, model_dir=args.model_dir)
    print(f"Fitted and saved to {args.model_dir}")

def _cli_rec(args):
    rec = Recommender.load(args.model_dir)
    t0 = time.time()
    out = rec.recommend(by=args.by, query=args.q, top_k=args.top_k)
    elapsed_sec = time.time() - t0
    returned_ids = out["track_id"].tolist()
    
    # seeds: recommend(by, query)는 내부에서 매칭한 seed들을 사용함
    # 간단히 'by,query'로 다시 찾아서 seed track_ids를 확보 (혹은 코드에 seed 반환값을 추가)
    seed_ids = []
    try:
        matches = rec._lookup_indices(by=args.by, query=args.q)  # 내부 헬퍼 사용
        seed_ids = rec.artifacts.id_index.iloc[matches]["track_id"].tolist()[:10]
    except:
        pass

    # 지연시간은 호출부에서 측정했다고 가정 (없으면 0으로)
    # elapsed_sec = 0.0
    print("soeun!!")
    print(returned_ids)
    mysql_logger.log_recommend(
        by_field=args.by,
        query=args.q,
        top_k=args.top_k,
        elapsed_sec=elapsed_sec,
        seed_track_ids=seed_ids,
        returned_track_ids=returned_ids,
    )
    
    print(out.to_string(index=False))

def _build_argparser():
    p = argparse.ArgumentParser(description="Spotify-style content-based recommender")
    sub = p.add_subparsers(required=True)

    pf = sub.add_parser("fit", help="Fit model from CSV")
    pf.add_argument("--csv", required=True, help="Path to CSV")
    pf.add_argument("--model_dir", default="./models", help="Where to save the model artifacts")
    pf.add_argument("--n_neighbors", type=int, default=50)
    pf.add_argument("--metric", default="cosine", choices=["cosine","euclidean","manhattan"])
    pf.set_defaults(func=_cli_fit)

    pr = sub.add_parser("rec", help="Get recommendations")
    pr.add_argument("--model_dir", default="./models")
    pr.add_argument("--by", default="track_name", choices=["track_id","track_name","artist_name"])
    pr.add_argument("--q", required=True, help="search query for the --by field")
    pr.add_argument("--top_k", type=int, default=10)
    pr.set_defaults(func=_cli_rec)

    return p

if __name__ == "__main__":
    parser = _build_argparser()
    args = parser.parse_args()
    args.func(args)
