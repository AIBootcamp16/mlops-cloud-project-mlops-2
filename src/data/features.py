from __future__ import annotations

import pandas as pd
import numpy as np

NUMERIC_CANDIDATES = [
    "danceability","energy","loudness","speechiness","acousticness",
    "instrumentalness","liveness","valence","tempo","duration_ms",
    "time_signature","popularity","year"
]
CATEGORICAL_CANDIDATES = ["genre","key","mode"]
REQUIRED_ID_COL = "track_id"
NAME_COLS = ["track_name", "artist.name", "artist_name"]

def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
    if "artist.name" in df.columns and "artist_name" not in df.columns:
        df.rename(columns={"artist.name": "artist_name"}, inplace=True)
    return df

def coerce_types(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in NUMERIC_CANDIDATES:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    for col in CATEGORICAL_CANDIDATES:
        if col in df.columns:
            df[col] = df[col].astype("category")
    return df

def basic_feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "duration_ms" in df.columns:
        df["duration_min"] = df["duration_ms"] / 60000.0
    if "loudness" in df.columns:
        df["loudness"] = df["loudness"].clip(lower=-40, upper=5)
    if "key" in df.columns and "mode" in df.columns:
        key_str = df["key"].astype("Int64").astype(str) if str(df["key"].dtype) != "category" else df["key"].astype(str)
        mode_str = df["mode"].astype("Int64").astype(str) if str(df["mode"].dtype) != "category" else df["mode"].astype(str)
        df["key_mode"] = (key_str + "_" + mode_str).astype("category")
    if "tempo" in df.columns:
        df["tempo_bucket"] = pd.cut(
            df["tempo"].clip(lower=0, upper=250),
            bins=[0,60,90,110,130,160,250],
            labels=["slow","chill","mid","groove","up","fast"],
            include_lowest=True
        ).astype("category")
    return df

def select_feature_columns(df: pd.DataFrame, max_genres: int = 30):
    numeric, categorical = [], []
    for col in NUMERIC_CANDIDATES + ["duration_min"]:
        if col in df.columns:
            numeric.append(col)
    for col in CATEGORICAL_CANDIDATES + ["key_mode","tempo_bucket"]:
        if col in df.columns:
            categorical.append(col)
    if "genre" in categorical and "genre" in df.columns:
        top_genres = df["genre"].astype(str).value_counts().nlargest(max_genres).index
        if pd.api.types.is_categorical_dtype(df["genre"]):
            df["genre"] = df["genre"].cat.add_categories(["__other__"])
            mask = df["genre"].astype(str).isin(top_genres)
            df.loc[~mask, "genre"] = "__other__"
        else:
            mask = df["genre"].astype(str).isin(top_genres)
            df.loc[~mask, "genre"] = "__other__"
            df["genre"] = df["genre"].astype("category")
    return numeric, categorical
