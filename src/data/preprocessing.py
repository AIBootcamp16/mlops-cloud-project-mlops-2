from __future__ import annotations

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline

def build_preprocess_pipeline(num_cols, cat_cols) -> Pipeline:
    preproc = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(with_mean=True, with_std=True), num_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols),
        ],
        remainder="drop"
    )
    pipe = Pipeline([("preprocess", preproc)])
    return pipe
