import os

import pandas as pd
import mlflow

mlflow_addr = os.environ.get("MLFLOW_ADDR")

mlflow.set_tracking_uri(f"{mlflow_addr}")
mlflow.set_experiment("spotipy_recommender")

file_path = os.path.abspath(__file__)  # /home/user/project/main.py
dir_path = os.path.dirname(file_path)  # /home/user/project

with mlflow.start_run(run_name="dataset_run"):
    mlflow.log_artifact(f"{dir_path}/../dataset/raw/spotify_data.csv", artifact_path="Dataset")