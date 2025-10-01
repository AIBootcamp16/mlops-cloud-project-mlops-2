import os

import mlflow
import pandas as pd
import time


run_id = "86fcfb906b814ebb8359e5bad5a505c7"          # start_run에서 지정한 이름
artifact_path = "Dataset/spotify_data.csv"

mlflow_addr = os.environ.get("MLFLOW_ADDR")
mlflow.set_tracking_uri(f"{mlflow_addr}")
mlflow.set_experiment("spotipy_recommender")

# artifact 다운로드
local_path = mlflow.artifacts.download_artifacts(
    run_id=run_id,
    artifact_path=artifact_path
)

print("다운로드된 파일 경로:", local_path)

if not os.path.isfile(local_path):
    raise FileNotFoundError(f"파일이 존재하지 않습니다: {local_path}")

file_path = os.path.abspath(__file__)  # /home/user/project/main.py
dir_path = os.path.dirname(file_path)  # /home/user/project

# 바로 pandas로 읽기
df = pd.read_csv(local_path)
print(df.head())
print(f"{dir_path}/../../dataset/raw/spotify_data.csv")
df.to_csv(f"{dir_path}/../../dataset/raw/spotify_data.csv")
time.sleep(3)