import os

import pandas as pd

from model.NModel import FAISSRecommender
from model.NModel import LGBMRecommender


file_path = os.path.abspath(__file__)  # /home/user/project/main.py
dir_path = os.path.dirname(file_path)  # /home/user/project

df = pd.read_csv(f"{dir_path}/../dataset/processed/spotify_data_clean.csv")

# 2. 사용할 피처 선택 (벡터화)
features = [
    'danceability', 'energy', 'key', 'loudness', 'mode', 'speechiness',
    'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo', 'duration_ms'
]

FAISS_REC = FAISSRecommender(data = df, features=features, nlist=1000)

FAISS_REC.fit()

FAISS_REC.save(path="./")

FAISS_REC.release()

LGBM_REC = LGBMRecommender()

LGBM_REC.set_params({
            "num_leaves": 36,
            "learning_rate": 0.1,
            "num_threads": 4,   # CPU 4코어 사용
            "random_state": 42,
            "n_estimators": 1000
        })

LGBM_REC.fit(df=df,features=features,target='popularity')

LGBM_REC.MLFLOWProducionSelect()