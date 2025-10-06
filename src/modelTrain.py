import numpy as np
import pandas as pd
import joblib 
import os  
import mlflow 
import mlflow.pyfunc 
import sys 

# 💡 현재 디렉토리를 sys.path에 추가하여 모듈 검색 경로를 설정
sys.path.append(os.path.dirname(os.path.abspath(__file__))) 

# nn_model.py 파일에서 SimpleNN 클래스를 임포트합니다.
# MLflow 아티팩트 저장 시 이 클래스의 정의를 찾을 수 있도록 합니다.
from nn_model import SimpleNN 
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

# 💡 수정: 모델 저장 경로를 "src" 디렉토리에서 한 단계 위인 "프로젝트 루트" 아래의 "models"로 설정
# src/modelTrain.py가 실행되면, '../models'는 '프로젝트_루트/models'가 됩니다.
FAISS_REC.save(path="../models") 

# FAISS_REC.release()

LGBM_REC = LGBMRecommender()

LGBM_REC.set_params({
            "num_leaves": 36,
            "learning_rate": 0.1,
            "num_threads": 4,   # CPU 4코어 사용
            "random_state": 42,
            "n_estimators": 1000
        })


LGBM_REC.fit(df=df,features=features,target='popularity')

# 💡 수정된 부분: Producion -> Production (오타 수정)
LGBM_REC.MLFLOWProductionSelect()

