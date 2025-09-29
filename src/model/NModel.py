from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Dict
import json
import threading
import time
import os
import queue

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import joblib
import faiss
import pickle
import lightgbm as lgb
import mlflow
import mlflow.pyfunc
from mlflow.models import get_model_info
from mlflow.artifacts import download_artifacts
#from mlflow import artifacts

mlflow_addr = os.environ.get("MLFLOW_ADDR")

mlflow.set_tracking_uri(f"{mlflow_addr}")
mlflow.set_experiment("Recommender_Exp")

class MLFLOWLogBuilder:
    def __init__(self,run_name):
        self.run_name = run_name
        self.params = {}
        self.metrics = {}

    def add_param(self, key, value):
        self.params[key] = value
        return self  # 체이닝 가능

    def add_metric(self, key, value):
        self.metrics[key] = value
        return self  # 체이닝 가능

    def build(self):
        """최종 MLflow log_data 생성"""
        return {"run_name":self.run_name,"params": self.params, "metrics": self.metrics}

class MLFLOW_Logger:
    _instance = None
    _lock = threading.Lock()  # 멀티스레드 환경 안전하게 싱글톤 생성
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        self.log_queue = queue.Queue()
        self.log_thread = None
        self._initialized = True
    
    def loger_start(self):
        # MLflow 로깅을 위한 큐와 스레드 초기화
        if self.log_thread is None or not self.log_thread.is_alive():
            self.log_thread = threading.Thread(target=self._log_worker, daemon=True)
            self.log_thread.start()

    def _log_worker(self):
        """백그라운드에서 큐를 모니터링하며 MLflow에 기록"""
        while True:
            log_data = self.log_queue.get()
            if log_data is None:  # 종료 신호
                break
            try:

                with mlflow.start_run(run_name=log_data["run_name"]):
                    mlflow.log_params(log_data["params"])
                    mlflow.log_metrics(log_data["metrics"])
            except Exception as e:
                print("MLflow logging error:", e)
            time.sleep(0.01)  # 큐 비우기 대기
        
    def write_log(self,log_data):
        self.log_queue.put(log_data)
        
    def loger_stop(self):
        if self.log_thread is None or not self.log_thread.is_alive():
            self.log_queue.put(None)
            self.log_thread.join()

class RecommenderWrapper(mlflow.pyfunc.PythonModel):
    # FAISSRecommender save 함수
    @classmethod
    def FAISS_save_model_with_mlflow(self, path: str):
        os.makedirs(path, exist_ok=True)
        faiss_index_path = os.path.join(path, "faiss.index")
        mapping_path = os.path.join(path, "mapping.pkl")

        mlflow.pyfunc.log_model(
            python_model=RecommenderWrapper(),   # wrapper 인스턴스 사용
            artifacts={
                "faiss_index": faiss_index_path,
                "mapping": mapping_path
            },
            registered_model_name="Recommender_FAISS",
            name="recommender_faiss"
        )

class FAISSRecommender():
    def __init__(self, data: pd.DataFrame, features, nlist = 100):
        # 피처 벡터화
        self.df = data
        self.features = features
        self.X = data[features].astype("float32").values
        d = self.X.shape[1]

        # 3. IVF 인덱스 생성
        quantizer = faiss.IndexFlatL2(d)
        self.index = faiss.IndexIVFFlat(quantizer, d, nlist, faiss.METRIC_L2)
        self.mlflow_logger = MLFLOW_Logger()
        self.mlflow_logger.loger_start()
        
    def release(self):
        self.mlflow_logger.loger_stop()

    def fit(self):
        start_time = time.time()
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


        # MLflow 로깅
        with mlflow.start_run(run_name = "faiss_fit") as run:
            mlflow.log_param("n_vectors", len(self.X))
            mlflow.log_param("n_features", len(self.features))
            mlflow.log_param("index_type", "IVFFlat")
            mlflow.log_metric("ntotal", self.index.ntotal)
            mlflow.log_metric("fit_sec", time.time() - start_time)
            print(f"Logged to MLflow Run ID={run.info.run_id}")


    def recommend(self, by: str, query: str, top_k: int = 5):
        start_time = time.time()
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

        result_df = pd.DataFrame(results)
        # MLflow 로깅
        log_builder = MLFLOWLogBuilder("faiss_recommender")
        log_builder.add_param("query_by", by)
        log_builder.add_param("query_value", query)
        log_builder.add_param("top_k", top_k)
        log_builder.add_metric("returned", len(result_df))
        log_builder.add_metric("recommender_sec", time.time() - start_time)
        log_builder.add_metric("avg_distance", float(result_df["distance"].mean()))
        mlflow_log = log_builder.build()
        self.mlflow_logger.write_log(mlflow_log)
        return result_df
    
    # 모델 저장
    def save(self, path: str):
        """FAISS 인덱스와 매핑을 함께 저장"""
        # 1. FAISS 인덱스 저장
        os.makedirs(path, exist_ok=True)
        faiss_index_path = os.path.join(path, "faiss.index")
        mapping_path = os.path.join(path, "mapping.pkl")
        faiss.write_index(self.index, faiss_index_path)

        # 2. 매핑 객체 저장
        with open(mapping_path, "wb") as f:
            pickle.dump({
                "track_to_idx": self.track_to_idx,
                "artist_to_idx": self.artist_to_idx,
                "id_to_idx": self.id_to_idx,
                "features": self.features,
                "X": self.X,
                "df": self.df
            }, f)

        RecommenderWrapper.FAISS_save_model_with_mlflow(path)
        
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
    
    # 모델 불러오기
    @classmethod
    def MLFLOWload(cls):
        """저장된 FAISS 인덱스와 매핑을 불러와 Recommender 객체 생성"""
        model_versions = mlflow.search_model_versions(filter_string = f"name='Recommender_FAISS'")
        FAISS_version = max(v.version for v in model_versions)

        # 1. artifacts를 로컬 경로로 다운로드
        local_artifacts_path = download_artifacts(f"models:/Recommender_FAISS/{FAISS_version}")

        mapping_path = os.path.join(local_artifacts_path, "artifacts/mapping.pkl")
        faiss_path = os.path.join(local_artifacts_path, "artifacts/faiss.index")

        # 3. 매핑 불러오기
        with open(mapping_path, "rb") as f:
            data = pickle.load(f)

        # 4. 객체 생성
        obj = cls(data["df"], data["features"])
        obj.X = data["X"]

        # 5. FAISS 인덱스 불러오기
        obj.index = faiss.read_index(faiss_path)

        # 6. 매핑 복원
        obj.track_to_idx = data["track_to_idx"]
        obj.artist_to_idx = data["artist_to_idx"]
        obj.id_to_idx = data["id_to_idx"]

        #print(f"Model loaded from {path}")
        return obj

class LGBMRecommender:
    
    def __init__(self):
        self.model = None
        self.features = None
        self.params = {
            "num_leaves": 31,
            "learning_rate": 0.05,
            "num_threads": 4,   # CPU 4코어 사용
            "random_state": 42,
            "n_estimators": 1000
        }
        self.test_rmse = 0

    def set_params(self,params):
        self.params = params

    def set_features(self,features):
        self.features = features

    def fit(self, df: pd.DataFrame, features, target: str):
        start_time=time.time();
        self.features = features
        X = df[features]
        y = df[target]
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
        X_train_val, x_test, y_train_val, y_test = train_test_split(X_val, y_val, test_size=0.5, random_state=42)


        self.test_x = x_test
        self.test_y = y_test

        # LightGBM 회귀 모델 생성
        self.model = lgb.LGBMRegressor(**self.params)

        # LGBMRegressor 학습 (sklearn API)
        self.model.fit(
            X_train, y_train,
            eval_set=[(X_train_val, y_train_val)],
            eval_metric='rmse',
            callbacks=[
                lgb.early_stopping(stopping_rounds=100),
                lgb.log_evaluation(period=100)  # verbose 대신 사용
            ]
        )

        y_test_pred = self.predict(x_test)
        self.test_rmse = mean_squared_error(y_test, y_test_pred)

        input_example = pd.DataFrame([X_train.iloc[0]])

        # MLflow 로깅
        with mlflow.start_run(run_name = "LGBM_fit") as run:
            mlflow.log_params(self.params)
            mlflow.log_metric("rmse", self.test_rmse)
            mlflow.log_metric("fit_sec", time.time() - start_time)
            model_info = mlflow.lightgbm.log_model(
                self.model, name="lgbm_model", 
                registered_model_name="spotipy_LGBM", 
                input_example=input_example)
            
            with open("features.json", "w") as f:
                json.dump(self.features, f)

            mlflow.log_artifact("features.json", artifact_path="lgbm_model")

            self.model_version = model_info.registered_model_version
            mlflow.set_model_version_tag(
                name="spotipy_LGBM",
                version=f'{model_info.registered_model_version}',
                key='status',
                value="archived"
            )
        
    def MLFLOWProducionSelect(self):
        if(self.test_rmse == 0):
            return

        production_model = LGBMRecommender.MLFLOW_load()
        
        y_production_pred = production_model.predict(self.test_x)

        test_production_rmse = mean_squared_error(self.test_y, y_production_pred)
        if test_production_rmse > self.test_rmse:
                mlflow.set_model_version_tag(
                name="spotipy_LGBM",
                version=f'{self.model_version}',
                key='status',
                value="production")
                mlflow.set_model_version_tag(
                name="spotipy_LGBM",
                version=f'{production_model.model_version}',
                key='status',
                value="archived")

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        if self.model is None:
            print("not model")
        if self.features is None:
                self.MLFLOW_load_features()
        X = df[self.features]
        return self.model.predict(X)

    def save(self, path: str):
        if self.model is None or self.features is None:
            raise ValueError("Model is not trained yet.")
        joblib.dump({
            "model": self.model,
            "features": self.features
        }, path)
        mlflow.lightgbm.log_model(self.model, name="lgbm_model", registered_model_name="FAISS_LGBM")
        print(f"LGBM model saved to {path}")

    #Model file features load
    def MLFLOW_load_features(self):
        model_name = 'spotipy_LGBM'
        # 모든 버전 검색
        all_versions = mlflow.search_model_versions(filter_string =f"name='{model_name}'")

        # tag 'state'가 'production'인 버전만 필터
        prod_versions = [v for v in all_versions if v.tags.get("status") == "production"]

        model_uri = f"models:/spotipy_LGBM/{prod_versions[0].version}"
        local_path = download_artifacts(
            artifact_uri=model_uri + "/input_example.json"
        )

        with open(local_path) as f:
            features = json.load(f)
        self.features = features['columns']

    @classmethod
    def load(cls, path: str) -> "LGBMRecommender":
        data = joblib.load(path)
        obj = cls()
        obj.model = data["model"]
        obj.features = data["features"]
        print(f"LGBM model loaded from {path}")
        return obj
    
    # MLFLOW production model load
    @classmethod
    def MLFLOW_load(cls) -> "LGBMRecommender":
        model_name = 'spotipy_LGBM'
        # 모든 버전 검색
        all_versions = mlflow.search_model_versions(filter_string =f"name='{model_name}'")

        # tag 'state'가 'production'인 버전만 필터
        prod_versions = [v for v in all_versions if v.tags.get("status") == "production"]

        model_uri = f"models:/spotipy_LGBM/{prod_versions[0].version}"
        result = LGBMRecommender()
        result.model_version = prod_versions[0].version
        model = mlflow.lightgbm.load_model(model_uri)
        result.model = model
        return result