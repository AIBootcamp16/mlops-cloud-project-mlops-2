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

mlflow_addr = os.environ.get("MLFLOW_ADDR")

mlflow.set_tracking_uri(f"{mlflow_addr}")
mlflow.set_experiment("spotipy_recommender")

@dataclass
class FinderArtifacts:
    pipeline: object
    id_index: pd.DataFrame
    feature_names: List[str]

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
    
    def logger_start(self):
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
        
    def logger_stop(self):
        if self.log_thread is None or not self.log_thread.is_alive():
            self.log_queue.put(None)
            self.log_thread.join()

# 음악 검색
class Finder:
    def __init__(self, n_neighbors: int = 50, metric: str = "cosine"):
        self.n_neighbors = n_neighbors
        self.metric = metric
        self.artifacts: Optional[FinderArtifacts] = None

    @classmethod
    def load(cls, model_dir: str) -> "Finder":
        p = Path(model_dir)
        
        # --- [수정] 실제 존재하는 파일 이름으로 경로 변경 ---
        pipeline_path = p / "simple_nn_model.joblib" # 'preprocess.joblib' 대신 사용 (파이프라인 또는 NN 모델로 가정)
        mapping_pkl_path = p / "mapping.pkl"         # 'id_index.parquet' 대신 사용 (FAISSRecommender.save 참조)
        meta_path = p / "meta.json"                   # meta.json은 여전히 기대하지만, 없으면 경고 후 진행하도록 로직 변경

        # 파일 존재 여부 확인 (meta.json 요구 사항은 일단 제거하고 핵심 파일만 확인)
        if not (pipeline_path.exists() and mapping_pkl_path.exists()):
            raise FileNotFoundError(f"필수 모델 아티팩트 (simple_nn_model.joblib, mapping.pkl)가 {model_dir}에 없습니다. 훈련을 먼저 실행하세요.")

        # 1. pipeline 로드 (simple_nn_model.joblib이 LGBM 모델 형태의 딕셔너리로 저장되었을 수 있음)
        pipe_data = joblib.load(pipeline_path)
        # LGBMRecommender.save를 보면, { "model": <lgbm_model>, "features": [...] } 형태로 저장됩니다.
        # Finder가 사용하는 pipeline은 전처리 파이프라인(Scaler) 또는 NN 모델 자체여야 합니다. 
        # 여기서는 단순화를 위해 joblib.load 결과를 pipe로 바로 사용하거나, dict에서 모델을 꺼냅니다.
        pipe = pipe_data.get("model") if isinstance(pipe_data, dict) and "model" in pipe_data else pipe_data
        
        # 2. id_index 로드 (mapping.pkl에서 'df' 전체 데이터프레임을 추출)
        with open(mapping_pkl_path, "rb") as f:
            mapping_data = pickle.load(f)
            
        # FAISSRecommender.save를 보면, 'df' 키에 전체 데이터프레임이 저장되어 있습니다.
        id_index: pd.DataFrame = mapping_data.get("df")
        feature_names: List[str] = mapping_data.get("features", []) # FAISS에 사용된 피처 이름

        # 3. meta 정보 로드 또는 기본값 사용
        if meta_path.exists():
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        else:
            print("⚠️ meta.json 파일을 찾을 수 없습니다. 기본 메타데이터를 사용합니다.")
            meta = {"n_neighbors": 50, "metric": "cosine", "feature_names": feature_names}

        # 4. Finder 인스턴스 생성 및 아티팩트 할당
        rec = cls(n_neighbors=meta.get("n_neighbors", 50), metric=meta.get("metric","cosine"))
        rec.artifacts = FinderArtifacts(
            pipeline=pipe,
            id_index=id_index,
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

class RecommenderWrapper(mlflow.pyfunc.PythonModel):
    # FAISSRecommender save 함수
    @classmethod
    def FAISS_save_model_with_mlflow(self, path: str):
        os.makedirs(path, exist_ok=True)
        faiss_index_path = os.path.join(path, "faiss.index")
        mapping_path = os.path.join(path, "mapping.pkl")

        mlflow.pyfunc.log_model(
            python_model=RecommenderWrapper(),  # wrapper 인스턴스 사용
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
        #self.mlflow_logger = MLFLOW_Logger()
        #self.mlflow_logger.logger_start()
        
    #def release(self):
    #    self.mlflow_logger.logger_stop()

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

        # 2. 사용할 피처 선택 (벡터화)
        features = [
            'danceability', 'energy', 'key', 'loudness', 'mode', 'speechiness',
            'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo', 'duration_ms'
        ]
        # 결과 DataFrame 생성
        results = []
        for i, r_idx in enumerate(indices[0]):
            results.append({
                "rank": i + 1,
                "track_id": self.df.loc[r_idx, "track_id"],
                "track_name": self.df.loc[r_idx, "track_name"],
                "artist_name": self.df.loc[r_idx, "artist_name"],
                "distance": float(distances[0][i]),
                #추가 아이템
                "danceability": self.df.loc[r_idx, "danceability"],
                "energy": self.df.loc[r_idx, "energy"],
                "key": self.df.loc[r_idx, "key"],
                "loudness": self.df.loc[r_idx, "loudness"],
                "mode": self.df.loc[r_idx, "mode"],
                "speechiness": self.df.loc[r_idx, "speechiness"],
                "acousticness": self.df.loc[r_idx, "acousticness"],
                "instrumentalness": self.df.loc[r_idx, "instrumentalness"],
                "liveness": self.df.loc[r_idx, "liveness"],
                "valence": self.df.loc[r_idx, "valence"],
                "tempo": self.df.loc[r_idx, "tempo"],
                "duration_ms": self.df.loc[r_idx, "duration_ms"],
                
            })

        result_df = pd.DataFrame(results)
        # MLflow 로깅
        #log_builder = MLFLOWLogBuilder("faiss_recommender")
        #log_builder.add_param("query_by", by)
        #log_builder.add_param("query_value", query)
        #log_builder.add_param("top_k", top_k)
        #log_builder.add_metric("returned", len(result_df))
        #log_builder.add_metric("recommender_sec", time.time() - start_time)
        #log_builder.add_metric("avg_distance", float(result_df["distance"].mean()))
        #mlflow_log = log_builder.build()
        #self.mlflow_logger.write_log(mlflow_log)
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

    # 모델 불러오기 (로컬 경로)
    @classmethod
    def load(cls, path: str):
        """저장된 FAISS 인덱스와 매핑을 불러와 Recommender 객체 생성"""
        p = Path(path)
        if p.is_dir():
            mapping_path = p / "mapping.pkl" # <-- 'mappings.pkl'에서 'mapping.pkl'로 수정
            index_path   = p / "faiss.index"
        else:
            mapping_path = p
            index_path   = p.with_name("faiss.index")

        if not mapping_path.exists():
            raise FileNotFoundError(f"Mapping file not found: {mapping_path}")
        if not index_path.exists():
            raise FileNotFoundError(f"FAISS index not found: {index_path}")
        
        # 1. 매핑 불러오기
        with open(mapping_path, "rb") as f:
            data = pickle.load(f)
        
        # 2. 객체 생성
        obj = cls(data["df"], data["features"])
        obj.X = data["X"]
        
        # 3. FAISS 인덱스 불러오기
        # obj.index = faiss.read_index(f"{path}/faiss.index")
        obj.index = faiss.read_index(str(index_path))

        # 4. 매핑 복원
        obj.track_to_idx = data["track_to_idx"]
        obj.artist_to_idx = data["artist_to_idx"]
        obj.id_to_idx = data["id_to_idx"]
        
        print(f"Model loaded from {mapping_path.parent.resolve()}")
        return obj
    
    # 모델 불러오기 (MLflow)
    @classmethod
    def MLFLOWload(cls):
        os.environ["MLFLOW_ADDR"] = "http://114.203.195.166:5000"
        os.environ["MLFLOW_S3_ENDPOINT_URL"] = "http://114.203.195.166:9000"
        os.environ["AWS_ACCESS_KEY_ID"] = "admin"
        os.environ["AWS_SECRET_ACCESS_KEY"] = "admin1234"

        mlflow_addr = os.environ.get("MLFLOW_ADDR")
        
        mlflow.set_tracking_uri(f"{mlflow_addr}")
        mlflow.set_experiment("spotipy_recommender")

        model_name = 'Recommender_FAISS'
        model_versions = mlflow.search_model_versions(filter_string = f"name='{model_name}'")
        
        # FIX 1: 모델 버전이 없는 경우 예외 처리 추가 (유지)
        if not model_versions:
              raise RuntimeError(f"MLflow 등록 모델 '{model_name}' 버전을 찾을 수 없습니다. 훈련을 먼저 실행하세요.")

        FAISS_version = max(v.version for v in model_versions)
        
        # 2. artifacts를 로컬 경로로 다운로드
        # 변수명을 local_download_root_path로 변경하고, 경로 탐색 로직 강화
        local_download_root_path = download_artifacts(f"models:/{model_name}/{FAISS_version}")
        print(f"MLFLOW load: MLflow 아티팩트 다운로드 루트: {local_download_root_path}") # 디버깅용 로그

        # --- FIX 2 (경로 탐색 로직 강화): 여러 가능한 경로를 시도하여 mapping.pkl 찾기 ---
        
        # 시도할 경로 목록 정의
        possible_paths = [
            # 시도 1: <root>/artifacts/mapping.pkl (mlflow.pyfunc.log_model의 artifacts 구조)
            os.path.join(local_download_root_path, "artifacts/mapping.pkl"),
            # 시도 2: <root>/mapping.pkl (직접 로그한 경우)
            os.path.join(local_download_root_path, "mapping.pkl"),
            # 시도 3: <root>/mapping/mapping.pkl (일부 MLflow 환경의 심볼릭 링크 처리)
            os.path.join(local_download_root_path, "mapping/mapping.pkl"),
            # 시도 4: <root>/artifacts/mapping/mapping.pkl (가장 보수적인 경로)
            os.path.join(local_download_root_path, "artifacts/mapping/mapping.pkl"),
        ]

        data = None
        current_mapping_path = None

        for p in possible_paths:
            if os.path.exists(p):
                try:
                    with open(p, "rb") as f:
                        data = pickle.load(f)
                    current_mapping_path = p
                    print(f"MLFLOWload: 아티팩트 로드 성공 (경로: {current_mapping_path})")
                    break # 성공했으면 루프 종료
                except Exception as e:
                    # 파일이 있지만 로드에 실패한 경우
                    print(f"경로 {p}에서 파일 로드 중 예외 발생: {e}")
                    pass


        if current_mapping_path is None:
            # 모든 경로 시도가 실패했을 때 최종 에러 발생
            raise RuntimeError(
                f"MLflow 아티팩트 로드 실패: mapping.pkl을 찾을 수 없습니다. "
                f"시도된 경로: {', '.join(possible_paths)}"
            )

        # mapping.pkl 경로가 확인되면, faiss.index의 경로도 추론합니다.
        current_faiss_path = current_mapping_path.replace("mapping.pkl", "faiss.index")

        if not os.path.exists(current_faiss_path):
            raise RuntimeError(f"FAISS index 파일({current_faiss_path})을 찾을 수 없습니다. 경로 확인 필요.")


        # 3. 매핑 불러오기 (이미 위에서 로드되었으므로 data 변수 사용)
        if data is None:
            raise RuntimeError("data 변수가 로드되지 않았습니다. 로드 로직을 확인하세요.")

        
        # 4. 객체 생성
        obj = cls(data["df"], data["features"])
        obj.X = data["X"]

        # 5. FAISS 인덱스 불러오기
        obj.index = faiss.read_index(current_faiss_path)

        # 6. 매핑 복원
        obj.track_to_idx = data["track_to_idx"]
        obj.artist_to_idx = data["artist_to_idx"]
        obj.id_to_idx = data["id_to_idx"]

        print(f"MLFLOW load: Model loaded successfully from MLflow version {FAISS_version}")
        return obj

class LGBMRecommender:
    
    def __init__(self):
        self.model = None
        self.features = None
        self.params = {
            "num_leaves": 31,
            "learning_rate": 0.05,
            "num_threads": 4,  # CPU 4코어 사용
            "random_state": 42,
            "n_estimators": 1000
        }
        self.test_rmse = 0

    def set_params(self,params):
        self.params = params

    def set_features(self,features):
        self.features = features

    def fit(self, df: pd.DataFrame, features, target: str):
        start_time=time.time()
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
        
    def MLFLOWProductionSelect(self):
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
        # --- [수정] 디렉토리 경로 대신 파일 경로를 받도록 Path 객체 사용 ---
        p = Path(path)
        if p.is_dir():
             # main.py에서 모델 디렉토리를 넘겼다고 가정하고, 파일 이름을 추가
            model_file_path = p / "simple_nn_model.joblib" 
        else:
             # 파일 경로가 이미 넘어왔다면 그대로 사용
            model_file_path = p 

        if not model_file_path.exists():
            raise FileNotFoundError(f"LGBM model file not found: {model_file_path}")
            
        data = joblib.load(model_file_path)
        
        obj = cls()
        
        # --- [수정된 부분: 'model' KeyError 방지 로직 추가] ---
        # 로드된 데이터가 딕셔너리가 아닌 경우, data 자체가 모델 객체라고 가정합니다.
        if isinstance(data, dict) and "model" in data:
            # 딕셔너리 형태로 저장된 경우 (권장 방식)
            obj.model = data["model"]
            obj.features = data.get("features")
        else:
            # 모델 객체만 저장된 경우 (비권장 방식)
            obj.model = data
            # features 정보는 로컬 파일에서 로드되지 않았으므로 None으로 설정
            obj.features = None
            print("경고: 로드된 아티팩트가 'model' 키를 포함하는 딕셔너리가 아닙니다. 저장 로직을 확인하세요.")

        print(f"LGBM model loaded from {model_file_path}")
        return obj
    
    # MLFLOW production model load
    @classmethod
    def MLFLOW_load(cls) -> "LGBMRecommender":
        model_name = 'spotipy_LGBM'

        os.environ["MLFLOW_ADDR"] = "http://114.203.195.166:5000"
        os.environ["MLFLOW_S3_ENDPOINT_URL"] = "http://114.203.195.166:9000"
        os.environ["AWS_ACCESS_KEY_ID"] = "admin"
        os.environ["AWS_SECRET_ACCESS_KEY"] = "admin1234"

        mlflow_addr = os.environ.get("MLFLOW_ADDR")
        
        mlflow.set_tracking_uri(f"{mlflow_addr}")
        mlflow.set_experiment("spotipy_recommender")
        
        # 모든 버전 검색
        all_versions = mlflow.search_model_versions(filter_string =f"name='{model_name}'")

        # tag 'state'가 'production'인 버전만 필터
        prod_versions = [v for v in all_versions if v.tags.get("status") == "production"]

        # FIX: 모델 버전이 없는 경우 예외 처리
        if not prod_versions:
              raise RuntimeError(f"MLflow 등록 모델 '{model_name}'의 Production 버전을 찾을 수 없습니다. 모델 스테이터스를 확인하세요.")
              
        model_uri = f"models:/spotipy_LGBM/{prod_versions[0].version}"
        
        # artifacts를 로컬 경로로 다운로드
        local_path = download_artifacts(
            artifact_uri=model_uri + "/input_example.json"
        )
        
        result = LGBMRecommender()
        result.model_version = prod_versions[0].version
        model = mlflow.lightgbm.load_model(model_uri)
        with open(local_path) as f:
            features = json.load(f)
        result.features = features['columns']
        result.model = model
        return result

# -------------------- [통합 모델 (TopN_Model) 추가] --------------------

class TopN_Model:
    """
    Finder, FAISSRecommender, LGBMRecommender를 통합하여
    유사 항목 검색 및 재순위(Re-ranking)를 수행하는 최종 추천 모델입니다.
    """
    def __init__(self, finder: Finder, faiss_recommender: FAISSRecommender, 
                 lgbm_recommender: LGBMRecommender, data_df: pd.DataFrame):
        self.finder = finder
        self.faiss_rec = faiss_recommender
        self.lgbm_rec = lgbm_recommender
        self.data_df = data_df # 전체 데이터프레임 (FAISS의 df와 동일해야 함)

        if self.lgbm_rec and not self.lgbm_rec.features:
            # LGBM 모델이 로드되었지만 (로컬 로드가 아닌 경우) 피처 목록이 없는 경우 로드 시도
            try:
                self.lgbm_rec.MLFLOW_load_features()
            except Exception as e:
                print(f"LGBM features load failed: {e}")
            
    def Search(self, by: str, query: str, top_k: int = 10) -> pd.DataFrame:
        """
        주어진 쿼리를 기반으로 FAISS를 통해 유사 트랙을 찾고, 
        LGBM으로 재순위를 매겨 최종 추천 결과를 반환합니다.
        
        Args:
            by (str): 검색 기준 (track_id, track_name 등)
            query (str): 검색어 (시드 트랙 ID 등)
            top_k (int): 최종 반환할 추천 개수
            
        Returns:
            pd.DataFrame: 재순위된 최종 추천 결과 (track_id, track_name, score 등 포함)
        """
        
        # 1. FAISS를 이용한 후보 트랙 검색
        # LGBM 모델이 FAISS 결과를 재순위하기 때문에, 
        # FAISS에서는 top_k보다 더 많은 후보(예: top_k * 5 또는 50)를 가져와야 합니다.
        FAISS_CANDIDATE_COUNT = max(50, top_k * 5)
        
        # FAISSRecommender의 recommend 메서드를 사용하여 후보 트랙 검색
        # 참고: 이 메서드는 내부적으로 시드 트랙을 찾고, 유사한 트랙 리스트를 DataFrame으로 반환합니다.
        candidate_df = self.faiss_rec.recommend(
            by=by, 
            query=query, 
            top_k=FAISS_CANDIDATE_COUNT
        )
        
        if candidate_df.empty:
            return candidate_df # 결과가 없으면 빈 DataFrame 반환
        
        # 2. LGBM 재순위 모델용 피처 준비
        # LGBM의 피처 목록에 'distance' 피처가 포함될 것으로 가정하고 로직 작성
        
        lgbm_features = self.lgbm_rec.features
        
        if lgbm_features is None:
            # LGBM 피처가 로드되지 않은 경우, FAISS 결과만 반환 (재순위 없음)
            print("⚠️ LGBM features not available. Returning only FAISS results.")
            return candidate_df.head(top_k)

        # LGBM 예측에 필요한 피처를 candidate_df에서 추출합니다.
        # FAISS 결과에는 오디오 피처와 'distance'가 포함되어 있습니다.
        
        # FAISS 결과를 LGBM 입력 형식에 맞춥니다.
        # LGBM은 일반적으로 FAISS의 distance를 입력 피처 중 하나로 사용합니다.
        X_rerank = candidate_df.copy()

        # LGBM 모델 예측
        # LGBM 모델은 예측 점수(예: 인기도, 클릭 확률 등)를 반환합니다. 
        # 점수가 높을수록 추천 순위가 높아야 하므로, 재순위 기준을 예측값으로 합니다.
        try:
            # LGBM 모델의 predict 메서드는 피처 이름 목록(self.features)을 사용하여 예측합니다.
            X_rerank['predicted_score'] = self.lgbm_rec.predict(X_rerank)
        except Exception as e:
            print(f"❌ LGBM prediction failed: {e}. Returning FAISS results.")
            return candidate_df.head(top_k)

        # 3. 재순위 및 최종 결과 반환
        # predicted_score를 내림차순으로 정렬합니다. (점수가 높을수록 상위 추천)
        reranked_df = X_rerank.sort_values(by='predicted_score', ascending=False)
        
        # 최종 Top-K 결과만 선택
        final_result = reranked_df.head(top_k).reset_index(drop=True)
        
        # 순위 업데이트
        final_result['rank'] = final_result.index + 1
        
        # 불필요한 LGBM 예측 피처 제거
        final_result = final_result.drop(columns=['predicted_score'], errors='ignore')
        
        return final_result
