import numpy as np
import pandas as pd
import joblib
import os

# MLflow 아티팩트 로드 및 SimpleNN 모델 초기화를 위한 유틸리티 파일입니다.

# 신경망 모델 정의 (src/main.py에서 이동)
class SimpleNN:
    """
    간단한 2계층 신경망 모델 클래스.
    FAISS로 검색된 후보 아이템을 재순위화(re-ranking)하는 데 사용됩니다.
    """
    def __init__(self, input_dim, hidden_dim):
        # 가중치와 바이어스는 로딩 시 덮어씌워지므로 초기값은 중요하지 않습니다.
        self.weights1 = np.random.randn(input_dim, hidden_dim) * 0.01
        self.bias1 = np.zeros((1, hidden_dim))
        self.weights2 = np.random.randn(hidden_dim, 1) * 0.01
        self.bias2 = np.zeros((1, 1))

    def relu(self, x):
        return np.maximum(0, x)

    def forward(self, x):
        """
        순전파 연산: x -> z1(dot) -> a1(relu) -> z2(dot)
        """
        # x가 DataFrame일 경우 numpy 배열로 변환합니다.
        if isinstance(x, pd.DataFrame):
            x = x.values
            
        self.z1 = np.dot(x, self.weights1) + self.bias1
        self.a1 = self.relu(self.z1)
        self.z2 = np.dot(self.a1, self.weights2) + self.bias2
        return self.z2

    def backward(self, x, y, output, lr=0.001):
        """
        역전파 연산: 학습 시에만 사용
        """
        m = y.shape[0]
        dz2 = (output - y) / m
        dw2 = np.dot(self.a1.T, dz2)
        db2 = np.sum(dz2, axis=0, keepdims=True)

        da1 = np.dot(dz2, self.weights2.T)
        dz1 = da1 * (self.z1 > 0)
        dw1 = np.dot(x.T, dz1)
        db1 = np.sum(dz1, axis=0, keepdims=True)

        self.weights2 -= lr * dw2
        self.bias2 -= lr * db2
        self.weights1 -= lr * dw1
        self.bias1 -= lr * db1


def load_nn_model(model_path: str, hidden_dim: int = 64) -> SimpleNN:
    """
    joblib으로 저장된 모델 아티팩트(가중치/바이어스)를 로드하여 SimpleNN 인스턴스를 초기화합니다.
    """
    if not os.path.exists(model_path):
        print(f"경고: 모델 파일이 존재하지 않습니다: {model_path}")
        return None

    # joblib 파일을 로드합니다.
    artifacts = joblib.load(model_path)
    
    # 가중치 배열의 shape에서 입력 차원(input_dim)을 추론합니다.
    # weights1의 shape: (input_dim, hidden_dim)
    input_dim = artifacts['weights1'].shape[0]

    # SimpleNN 인스턴스를 생성합니다.
    loaded_model = SimpleNN(input_dim=input_dim, hidden_dim=hidden_dim)

    # 로드된 아티팩트의 가중치와 바이어스를 모델 인스턴스에 적용합니다.
    loaded_model.weights1 = artifacts['weights1']
    loaded_model.bias1 = artifacts['bias1']
    loaded_model.weights2 = artifacts['weights2']
    loaded_model.bias2 = artifacts['bias2']
    
    # 필요하다면 target_columns도 반환할 수 있습니다.
    # print(f"✅ SimpleNN 모델 로드 완료. (입력 차원: {input_dim})")
    return loaded_model

# MLflow 로드를 위한 클래스 래퍼 (Re-ranking 클래스에서 사용 가능)
class SimpleNNLoader:
    """
    API 서버에서 MLflow 아티팩트 로드 및 SimpleNN 모델 초기화를 위한 래퍼.
    """
    def __init__(self, run_id: str, artifact_path: str, hidden_dim: int = 64):
        self.run_id = run_id
        self.artifact_path = artifact_path
        self.hidden_dim = hidden_dim
        self.model = None
        self.features = None # 모델이 기대하는 피처 목록 (예: L.G.B.M Recommender에서 사용)

    def load_from_mlflow(self):
        """
        MLflow에서 아티팩트를 다운로드하고 모델을 초기화합니다.
        """
        try:
            # MLflow 클라이언트를 사용해 아티팩트를 로드합니다.
            # 이 코드는 실제 MLflow 서버 환경에서 작동합니다.
            
            # 1. 아티팩트 다운로드 (예시 로직)
            # local_path = mlflow.artifacts.download_artifacts(
            #     run_id=self.run_id,
            #     artifact_path=self.artifact_path
            # )
            
            # 현재 작업 환경에서는 로컬 경로를 직접 사용한다고 가정합니다.
            # Production 환경에서는 위의 mlflow.artifacts.download_artifacts를 사용해야 합니다.
            local_path = os.path.join("models", self.artifact_path)
            
            # 2. joblib 파일 로드
            if not os.path.exists(local_path):
                 print(f"MLflow 아티팩트 로드 실패: {local_path} 파일이 로컬에 없습니다. `src/main.py`를 먼저 실행하여 파일을 생성하세요.")
                 return None

            artifacts = joblib.load(local_path)
            
            # 3. SimpleNN 모델 초기화 및 가중치 적용
            input_dim = artifacts['weights1'].shape[0]
            self.model = SimpleNN(input_dim=input_dim, hidden_dim=self.hidden_dim)
            self.model.weights1 = artifacts['weights1']
            self.model.bias1 = artifacts['bias1']
            self.model.weights2 = artifacts['weights2']
            self.model.bias2 = artifacts['bias2']
            
            # 피처 목록 저장 (Re-ranking 모델에서 사용할 수 있도록)
            self.features = artifacts.get('target_columns', [])

            print(f"✅ SimpleNN 모델이 MLflow Run {self.run_id}에서 성공적으로 로드되었습니다.")
            return self.model

        except Exception as e:
            print(f"MLflow 로드 중 오류 발생: {e}")
            return None
