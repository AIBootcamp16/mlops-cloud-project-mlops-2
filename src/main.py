import numpy as np
import pandas as pd
import joblib 
import os     
import mlflow 
import mlflow.pyfunc 

# nn_model.py 파일에서 SimpleNN 클래스를 임포트합니다.
# MLflow 아티팩트 저장 시 이 클래스의 정의를 찾을 수 있도록 합니다.
from src.nn_model import SimpleNN 


# MLflow 설정
# MLOps 환경에서 MLflow 서버가 실행 중이라고 가정하고, Experiment 이름을 설정합니다.
# 기본 추적 URI는 로컬 파일(file:./mlruns)로 설정됩니다.
mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", "file:./mlruns"))
mlflow.set_experiment("Simple_NN_Spotify_Prediction")


# 데이터 로드
df = pd.read_csv('dataset/processed/spotify_data_clean.csv')
target_columns = ["popularity", "danceability", "energy", "key", "loudness",]
df = df[target_columns].drop_duplicates()
data = df.values

# 데이터 분할
np.random.shuffle(data)
split = int(len(data) * 0.8)
train_data = data[:split]
val_data = data[split:]

# 데이터 준비
# 예시 NN은 첫 두 피처(danceability, energy)를 입력으로 사용하고
# 세 번째 피처(key)를 예측한다고 가정합니다.
X_train = train_data[:, :2]
y_train = train_data[:, 2].reshape(-1, 1)
X_val = val_data[:, :2]
y_val = val_data[:, 2].reshape(-1, 1)

# 모델 설정
input_dim = X_train.shape[1]
hidden_dim = 64
epochs = 15
learning_rate = 0.001 # 학습률을 변수로 정의

# 모델 초기화 (SimpleNN은 이제 src/nn_model.py에서 import 됨)
model = SimpleNN(input_dim=input_dim, hidden_dim=hidden_dim)


# ==========================================================
# MLflow 훈련 로깅 시작
# ==========================================================
with mlflow.start_run(run_name="Simple_NN_Training"):
    
    # 1. 하이퍼파라미터 로깅
    mlflow.log_param("input_dim", input_dim)
    mlflow.log_param("hidden_dim", hidden_dim)
    mlflow.log_param("epochs", epochs)
    mlflow.log_param("learning_rate", learning_rate)

    # 2. 학습 루프 및 메트릭 로깅
    print("--- Training Started (Logging to MLflow) ---")
    for epoch in range(epochs):
        output = model.forward(X_train)
        train_loss = np.mean((output - y_train) ** 2)

        # 정의된 learning_rate 변수를 backward 메서드에 전달
        model.backward(X_train, y_train, output, lr=learning_rate)

        val_output = model.forward(X_val)
        val_loss = np.mean((val_output - y_val) ** 2)

        # 에폭별 손실 로깅
        mlflow.log_metric("train_loss", train_loss, step=epoch)
        mlflow.log_metric("val_loss", val_loss, step=epoch)

        print(f"Epoch {epoch + 1}/{epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

    # 최종 성능 로깅
    mlflow.log_metric("final_val_loss", val_loss)

    # ==========================================================
    # 3. 훈련된 모델 아티팩트 저장 및 MLflow 로깅
    # ==========================================================
    MODEL_DIR = "models"
    MODEL_PATH = os.path.join(MODEL_DIR, "simple_nn_model.joblib")

    # models 디렉토리 생성
    os.makedirs(MODEL_DIR, exist_ok=True)

    # 모델의 가중치와 바이어스를 딕셔너리 형태로 저장
    model_artifacts = {
        "weights1": model.weights1,
        "bias1": model.bias1,
        "weights2": model.weights2,
        "bias2": model.bias2,
        "target_columns": target_columns 
    }

    # joblib을 사용하여 디스크에 저장
    joblib.dump(model_artifacts, MODEL_PATH)

    # joblib 파일을 MLflow 아티팩트로 로깅
    mlflow.log_artifact(MODEL_PATH)
    
    print(f"\n✅ Model successfully saved to {MODEL_PATH}")
    print(f"✅ Model artifacts logged to MLflow Run ID: {mlflow.active_run().info.run_id}")
# with 블록이 끝나면 run이 자동 종료됩니다.
