import numpy as np
import pandas as pd

# 데이터 로드
df = pd.read_csv('../dataset/processed/spotify_data_clean.csv')
target_columns = ["popularity", "danceability", "energy", "key", "loudness",]
df = df[target_columns].drop_duplicates()
data = df.values

# 데이터 분할
np.random.shuffle(data)
split = int(len(data) * 0.8)
train_data = data[:split]
val_data = data[split:]

# 신경망 모델 정의
class SimpleNN:
    def __init__(self, input_dim, hidden_dim):
        self.weights1 = np.random.randn(input_dim, hidden_dim) * 0.01
        self.bias1 = np.zeros((1, hidden_dim))
        self.weights2 = np.random.randn(hidden_dim, 1) * 0.01
        self.bias2 = np.zeros((1, 1))

    def relu(self, x):
        return np.maximum(0, x)

    def forward(self, x):
        self.z1 = np.dot(x, self.weights1) + self.bias1
        self.a1 = self.relu(self.z1)
        self.z2 = np.dot(self.a1, self.weights2) + self.bias2
        return self.z2

    def backward(self, x, y, output, lr=0.001):
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

# 데이터 준비
X_train = train_data[:, :2]
y_train = train_data[:, 2].reshape(-1, 1)
X_val = val_data[:, :2]
y_val = val_data[:, 2].reshape(-1, 1)

# 모델 초기화
model = SimpleNN(input_dim=2, hidden_dim=64)

# 학습 루프
epochs = 15
for epoch in range(epochs):
    output = model.forward(X_train)
    train_loss = np.mean((output - y_train) ** 2)

    model.backward(X_train, y_train, output)

    val_output = model.forward(X_val)
    val_loss = np.mean((val_output - y_val) ** 2)

    print(f"Epoch {epoch + 1}/{epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")