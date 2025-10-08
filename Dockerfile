# ============================================================
# 🧠 MLOps Music Recommender - FastAPI Service
# ------------------------------------------------------------
#  ✅ 3단계 빌드 구조
#     1️⃣ builder   - Python 의존성 설치
#     2️⃣ artifact  - 코드 및 모델 복사
#     3️⃣ runtime   - 최소 실행 환경
# ============================================================

# ---------- 1️⃣ Builder ----------
FROM python:3.10-slim AS builder

WORKDIR /install

# LightGBM, FAISS, Requests 등 필수 시스템 패키지
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential libgomp1 curl && \
    rm -rf /var/lib/apt/lists/*

# requirements 캐시 빌드 (의존성 고정)
COPY requirements_api.txt .
RUN pip install --no-cache-dir -r requirements_api.txt


# ---------- 2️⃣ Artifact ----------
FROM python:3.10-slim AS artifact
WORKDIR /app/mlops

# builder 단계에서 설치된 패키지 복사
COPY --from=builder /usr/local/lib/python3.10/site-packages /usr/local/lib/python3.10/site-packages

# 코드 및 모델 복사
COPY src ./src
COPY main.py ./main.py
COPY models ./models
COPY .env ./.env

# 데이터셋은 제외 (로컬 볼륨 마운트로 제공)
RUN mkdir -p dataset/raw dataset/processed


# ---------- 3️⃣ Runtime ----------
FROM python:3.10-slim

WORKDIR /app/mlops

# Python 패키지 복사
COPY --from=artifact /usr/local/lib/python3.10/site-packages /usr/local/lib/python3.10/site-packages
COPY --from=artifact /app/mlops /app/mlops

EXPOSE 8000

# 컨테이너 기본 명령
CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]
