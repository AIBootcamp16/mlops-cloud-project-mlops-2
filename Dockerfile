# =========================================================
# 🧠 MLOps Music Recommender - FastAPI Service (Optimized)
# ---------------------------------------------------------
# 3단계 빌드 전략:
#   1️⃣ builder   - 패키지 캐시 최적화
#   2️⃣ artifact  - 모델/데이터 포함
#   3️⃣ runtime   - 경량화 실행 이미지
# =========================================================

# ---------- 1️⃣ Builder ----------
FROM python:3.10-slim AS builder

WORKDIR /install

# 필수 시스템 패키지 (LightGBM / FAISS / Requests)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 curl && rm -rf /var/lib/apt/lists/*

# 의존성 설치 (requirements_api.txt)
COPY requirements_api.txt .
RUN pip install --no-cache-dir -r requirements_api.txt


# ---------- 2️⃣ Artifact ----------
FROM python:3.10-slim AS artifact

WORKDIR /app/mlops
COPY --from=builder /usr/local/lib/python3.10/site-packages /usr/local/lib/python3.10/site-packages

# 코드 복사
COPY src ./src
COPY main.py ./main.py
COPY models ./models
COPY .env ./.env

# 데이터 (대용량 CSV는 제외)
RUN mkdir -p dataset/processed && mkdir -p dataset/raw


# ---------- 3️⃣ Runtime ----------
FROM python:3.10-slim

WORKDIR /app/mlops

# 필수 패키지 복사
COPY --from=artifact /usr/local/lib/python3.10/site-packages /usr/local/lib/python3.10/site-packages
COPY --from=artifact /app/mlops /app/mlops

EXPOSE 8000

CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]
