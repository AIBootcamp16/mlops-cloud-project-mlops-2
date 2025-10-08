# =======================================================
# 🎵 FastAPI 기반 음악 추천 서버 (Production-ready)
# =======================================================

FROM python:3.10-slim

# 작업 디렉토리 설정
WORKDIR /app

# 1️⃣ 시스템 패키지 설치 (LightGBM, FAISS 등 필요 라이브러리)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 curl && \
    rm -rf /var/lib/apt/lists/*

# 2️⃣ 의존성 설치 (requirements_api.txt만 복사 → 캐시 효율 ↑)
COPY requirements_api.txt .
RUN pip install --no-cache-dir -r requirements_api.txt

# 3️⃣ 애플리케이션 코드 복사
COPY src ./src
COPY models ./models
COPY dataset/processed ./dataset/processed
COPY .env ./.env

# 4️⃣ FastAPI 서버 포트
EXPOSE 8000

# 5️⃣ 실행 명령 (main.py 진입점)
CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]
