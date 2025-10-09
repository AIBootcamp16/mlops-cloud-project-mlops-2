#!/usr/bin/env bash
set -e

# ============================================================
# 🚀 MLOps Cloud Project - GHCR Image Runner
# ------------------------------------------------------------
# 버전:
#   - v1.0.25 기준 (태그 변경 시 VERSION 변수만 수정)
# 기능:
#   - GHCR에서 FastAPI / Streamlit 이미지 Pull
#   - 로컬 포트 매핑 및 실행
# ============================================================

# ===== 기본 설정 =====
VERSION="v1.0.25"
REGISTRY="ghcr.io/gogoaihunter"
IMAGE_API="mlops-cloud-project-mlops-2-api"
IMAGE_UI="mlops-cloud-project-mlops-2-ui"

# ===== 사전 확인 =====
echo "🔎 Checking Docker login to GHCR..."
if ! docker info | grep -q "ghcr.io"; then
  echo "⚠️  GHCR 로그인 필요:"
  echo "👉 docker login ghcr.io -u <your-username> -p <your-token>"
  exit 1
fi

# ===== FastAPI 서버 =====
echo "🐍 Pulling FastAPI image..."
docker pull $REGISTRY/$IMAGE_API:$VERSION

echo "🚀 Starting FastAPI container..."
docker run -d \
  --name music_recommender_api \
  -p 8000:8000 \
  --env-file .env \
  $REGISTRY/$IMAGE_API:$VERSION

echo "⏳ Waiting for FastAPI to initialize..."
sleep 10
curl -s http://localhost:8000/health || echo "⚠️ API health endpoint not responding yet"

# ===== Streamlit UI =====
echo "🎨 Pulling Streamlit UI image..."
docker pull $REGISTRY/$IMAGE_UI:$VERSION

echo "🚀 Starting Streamlit container..."
docker run -d \
  --name music_recommender_ui \
  -p 8501:8501 \
  --env-file .env \
  $REGISTRY/$IMAGE_UI:$VERSION

# ===== 완료 메시지 =====
echo ""
echo "✅ All containers are up!"
echo "---------------------------------------------"
echo "FastAPI:   http://localhost:8000/docs"
echo "Streamlit: http://localhost:8501"
echo "---------------------------------------------"
echo "🧹 To stop and remove containers:"
echo "    docker stop music_recommender_api music_recommender_ui && docker rm music_recommender_api music_recommender_ui"
