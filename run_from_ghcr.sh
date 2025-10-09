#!/usr/bin/env bash
set -e

# ============================================================
# 🚀 run_from_ghcr.sh (Auto latest GHCR version)
# ------------------------------------------------------------
# 기능:
#   - GHCR에서 최신 태그 자동 감지
#   - FastAPI / Streamlit 컨테이너 실행
# ============================================================

REGISTRY="ghcr.io/gogoaihunter"
IMAGE_API="mlops-cloud-project-mlops-2-api"
IMAGE_UI="mlops-cloud-project-mlops-2-ui"

# 1️⃣ 최신 버전 자동 감지
echo "🔍 GHCR에서 최신 버전 확인 중..."
VERSION=$(curl -s "https://ghcr.io/v2/gogoaihunter/$IMAGE_API/tags/list" | \
           grep -oE '"v[0-9]+\.[0-9]+\.[0-9]+"' | tail -1 | tr -d '"')

if [[ -z "$VERSION" ]]; then
  echo "❌ 최신 버전 정보를 가져오지 못했습니다. (GHCR 로그인 필요?)"
  exit 1
fi

echo "✅ 최신 버전: $VERSION"

# 2️⃣ GHCR 로그인 확인
if ! docker info | grep -q "ghcr.io"; then
  echo "⚠️  GHCR 로그인 필요:"
  echo "👉 docker login ghcr.io -u <your-username> -p <your-token>"
  exit 1
fi

# 3️⃣ FastAPI 컨테이너
echo "🐍 FastAPI 이미지 실행 중..."
docker pull $REGISTRY/$IMAGE_API:$VERSION
docker rm -f music_recommender_api 2>/dev/null || true
docker run -d --name music_recommender_api -p 8000:8000 --env-file .env $REGISTRY/$IMAGE_API:$VERSION

# 4️⃣ Streamlit 컨테이너
echo "🎨 Streamlit UI 이미지 실행 중..."
docker pull $REGISTRY/$IMAGE_UI:$VERSION
docker rm -f music_recommender_ui 2>/dev/null || true
docker run -d --name music_recommender_ui -p 8501:8501 --env-file .env $REGISTRY/$IMAGE_UI:$VERSION

# 5️⃣ 완료 메시지
echo ""
echo "✅ 최신 버전($VERSION) 컨테이너 실행 완료!"
echo "---------------------------------------------"
echo "FastAPI:   http://localhost:8000/docs"
echo "Streamlit: http://localhost:8501"
echo "---------------------------------------------"
