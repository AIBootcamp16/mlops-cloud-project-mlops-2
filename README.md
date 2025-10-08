🎧 사용자 선호도 기반 음악 추천 시스템

FastAPI + Streamlit + Airflow + MLflow 기반의 MLOps 음악 추천 서비스

💻 프로젝트 소개
🎧 Seed-based Music Recommendation System

사용자가 검색으로 노래 5곡을 선택하면, 해당 곡들의 **오디오 특성(Audio Features)**을 기반으로
콘텐츠 기반 추천(Content-based Filtering)을 수행하여 맞춤형 추천 리스트를 제공합니다.

Spotify API를 통해 검색 / 메타데이터 / 오디오 특성을 가져오며,
추천 알고리즘은 **코사인 유사도(Cosine Similarity)**를 기반으로 합니다.

✨ 주요 기능

🔍 검색(Search): Spotify API를 통한 트랙 검색

🎶 Seed Selection: 사용자가 좋아하는 노래 5곡 선택

🧩 프로필 벡터 생성: 선택한 곡의 오디오 특성을 평균화하여 사용자 프로필 구성

📊 추천 리스트 생성: 코사인 유사도로 후보 카탈로그와 비교하여 Top-K 추천

🎧 UI 제공: Streamlit으로 간단한 웹 인터페이스

⚡ API 제공: FastAPI 기반 음악 추천 API

🧠 MLflow / Airflow 연동: 모델 관리 및 파이프라인 실행

👨‍👩‍👦‍👦 팀 구성원
	
	
	
	

김소은
	김재록
	김종화
	최보경
	황은혜

팀장, 담당 역할	담당 역할	담당 역할	담당 역할	담당 역할
🔨 개발 환경 및 기술 스택
분류	기술
언어	Python 3.10
웹 프레임워크	FastAPI, Streamlit
MLOps 도구	MLflow, Airflow
모델 라이브러리	LightGBM, FAISS
데이터 처리	Pandas, Spotipy
환경 관리	Docker, Docker Compose
버전 관리	Git, GitHub
📂 프로젝트 구조 (최신)
mlops-cloud-project-mlops-2/
├─ .env                     # 실제 실행용 환경 변수 (커밋 금지)
├─ .env.template            # 팀원용 예시 (Spotify ID/Secret 직접 입력)
├─ .env.safe                # Spotify 없이 서버 부팅용 안전모드
├─ .dockerignore            # Docker 빌드 제외 목록
├─ Dockerfile               # FastAPI 서버용
├─ Dockerfile.ui            # Streamlit UI용
├─ Dockerfile.airflow       # Airflow 전용 컨테이너
├─ Dockerfile.mlflow        # MLflow 전용 컨테이너
├─ docker-compose.yml       # API + UI + Airflow + MLflow 통합 구성
├─ requirements_api.txt     # FastAPI 서버 의존성
├─ requirements_ui.txt      # Streamlit UI 의존성
├─ requirements_airflow.txt # Airflow 의존성
├─ requirements_mlflow.txt  # MLflow 의존성
├─ dataset/
│   ├─ raw/spotify_data.csv              # 공유받은 원본 데이터
│   └─ processed/spotify_data_clean.csv  # 공유받은 전처리 데이터
├─ models/                               # 학습된 모델 아티팩트
├─ src/
│   ├─ main.py                           # FastAPI 엔트리포인트
│   ├─ api/
│   ├─ model/
│   ├─ data/
│   ├─ utils/
│   └─ web/
└─ web/                                  # Streamlit UI 소스

🚀 실행 가이드 (for 개발자 및 팀원)

⚙️ Docker Compose 기반으로 실행되며,
팀원은 CSV 2개만 공유받으면 git pull → docker compose up으로 완전 실행 가능합니다.

1️⃣ 사전 준비
필수 조건

Docker 및 Docker Compose 설치

Python 설치 불필요 (모든 종속성은 컨테이너 내 자동 설치)

클론
git clone https://github.com/<your_team>/<your_repo>.git
cd mlops-cloud-project-mlops-2

2️⃣ 데이터 추가

팀원에게 아래 두 개 파일을 전달받아 폴더에 추가하세요:

dataset/raw/spotify_data.csv
dataset/processed/spotify_data_clean.csv


이 두 파일만 추가하면 전체 서비스가 정상 작동합니다 ✅

3️⃣ 환경 설정
✅ (A) 안전모드 (Spotify 인증 없이 서버만 부팅)

Spotify Client ID/Secret이 없어도 /health 및 UI 정상 동작

cp .env.safe .env
docker compose up -d --build

✅ (B) 전체기능 모드 (Spotify 이미지/메타데이터 포함)

Spotify Developer Dashboard에서 발급받은 Client ID/Secret을 .env에 입력

cp .env.template .env
# .env 파일 열어 SPOTIPY_CLIENT_ID / SPOTIPY_CLIENT_SECRET 입력
docker compose up -d --build

4️⃣ 서비스 확인
서비스	주소
FastAPI Health Check	http://localhost:8000/health

Streamlit UI	http://localhost:8501

MLflow Tracking	http://localhost:5000

Airflow Web UI	http://localhost:8080
5️⃣ 종료 및 로그 확인
docker compose ps
docker compose logs -f music_recommender_api
docker compose down

6️⃣ 주요 실행 파일 설명
파일	역할
Dockerfile	FastAPI 서버용 (3단계 빌드 구조)
Dockerfile.ui	Streamlit UI 빌드 전용
Dockerfile.airflow	Airflow 워크플로 관리용 (DAG 실행 및 웹 UI 포함)
Dockerfile.mlflow	MLflow 실험 관리 서버용 (Tracking & Artifacts)
docker-compose.yml	전체 서비스 통합 실행 (API/UI/Airflow/MLflow/Runner)
.dockerignore	Docker 빌드 제외 목록
.env.safe	Spotify 인증 없이 서버 부팅용
.env.template	Spotify 인증 포함 개발용 예시
requirements_api.txt	FastAPI 의존성 (LightGBM/FAISS 포함)
requirements_ui.txt	Streamlit UI 의존성 (Spotipy 포함)
requirements_airflow.txt	Airflow 환경 의존성
requirements_mlflow.txt	MLflow 환경 의존성
7️⃣ 빠른 커밋 예시
git add -A
git commit -m "READY: Full reproducible setup — CSV-only auto execution + env templates + Airflow/MLflow"
git push origin main

💻 구현 기능
기능1

작품에 대한 주요 기능을 작성해주세요

기능2

작품에 대한 주요 기능을 작성해주세요

기능3

작품에 대한 주요 기능을 작성해주세요

🛠️ 작품 아키텍처 (선택)

🚨 트러블 슈팅 (예시)
1. Spotify API 인증 실패 시

.env.safe 모드로 실행하면 FastAPI는 정상적으로 부팅됨

단, 이미지 URL 및 메타데이터는 비활성화 상태로 표시됨

📌 프로젝트 회고
박패캠

프로젝트 회고를 작성해주세요

📰 참고자료

Spotify API Docs

FastAPI

Streamlit

MLflow

Apache Airflow