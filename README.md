# 🎧 사용자 선호도 기반 음악 추천 시스템

<br>

## 💻 프로젝트 소개
### 🎵 Seed-based Music Recommendation System

사용자가 **검색으로 노래 5곡을 선택**하면, 해당 곡들의 **오디오 특성(Audio Features)**을 기반으로  
콘텐츠 기반 추천(Content-based Filtering)을 수행하여 **맞춤형 추천 리스트**를 제공합니다.  

Spotify API를 활용하여 **검색 / 메타데이터 / 오디오 특성**을 가져오며,  
추천 알고리즘은 **코사인 유사도(Cosine Similarity)**를 기반으로 합니다.  

<br>

## ✨ Features
- 🔍 **검색(Search)**: Spotify API를 통한 트랙 검색  
- 🎶 **Seed Selection**: 사용자가 좋아하는 노래 5곡 선택  
- 🧩 **프로필 벡터 생성**: 선택한 곡의 오디오 특성을 평균화하여 사용자 프로필 구성  
- 📊 **추천 리스트 생성**: 코사인 유사도로 후보 카탈로그와 비교하여 Top-K 추천  
- 🎧 **UI 제공**: Streamlit으로 간단한 웹 UI  
- ⚙️ **API 제공**: FastAPI 기반 추천 API  
- 🧠 **MLOps 통합**: Airflow + MLflow를 통한 파이프라인 및 모델 관리  
- 🐳 **Docker 지원**: 전체 시스템을 Docker Compose로 통합 실행 가능  

<br>

## 👨‍👩‍👦‍👦 팀 구성원

| ![김소은](https://avatars.githubusercontent.com/u/156163982?v=4) | ![김재록](https://avatars.githubusercontent.com/u/156163982?v=4) | ![김종화](https://avatars.githubusercontent.com/u/156163982?v=4) | ![최보경](https://avatars.githubusercontent.com/u/156163982?v=4) | ![황은혜](https://avatars.githubusercontent.com/u/156163982?v=4) |
| :--------------------------------------------------------------: | :--------------------------------------------------------------: | :--------------------------------------------------------------: | :--------------------------------------------------------------: | :--------------------------------------------------------------: |
| [김소은](https://github.com/oriori88) | [김재록](https://github.com/UpstageAILab) | [김종화](https://github.com/UpstageAILab) | [최보경](https://github.com/UpstageAILab) | [황은혜](https://github.com/UpstageAILab) |
| 팀장, 담당 역할 | 담당 역할 | 담당 역할 | 담당 역할 | 담당 역할 |

<br>

## 🔨 개발 환경 및 기술 스택
- **언어**: Python 3.10  
- **웹 프레임워크**: FastAPI, Streamlit  
- **MLOps 도구**: MLflow, Airflow  
- **ML 라이브러리**: LightGBM, FAISS  
- **데이터 처리**: Pandas, Spotipy  
- **환경 관리**: Docker, Docker Compose  
- **버전 관리**: Git, GitHub  

<br>

## 📁 프로젝트 구조
mlops-cloud-project-mlops-2/
├─ dataset/
│ ├─ raw/ # Spotify 원본 데이터 (Git 업로드 제외)
│ └─ processed/ # 전처리 데이터 (로컬 유지)
├─ models/ # 학습된 모델 아티팩트
├─ src/
│ ├─ main.py # FastAPI 서버 엔트리포인트
│ ├─ api/api.py # API 라우팅
│ ├─ web/streamlit_app.py # Streamlit UI
│ ├─ model/ # 모델 정의 (FAISS, LGBM, Finder 등)
│ ├─ data/build_dataset.py # 데이터 처리 파이프라인
│ ├─ utils/ # 유틸리티 함수
│ └─ tests/ # 테스트 코드
├─ Dockerfile # FastAPI 서버용
├─ Dockerfile.ui # Streamlit UI용
├─ Dockerfile.airflow # Airflow 컨테이너 (내부에서 pip install)
├─ Dockerfile.mlflow # MLflow 컨테이너
├─ docker-compose.yml # 전체 서비스 통합 구성
├─ .env / .env.safe / .env.template # 환경 설정 파일
├─ requirements_api.txt / requirements_ui.txt
└─ README.md

> 💡 Airflow는 `Dockerfile.airflow` 내부에서 패키지를 직접 설치합니다.

<br>

## 💻 구현 기능
### 기능1
- FastAPI 서버: `/search`, `/recommend_ranked` 엔드포인트 제공  
### 기능2
- Streamlit UI: 실시간 검색 및 추천 결과 시각화  
### 기능3
- MLflow / Airflow 통합으로 파이프라인 및 실험 관리  

<br>

## 🛠️ 작품 아키텍처 (Architecture)
- #### _아래 이미지는 예시입니다_

![이미지 설명](https://miro.medium.com/v2/resize:fit:4800/format:webp/1*ub_u88a4MB5Uj-9Eb60VNA.jpeg)

**구성요소**
- **FastAPI (Backend)**: 추천 로직 및 데이터 API  
- **Streamlit (Frontend)**: 사용자 인터페이스  
- **Airflow**: 데이터 수집 및 전처리 워크플로 관리  
- **MLflow**: 실험 추적 및 모델 버전 관리  
- **Docker Compose**: 전체 환경 통합 및 실행  

<br>

## 🚀 실행 방법

### 1️⃣ 환경 변수 설정  
- `.env.safe`를 `.env`로 복사 시 **Spotify 인증 없이 서버 부팅 가능**  
- 실제 Spotify API 사용 시 `.env.template`의 Client ID/Secret 추가  

```bash
cp .env.safe .env
2️⃣ FastAPI 서버 실행
bash
코드 복사
docker build -t music_api -f Dockerfile .
docker run -p 8000:8000 music_api
3️⃣ Streamlit UI 실행
bash
코드 복사
docker build -t music_ui -f Dockerfile.ui .
docker run -p 8501:8501 music_ui
4️⃣ 전체 서비스 통합 실행
bash
코드 복사
docker compose up -d --build
FastAPI, Streamlit, Airflow, MLflow 컨테이너가 함께 구동됩니다.

<br>
⚙️ 빌드 최적화
3단계 빌드 전략 (builder → artifact → runtime)

dataset/*.csv 제외로 빌드 시간 단축

requirements_* 캐시 고정으로 3분 → 10초

API/UI 분리로 컨테이너 효율 향상

<br>
🚨 트러블 슈팅
1. Spotify API 인증 실패
설명
.env에 Client ID / Secret 누락 시 발생

해결
bash
코드 복사
SPOTIPY_CLIENT_ID=<your_id>
SPOTIPY_CLIENT_SECRET=<your_secret>
2. FastAPI 서버 부팅 실패
설명
spotify_data_clean.csv 미존재 시 발생

해결
dataset/processed/spotify_data_clean.csv 추가 후 재빌드

3. Docker 빌드 지연
설명
캐시 미사용 또는 dataset 복사 포함

해결
bash
코드 복사
docker compose build --no-cache
<br>
📌 프로젝트 회고
Docker 기반 통합 환경으로 실행 안정성 확보

빌드 캐시 최적화로 개발 효율 향상

Airflow / MLflow 연동으로 MLOps 자동화 기반 마련

<br>
📰 참고자료
Spotify Web API Documentation

FastAPI Official Docs

Streamlit Docs

MLflow Docs

Apache Airflow Docs
