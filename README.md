
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
```
mlops-cloud-project-mlops-2/
├── dataset/
│ ├─ raw/ # Spotify 원본 데이터 (Git 업로드 제외)
│ └─ processed/ # 전처리된 데이터 (로컬 유지)
│
├── models/ # 학습된 모델 및 Finder 아티팩트
│
├── src/ # 애플리케이션 주요 코드
│ ├─ main.py # FastAPI 서버 엔트리포인트
│ ├─ api/api.py # API 라우팅 및 엔드포인트 정의
│ ├─ web/streamlit_app.py # Streamlit UI 실행 스크립트
│ ├─ model/ # 추천 모델 (FAISS, LightGBM 등)
│ ├─ data/build_dataset.py # 데이터 전처리 및 파이프라인
│ ├─ utils/ # 공용 유틸리티 함수
│ └─ tests/ # 테스트 코드
│
├── dags/ # Airflow DAG 정의 (워크플로 관리)
├── tmp/ # 임시 디렉토리 (Airflow/MLflow 캐시)
│
├── Dockerfile.api # 🎵 FastAPI (Backend) 빌드용
├── Dockerfile.ui # 🖥️ Streamlit (Frontend) 빌드용
├── Dockerfile.airflow # ☁️ Airflow 컨테이너 빌드용
├── Dockerfile.mlflow # 🧪 MLflow Tracking Server 빌드용
│
├── docker-compose.yml # 전체 서비스 통합 실행 설정
│
├── .github/
│ └─ workflows/
│ ├─ ci-cd.yml # main 브랜치용 CI (테스트 빌드)
│ └─ build.yml # 태그 기반 CD (배포 빌드)
│
├── docs/
│ └─ ci-cd-setup.md # CI/CD 상세 설정 가이드 (README 부록 버전)
│
├── .env # 로컬 환경 변수 (Git 미포함)
├── .env.safe # Spotify 인증 없이 실행용
├── .env.template # Spotify API 인증 템플릿
│
├── requirements_api.txt # FastAPI 의존성
├── requirements_ui.txt # Streamlit 의존성
│
├── .dockerignore # Docker 빌드 제외 규칙
├── .gitignore # Git 제외 규칙
└── README.md # 📘 프로젝트 문서 (현재 파일)

💡 Airflow는 Dockerfile.airflow 내부에서 의존성을 직접 설치합니다.
CI/CD 파이프라인 설정은 .github/workflows 디렉토리에서 관리됩니다.
```

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

## 1️⃣ 환경 변수 설정  

Spotify 인증 없이 실행하려면:

- cp .env.safe .env 
- `.env.safe`를 `.env`로 복사 시 **Spotify 인증 없이 서버 부팅 가능**

Spotify API를 사용하려면:

- cp .env.template .env
- 실제 Spotify API 사용 시 `.env.template`의 Client ID/Secret 추가

## 2️⃣ FastAPI 서버 실행

- docker build -t music_api -f Dockerfile .
- docker run -p 8000:8000 music_api

## 3️⃣ Streamlit UI 실행

- docker build -t music_ui -f Dockerfile.ui .
- docker run -p 8501:8501 music_ui

## 4️⃣ 전체 서비스 통합 실행

- docker compose up -d --build
- FastAPI, Streamlit, Airflow, MLflow 컨테이너가 함께 구동됩니다.

<br>

### ⚙️ 빌드 최적화
3단계 빌드 전략 (builder → artifact → runtime)

- dataset/*.csv 제외로 빌드 시간 단축

- requirements_* 캐시 고정으로 3분 → 10초

- API/UI 분리로 컨테이너 효율 향상

<br>

### 🚨 트러블 슈팅
1. Spotify API 인증 실패
    - .env에 Client ID / Secret 누락 시 발생
    - 해결 :
      SPOTIPY_CLIENT_ID=<your_id>
      SPOTIPY_CLIENT_SECRET=<your_secret>
2. FastAPI 서버 부팅 실패
    - spotify_data_clean.csv 미존재 시 발생
    - 해결 :
      dataset/processed/spotify_data_clean.csv 추가 후 재빌드
3. Docker 빌드 지연
    - 캐시 미사용 또는 dataset 복사 포함
    - 해결 :
      docker compose build --no-cache
      
<br>

### 📌 프로젝트 회고

- Docker 기반 통합 환경으로 실행 안정성 확보
- 빌드 캐시 최적화로 개발 효율 향상
- Airflow / MLflow 연동으로 MLOps 자동화 기반 마련

<br>

### 📰 참고자료
- Spotify Web API Documentation
- FastAPI Official Docs
- Streamlit Docs
- MLflow Docs
- Apache Airflow Docs

 
---

## ⚙️ CI/CD 환경 설정 가이드 (GitHub Actions)

이 프로젝트는 GitHub Actions를 통해 자동으로 빌드 및 배포됩니다.  
Secrets를 등록하면 main 브랜치 Push 시에는 **CI(테스트 빌드)**,  
버전 태그 Push 시에는 **CD(배포 빌드)** 가 자동 실행됩니다.

### 🔐 등록해야 할 Secrets
| 이름 | 설명 |
|------|------|
| `SPOTIPY_CLIENT_ID` | Spotify API Client ID |
| `SPOTIPY_CLIENT_SECRET` | Spotify API Secret |
| `AWS_ACCESS_KEY_ID` | MinIO 또는 S3 Access Key |
| `AWS_SECRET_ACCESS_KEY` | MinIO 또는 S3 Secret Key |
| `MLFLOW_ADDR` | MLflow Tracking 서버 주소 |

> `.env` 파일은 Git에 포함하지 않습니다.  
> CI/CD 실행 중에 이 Secrets 값이 자동으로 `.env`로 주입됩니다.

### 🚀 실행 트리거
| 트리거 | 동작 |
|--------|------|
| `push → main` | FastAPI / Streamlit 빌드 및 헬스체크 (CI) |
| `push → tag (v1.0.0-stable 등)` | Docker 이미지 빌드 및 GHCR 배포 (CD) |

```bash
# 예시
git push origin main          # CI 자동 실행
git tag v1.0.0-stable
git push origin v1.0.0-stable # CD 자동 실행



