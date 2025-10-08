# 🎧 사용자 선호도 기반 음악 추천 시스템

> **FastAPI + Streamlit + Airflow + MLflow 기반의 MLOps 음악 추천 서비스**

---

## 💡 프로젝트 소개
### 🎵 Seed-based Music Recommendation System

사용자가 **검색으로 노래 5곡을 선택**하면,  
해당 곡들의 **오디오 특성(Audio Features)**을 기반으로  
콘텐츠 기반 추천(Content-based Filtering)을 수행하여 **맞춤형 추천 리스트**를 제공합니다.  

Spotify API를 통해 **검색 / 메타데이터 / 오디오 특성**을 가져오며,  
추천 알고리즘은 **코사인 유사도(Cosine Similarity)**를 기반으로 합니다.  

---

## ✨ 주요 기능

- 🔍 **검색 (Search)**: Spotify API를 통한 트랙 검색  
- 🎶 **Seed Selection**: 사용자가 좋아하는 노래 5곡 선택  
- 🧩 **프로필 벡터 생성**: 선택한 곡의 오디오 특성을 평균화하여 사용자 프로필 구성  
- 📊 **추천 리스트 생성**: 코사인 유사도로 후보 카탈로그와 비교하여 Top-K 추천  
- 🖥️ **UI 제공**: Streamlit으로 간단한 웹 인터페이스  
- ⚙️ **API 제공**: FastAPI 기반 음악 추천 API  
- 🚀 **MLflow / Airflow 연동**: 모델 관리 및 파이프라인 실행  

---

## 👨‍💻 팀 구성원

| ![김소은](https://avatars.githubusercontent.com/u/156163982?v=4) | ![김재록](https://avatars.githubusercontent.com/u/156163982?v=4) | ![김종화](https://avatars.githubusercontent.com/u/156163982?v=4) | ![최보경](https://avatars.githubusercontent.com/u/156163982?v=4) | ![황은혜](https://avatars.githubusercontent.com/u/156163982?v=4) |
|:--------------------------------------------------------------:|:--------------------------------------------------------------:|:--------------------------------------------------------------:|:--------------------------------------------------------------:|:--------------------------------------------------------------:|
| [김소은](https://github.com/oriori88) | [김재록](https://github.com/UpstageAILab) | [김종화](https://github.com/UpstageAILab) | [최보경](https://github.com/UpstageAILab) | [황은혜](https://github.com/UpstageAILab) |
| 팀장, 담당 역할 | 담당 역할 | 담당 역할 | 담당 역할 | 담당 역할 |

---

## ⚙️ 개발 환경 및 기술 스택

- **언어**: Python 3.10  
- **웹 프레임워크**: FastAPI, Streamlit  
- **MLOps 도구**: MLflow, Airflow  
- **ML 라이브러리**: LightGBM, FAISS  
- **데이터 처리**: Pandas, Spotipy  
- **환경 관리**: Docker, Docker Compose  
- **버전 관리**: Git, GitHub  

---

## 📁 프로젝트 구조 (최신)

mlops-cloud-project-mlops-2/
├─ dataset/
│ ├─ raw/spotify_data.csv
│ └─ processed/spotify_data_clean.csv
├─ models/
├─ src/
│ ├─ main.py # FastAPI 서버 엔트리포인트
│ ├─ api/api.py # API 라우팅
│ ├─ model/ # 모델 정의 (FAISS, LGBM, Finder 등)
│ ├─ web/streamlit_app.py # Streamlit 프론트엔드
│ └─ data/build_dataset.py # 데이터 처리 파이프라인
├─ Dockerfile # FastAPI 서버용
├─ Dockerfile.ui # Streamlit UI용
├─ Dockerfile.airflow # Airflow 전용 컨테이너 (내부에서 pip install)
├─ Dockerfile.mlflow # MLflow 전용 컨테이너
├─ docker-compose.yml # 통합 실행 환경
├─ .env / .env.safe / .env.template
├─ requirements_api.txt / requirements_ui.txt
└─ README.md

yaml
코드 복사

> 💡 `requirements_airflow.txt`는 존재하지 않습니다.  
> Airflow는 `Dockerfile.airflow` 내부에서 직접 설치됩니다.

---

## 🚀 실행 방법

### 1️⃣ 환경 변수 설정
- `.env.safe`를 `.env`로 복사하면 **Spotify 인증 없이 서버 부팅 가능**
- Spotify 인증을 사용하려면 `.env.template`에 Client ID/Secret을 입력

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
FastAPI, Streamlit, Airflow, MLflow가 함께 실행됩니다.

⚡ 빌드 최적화 요약
3단계 Docker 빌드 전략 적용 (builder → artifact → runtime)

dataset/ 내 대용량 CSV 제외 (.dockerignore)

의존성 캐시 고정으로 빌드 속도 3분 → 10초 이내 단축

API / UI 분리로 개발 속도 및 안정성 향상

🧠 참고 및 운영 팁
.env.safe: Spotify 인증 없이 테스트용

.env.template: 실제 인증값 추가용

.dockerignore: dataset, logs, venv 등 빌드 제외

requirements_*.txt: 서비스별 최소 의존성 분리

Airflow는 Dockerfile.airflow 내부에서 패키지 직접 설치

🧩 트러블슈팅
문제	원인	해결 방법
FastAPI 서버 부팅 실패	dataset 누락	dataset/processed/spotify_data_clean.csv 추가
Spotify API 인증 실패	환경 변수 누락	.env에 SPOTIPY_CLIENT_ID, SPOTIPY_CLIENT_SECRET 설정
Docker 빌드 느림	캐시 미사용	docker compose build --no-cache 로 재빌드

🏁 태그 및 버전
v1.0.0-stable → 완전 정리된 배포 기준 버전

이후 자동 빌드용: dev, staging, stable 등으로 태그 관리 예정

📰 참고자료
Spotify Web API Documentation

FastAPI Official Docs

Streamlit Docs

MLflow Docs

Apache Airflow Docs

yaml
코드 복사
