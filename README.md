
# 🎧 컨텐츠 유사도 기반 음악 추천 시스템

<br>

## 💻 프로젝트 소개
사용자가 검색한 음악의 **오디오 특성(Audio Features)**을 기반으로  
컨텐츠 기반 추천(Content-based Filtering)을 수행하여 **맞춤형 추천 리스트**를 제공합니다.  

Kaggle의 Spotify 1Million Track Dataset을 활용하여 **메타데이터 / 오디오 특성 / 앨범 이미지**를 가져오며,  
**FAISS 벡터 검색**과 **LightGBM 모델**을 결합하여 개인화된 음악 추천을 제공합니다.  

또한 **MLOps 파이프라인**을 통해 데이터 수집·전처리부터 모델 학습/배포, 버전 관리, 모니터링까지 자동화하여 운영할 수 있도록 설계되었습니다.  


<br>

## ✨ Features
- 🔍 **검색(Search)**: Spotify 1M Dataset 기반 트랙 검색  
- 🎶 **Seed Selection**: 사용자가 선택한 시드곡 기반 추천 시작  
- 📊 **추천 리스트 생성**  
  - FAISS로 최근접 이웃 벡터 검색  
  - LightGBM 모델로 인기도/추가 지표 반영  
  - Top-K 추천 결과 생성  
- 🎧 **웹 UI**: Streamlit 기반 UI (검색, 추천, 모니터링)  
- ⚙️ **API 서비스**: FastAPI 기반 추천 API 제공  
- 🗄️ **로그 관리**: 추천 로그 테이블 설계 및 DB 저장  
- 🧠 **MLOps 통합 관리**  
  - Airflow: 데이터/모델 파이프라인 스케줄링  
  - MLflow: 모델 학습/실험 및 버전 관리  
  - MinIO: 모델 아티팩트 저장소  
- 🐳 **Docker 지원**: Docker Compose로 전체 시스템 통합 실행  
- 🔄 **CI/CD**: Github Actions로 코드 통합/검증 및 자동 배포

<br>

## 👨‍👩‍👦‍👦 팀 구성원


| 김소은 | 김재록 | 김종화 | 최보경 | 황은혜 |
| :----: | :----: | :----: | :----: | :----: |
| [김소은](https://github.com/oriori88)<br>데이터전처리, api구현, UI개발 | [김재록](https://github.com/UpstageAILab)<br>데이터시각화, 재현성확보 | [김종화](https://github.com/UpstageAILab)<br>모델 학습 및 모델 서버 구축 | [최보경](https://github.com/UpstageAILab)<br>CI/CD | [황은혜](https://github.com/UpstageAILab)<br>airflow |


<br>

## 🔨 개발 환경 및 기술 스택
- **언어**: Python 3.10+ 
- **웹 프레임워크**: FastAPI, Streamlit  
- **MLOps 도구**: Airflow, MLflow, MinIO
- **ML 라이브러리**: LightGBM, FAISS  
- **환경 관리**: Docker, Docker Compose, AWS EC2
- **버전 관리**: Git, GitHub
- **CI/CD**: GitHub Actions
- **DB**: MySQL/PostgreSQL

<br>

## 📁 프로젝트 구조
```
mlops-cloud-project-mlops-2
├── dataset/              # 원본(raw) 및 전처리(processed) 데이터
│   ├── raw/
│   └── processed/
├── models/               # 학습된 모델 산출물 (faiss, joblib, json 등)
├── notebooks/            # Jupyter 노트북 (EDA 및 실험)
├── src/                  # 소스코드
│   ├── api/              # FastAPI 서버
│   ├── data/             # 데이터 전처리 및 수집 모듈
│   └── model/            # 모델 정의 및 학습 코드
├── web/                  # Streamlit 기반 웹 UI
├── log/                  # 로깅 모듈
├── docs/                 # SQL 및 문서
├── mlruns/               # MLflow 실험 기록
├── tests/                # 테스트 코드
├── docker-compose.yml    # Docker Compose 설정
├── Dockerfile            # Docker 이미지 빌드 파일
├── requirements.txt      # Python 패키지 의존성
└── README.md             # 프로젝트 문서


```

<br>

## 🏗️ Architecture
```
             ┌────────────────────┐
             │     ModelServer    │
             └─────────┬──────────┘
                       │
        ┌──────────────┼──────────────┐
        │                             │
   ┌────▼─────┐                 ┌─────▼─────┐
   │  MinIO   │                 │   MLflow   │
   └────┬─────┘                 └─────┬─────┘
        │                             │
┌───────▼────────┐             ┌──────▼───────┐
│ Model Training │             │     API      │
│   Container    │             │ (FastAPI)    │
└────────────────┘             └──────────────┘
```

<br>

## 🚀 Installation & Usage
```bash
# 1. Clone repository
git clone https://github.com/your-repo/music-recommender.git
cd music-recommender

# 2. Run with Docker Compose
docker-compose up --build

# 3. Access services
# Streamlit UI: http://localhost:8501
# FastAPI Docs: http://localhost:8000/docs
# Airflow: http://localhost:8080
# MLflow: http://localhost:5000
```

<br>

## 📊 Example Workflow
1. 사용자 UI에서 트랙 검색  
2. 시드곡 선택 → FastAPI 서버 호출  
3. FAISS로 유사 곡 검색  
4. LightGBM으로 인기도 반영  
5. 추천 결과 UI에 출력 & 로그 DB 저장  
6. Airflow DAG 실행으로 주기적 학습 & 평가  
7. MLflow Registry에 모델 버전 관리 → 최적 모델을 Production으로 배포  
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

 
<br>

## 📌 Conclusion
이 프로젝트는 단순한 추천 모델 구현을 넘어, **실제 서비스 운영 환경에서의 MLOps 워크플로우**를 실습한 사례입니다.  
데이터 파이프라인, 모델 관리, CI/CD, 버전 관리까지 통합하여 **End-to-End MLOps 시스템**을 완성하였습니다.



