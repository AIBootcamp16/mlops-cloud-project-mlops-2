
# 🎧 컨텐츠 유사도 기반 음악 추천 시스템
#### 사용자가 검색한 음악과 유사한 음악을 추천합니다.
<div>
  <a href="https://github.com/yourusername/yourrepo">
    <img src="https://img.shields.io/github/stars/yourusername/yourrepo?style=social" alt="GitHub stars"/>
  </a>
  <a href="https://github.com/yourusername/yourrepo">
    <img src="https://img.shields.io/github/forks/yourusername/yourrepo?style=social" alt="GitHub forks"/>
  </a>
</div>

## 👪 Team

| ![김소은](https://avatars.githubusercontent.com/u/11532528?v=4) | ![김재록](https://avatars.githubusercontent.com/u/47843619?v=4) | ![김종화](https://avatars.githubusercontent.com/u/221108223?v=4) | ![최보경](https://avatars.githubusercontent.com/u/110219144?v=4) | ![황은혜](https://avatars.githubusercontent.com/u/100017750?v=4) |
| :--------------------------------------------------------------: | :--------------------------------------------------------------: | :--------------------------------------------------------------: | :--------------------------------------------------------------: | :--------------------------------------------------------------: |
|            [김소은 (팀장)](https://github.com/oriori88)             |            [김재록](https://github.com/Raprok612)             |            [김종화](https://github.com/JHKIM-ItG)             |            [최보경](https://github.com/bekky1016)             |            [황은혜](https://github.com/eeunhyee)             |
| 전처리 <br> api구현 <br> UI개발 <br> 작업환경 구성 <br> readme 작성 <br> 문서 정리 | 데이터 시각화 <br> UI개발 <br> 문서 정리 | 전처리 <br> 모델 학습 <br> 모델 서버 구축 <br> 문서 정리 | CI <br> CD <br> 문서 정리 | airflow <br> 문서 정리 |

<br>

## 0. Overview
사용자가 검색한 음악의 **오디오 특성(Audio Features)** 을 기반으로 컨텐츠 기반 추천을 수행하여 **맞춤형 추천 리스트**를 제공합니다.  

Kaggle의 Spotify 1Million Track Dataset을 활용하여 **메타데이터 / 오디오 특성 / 앨범 이미지**를 가져오며, **FAISS 벡터 검색**과 **LightGBM 모델**을 결합하여 개인화된 음악 추천을 제공합니다.  

또한 **MLOps 파이프라인**을 통해 데이터 수집·전처리부터 모델 학습/배포, 버전 관리, 모니터링까지 자동화하여 운영할 수 있도록 설계되었습니다.  

#### 🛠️ Tech Stack
| 구분 | 기술 |
|------|------|
| **언어** | ![Python](https://img.shields.io/badge/Python-3776AB?style=flat&logo=python&logoColor=white) |
| **웹 프레임워크** | ![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=flat&logo=fastapi&logoColor=white) ![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=flat&logo=streamlit&logoColor=white) |
| **MLOps 도구** | ![Airflow](https://img.shields.io/badge/Airflow-017CEE?style=flat&logo=apacheairflow&logoColor=white) ![MLflow](https://img.shields.io/badge/MLflow-0194E2?style=flat&logo=mlflow&logoColor=white) ![MinIO](https://img.shields.io/badge/MinIO-C72E49?style=flat&logo=minio&logoColor=white) |
| **ML 라이브러리** | ![LightGBM](https://img.shields.io/badge/LightGBM-014728?style=flat&logo=lightgbm&logoColor=white) ![FAISS](https://img.shields.io/badge/FAISS-0052CC?style=flat) |
| **환경 관리** | ![Docker](https://img.shields.io/badge/Docker-2496ED?style=flat&logo=docker&logoColor=white) ![Docker Compose](https://img.shields.io/badge/Docker_Compose-2496ED?style=flat&logo=docker&logoColor=white) ![AWS EC2](https://img.shields.io/badge/AWS_EC2-FF9900?style=flat&logo=amazonaws&logoColor=white) |
| **버전 관리** | ![Git](https://img.shields.io/badge/Git-F05032?style=flat&logo=git&logoColor=white) ![GitHub](https://img.shields.io/badge/GitHub-181717?style=flat&logo=github&logoColor=white) |
| **CI/CD** | ![GitHub Actions](https://img.shields.io/badge/GitHub_Actions-2088FF?style=flat&logo=githubactions&logoColor=white) |
| **Database** | ![MySQL](https://img.shields.io/badge/MySQL-4479A1?style=flat&logo=mysql&logoColor=white) ![PostgreSQL](https://img.shields.io/badge/PostgreSQL-4169E1?style=flat&logo=postgresql&logoColor=white) |

#### 📦 Requirements
Package | Version
--- | ---
numpy | 1.26.4
pandas | 2.1.4
scikit-learn | 1.7.2
lightgbm | 4.6.0
xgboost | 3.1.1
catboost | 1.2.8
matplotlib | 3.10.7
seaborn | 0.13.2
plotly | 3.1.1
geopandas | 1.1.1
shap | 0.50.0
ipykernel | 6.27.1
ipython | 8.15.0
jupyter | 1.0.0

<br>

## 1. Dataset Overview
<img width="1066" height="477" alt="image" src="https://github.com/user-attachments/assets/b037f020-4603-45da-a7cb-a543bace2650" />

- 데이터셋 이름: Spotify Dataset
- 출처: Kaggle
- 목표: 오디오 특성을 이용하여 유사한 컨텐츠를 탐색하여 사용자에게 추천
  
<br>  
<img width="410" height="586" alt="image" src="https://github.com/user-attachments/assets/66f37ffb-9e4b-4ec8-947c-1e62bcccebb9" />

- 주요 컬럼 : 오디오 특성 (danceability, energy, loudness, speechiness, acousticness, instrumentalness, liveness, valence, tempo 등)

<br>

## 2. Data Processing
<table style="border: 0">
  <tr>
    <td>
      <img width="530" height="520" alt="image" src="https://github.com/user-attachments/assets/00759055-79a5-47c5-b5a0-f6f09ecc4c14" />
    </td>
    <td>
      <img width="350" height="523" alt="image" src="https://github.com/user-attachments/assets/1993438d-4676-4704-bad3-ad1b6644e6c2" />
    </td>
  </tr>
</table>

- 결측치 제거를 통해 정제된 데이터를 csv로 저장 후 사용

<br>

## 3. Model Description
**1) FAISS**
<table>
  <tr>
    <td>
      <img width="250" height="138" alt="image" src="https://github.com/user-attachments/assets/dd513727-e021-49c5-872e-5e628ad3a1dd" />
    </td>
    <td>
      음악 추천 시스템에서는 사용자 취향과 유사한 음악을 빠르게 도출하는것이 중요.<br>
      FAISS는 대규모 벡터 데이터에서 빠른 근접 이웃 검색이 가능<br>
      사용자가 선호하는 음악과 유사한 곡을 실시간으로 추천이 가능
    </td>
  </tr>
</table>
    
**2) LGBM**
<table>
  <tr>
    <td>
      <img width="295" height="64" alt="image" src="https://github.com/user-attachments/assets/7569943a-e45b-4f30-b622-8d4e1dc0861b" />
    </td>
    <td>
      Gradient Boosting Decision Tree 기반 모델로 학습 속도가 빠르며, 대용량 데이터에서도 효율적<br>
      메모리사용량이 적어 소규모 서버에서 돌리기에도 적합할 것이라 판단
    </td>
  </tr>
</table>

<br>

### 🏋️ 훈련 과정
- 데이터 준비: 데이터 전처리
- FAISS 임베딩 벡터 인덱스 구축
- LGBM 학습 데이터와 검증 데이터 분리
- LGBM 모델 학습

<br>

### *① FAISS 유사 음악 N개 추천 → ② LGBM 인기도 예측 → ③ 인기도 높은 최종 N개 도출*

<br>

## 4. Architecture
<img width="888" height="480" alt="image" src="https://github.com/user-attachments/assets/6f2a0945-ad8d-40fc-959b-309e6dda69ce" />

#### 🚩 CI/CD 파이프라인
<img width="426" height="286" alt="image" src="https://github.com/user-attachments/assets/b86d2f2a-989d-4102-9715-f94bb322424e" />

#### 🚩 전체 진행 플로우 (Airflow)
<img width="1366" height="98" alt="image" src="https://github.com/user-attachments/assets/a31e2604-2afb-442d-ba1c-bf3a54f4ecf9" />

* 데이터 수집 - 데이터 전처리 - mlflow 컨테이너 실행 - 모델 학습 - mlflow 컨테이너 종료 - 모델 평가
* [mlflow 컨테이너 실행 - 모델 학습 - mlflow 컨테이너 종료] -> DAG 생성

<br>

## 5. UI Preview
  #### 🎼 음악 추천
<img width="703" height="573" alt="image" src="https://github.com/user-attachments/assets/135e2643-58d5-470c-88b5-56cfe1a482ef" />
<img width="1021" height="464" alt="image" src="https://github.com/user-attachments/assets/b6b99e3d-fd9b-430f-8767-06d7cffd4fc4" />

  #### 🖥️ 모니터링
<img width="805" height="614" alt="image" src="https://github.com/user-attachments/assets/fcb17a1a-7efd-4494-abe8-f86ec6c5e25d" />
<img width="559" height="535" alt="image" src="https://github.com/user-attachments/assets/75a6cf81-0d6c-467d-8a93-df382b942582" />

<br>

## 6. Features
🔍 **검색(Search)** : Spotify 1M Dataset 기반 트랙 검색  

🎶 **Seed Selection** : 사용자가 선택한 시드곡 기반 추천 시작  

📊 **추천 리스트 생성**  
  - FAISS로 최근접 이웃 벡터 검색  
  - LightGBM 모델로 인기도/추가 지표 반영  
  - Top-K 추천 결과 생성

🎧 **웹 UI** : Streamlit 기반 UI (검색, 추천, 모니터링)  

⚙️ **API 서비스** : FastAPI 기반 추천 API 제공  

🗄️ **로그 관리** : 추천 로그 테이블 설계 및 DB 저장  

🧠 **MLOps 통합 관리**  
  - Airflow: 데이터/모델 파이프라인 스케줄링  
  - MLflow: 모델 학습/실험 및 버전 관리  
  - MinIO: 모델 아티팩트 저장소  

🐳 **Docker 지원** : Docker Compose로 전체 시스템 통합 실행  

🔄 **CI/CD** : Github Actions로 코드 통합/검증 및 자동 배포

<br>

## 7. 프로젝트 구조
#### 📂 Directory
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

## 8. Timeline
<img width="784" height="434" alt="image" src="https://github.com/user-attachments/assets/90fef594-fbec-423e-bbef-981c9a993b04" />

Jira로 협업 및 스케줄 관리 수행

<br>

## 9. 프로젝트 회고

#### 📌 Conclusion
이 프로젝트는 단순한 추천 모델 구현을 넘어, **실제 서비스 운영 환경에서의 MLOps 워크플로우**를 실습했다는 점에서 의미가 깊습니다.  
데이터 파이프라인, 모델 관리, CI/CD, 버전 관리까지 통합하여 **End-to-End MLOps 시스템**을 완성하였습니다.

- Docker 기반 통합 환경으로 실행 안정성 확보
- 빌드 캐시 최적화로 개발 효율 향상
- Airflow / MLflow 연동으로 MLOps 자동화 기반 마련

<br>

## 10. etc

#### 🎤 Presentation
[![Presentation](https://github.com/user-attachments/assets/2bbc0ccb-0263-4f93-ac29-0315b8979d20)](https://docs.google.com/presentation/d/19vyorym-La7C3MJQixHB4BqSferPyCJs/edit?slide=id.p1#slide=id.p1)

이미지를 누르면 상세한 PPT를 볼 수 있습니다.

#### 📚 참고자료
- Spotify Web API Documentation
- FastAPI Official Docs
- Streamlit Docs
- MLflow Docs
- Apache Airflow Docs




