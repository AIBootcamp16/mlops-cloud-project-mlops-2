# -------------------------------------------------------------------------
# MLOps Base Dockerfile (Airflow Scheduler & Webserver)
# Airflow 구동 및 MLflow/ML Dependencies 설치용
# -------------------------------------------------------------------------

# Python 3.11 기반의 안정적인 Debian (bookworm) 이미지를 사용
FROM python:3.11-bookworm

# 컨테이너 내부 환경 변수 설정
ENV PYTHONUNBUFFERED=1

# Airflow 홈 디렉토리 설정
ENV AIRFLOW_HOME=/opt/airflow

# -------------------------------------------------------------------------
# 시스템 및 로케일 설정 (한글 지원)
# -------------------------------------------------------------------------

RUN apt-get update && apt-get install -y --no-install-recommends \
      gcc libc-dev vim locales \
      && sed -i 's/# en_US.UTF-8 UTF-8/en_US.UTF-8 UTF-8/' /etc/locale.gen \
      && sed -i 's/# ko_KR.UTF-8 UTF-8/ko_KR.UTF-8 UTF-8/' /etc/locale.gen \
      && locale-gen \
      && rm -rf /var/lib/apt/lists/*

# -------------------------------------------------------------------------
# 의존성 설치 (Airflow, ML Dependencies)
# -------------------------------------------------------------------------

# requirements.txt 파일을 컨테이너의 임시 경로에 복사
COPY requirements.txt /tmp/requirements.txt

# pip 의존성 설치 (ML Dependencies)
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r /tmp/requirements.txt

# Airflow 설치 (ML 파이프라인과 충돌하지 않도록 별도로 설치)
RUN pip install --no-cache-dir \
    "apache-airflow==2.7.2" \
    --constraint "https://raw.githubusercontent.com/apache/airflow/constraints-2.7.2/constraints-3.11.txt"

# -------------------------------------------------------------------------
# Airflow 디렉토리 및 기본 구조 생성
# -------------------------------------------------------------------------

RUN mkdir -p "$AIRFLOW_HOME/dags" "$AIRFLOW_HOME/logs" "$AIRFLOW_HOME/plugins"

# 작업 디렉토리 설정 (MLOps 프로젝트 코드가 마운트될 위치)
WORKDIR /app

# -------------------------------------------------------------------------
# ✅ 수정: 프로젝트 소스 코드 복사 (이 부분이 누락되어 파일이 보이지 않았습니다)
# -------------------------------------------------------------------------
COPY . /app 

# -------------------------------------------------------------------------
# 컨테이너 실행 명령어
# ----------------------------------------------------------------
