<<<<<<< HEAD
FROM python:3.11-bookworm

# 환경 변수 설정
ENV AIRFLOW_HOME=/usr/local/airflow

# 언어 설정
# 시스템 패키지 + 로케일 생성 (한 레이어에 묶고, 끝에 캐시 삭제)
RUN apt-get update && apt-get install -y --no-install-recommends \
      gcc libc-dev vim locales \
  && sed -i 's/# en_US.UTF-8 UTF-8/en_US.UTF-8 UTF-8/' /etc/locale.gen \
  && sed -i 's/# ko_KR.UTF-8 UTF-8/ko_KR.UTF-8 UTF-8/' /etc/locale.gen \
  && locale-gen \
  && rm -rf /var/lib/apt/lists/*

# 의존성 파일 설치 (docker 캐시 아끼려면 코드 복사 전에 의존성 먼저 설치)
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt
		
# apache 설치 - airflow 2.7.2 (공식 constraints 이용)
RUN pip install --no-cache-dir \
    "apache-airflow==2.7.2" \
    --constraint "https://raw.githubusercontent.com/apache/airflow/constraints-2.7.2/constraints-3.11.txt"

# 디렉토리 준비, 작업 디렉토리 설정
RUN mkdir -p "$AIRFLOW_HOME/dags"
WORKDIR /app
RUN airflow db init

# Dag 배포
COPY dags/my_dag.py $AIRFLOW_HOME/dags/

# airflow port
EXPOSE 8080

# 웹서버 + 스케줄러 실행
# - 컨테이너 시작 시 DB 상태를 점검하고(없으면 init) 구동
# - scheduler는 PID 1로 실행되도록 exec 사용
CMD bash -lc 'airflow db check || airflow db init; airflow webserver -p 8080 & exec airflow scheduler'
=======
# 1. image build
FROM python:3.11-bookworm

# 2. 작업 디렉토리 생성
WORKDIR /app

# 3. 의존성 설치
RUN pip install --no-cache-dir \
    numpy \
    pandas \
    requests \
    scikit-learn \
    faiss-cpu \
    lightgbm \
    mlflow \
    joblib \
    boto3

# 4. 추가 의존성 설치 (dotenv)
RUN pip install --no-cache-dir python-dotenv

# 5. 로컬 프로젝트 코드 복사
# (Dockerfile이 프로젝트 루트에 있으므로, 프로젝트 전체 폴더를 컨테이너 내부로 복사)
#COPY mlops-cloud-project-mlops-2 /app/mlops
COPY . /app/mlops

# 6. 작업 디렉토리 재설정 (run_pipeline.sh가 이 디렉토리에 있다고 가정)
# 파이프라인 스크립트(run_pipeline.sh)가 위치한 곳으로 작업 경로를 변경합니다.
WORKDIR /app/mlops    

# 7. run_pipeline.sh 실행 권한 부여
RUN chmod +x run_pipeline.sh

# 8. 컨테이너 실행 명령어 명시
# 현재 WORKDIR이 /app/mlops이므로, ./run_pipeline.sh 대신 절대 경로를 사용하거나 
# ENTRYPOINT로 실행을 보장합니다.
ENTRYPOINT ["/app/mlops/run_pipeline.sh"]
>>>>>>> Rok-setup
