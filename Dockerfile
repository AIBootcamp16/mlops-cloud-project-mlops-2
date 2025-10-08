FROM python:3.11-bookworm

# 환경 변수 설정
ENV AIRFLOW_HOME=/usr/local/airflow \
    PYTHONUNBUFFERED=1 \
    LANG=ko_KR.UTF-8 \
    LC_ALL=ko_KR.UTF-8

# 언어 설정
# 시스템 패키지 + 로케일 생성 (한 레이어에 묶고, 끝에 캐시 삭제)
RUN apt-get update && apt-get install -y --no-install-recommends gcc libc-dev vim locales \
  && sed -i 's/# en_US.UTF-8 UTF-8/en_US.UTF-8 UTF-8/' /etc/locale.gen \
  && sed -i 's/# ko_KR.UTF-8 UTF-8/ko_KR.UTF-8 UTF-8/' /etc/locale.gen \
  && locale-gen \
  && rm -rf /var/lib/apt/lists/*

# 의존성 파일 설치 (docker 캐시 아끼려면 코드 복사 전에 의존성 먼저 설치)
# COPY requirements.txt /tmp/requirements.txt
# RUN pip install --no-cache-dir -r /tmp/requirements.txt && \
#     pip cache purge
# 빌드 속도 이슈로 캐시 사용... 
# RUN pip install --upgrade pip && \
#     pip install --cache-dir=/root/.cache/pip -r /tmp/requirements.txt
RUN pip install --upgrade pip
		
# ===== Apache Airflow 설치 (constraints 사용) =====
# apache 설치 - airflow 2.7.2 (공식 constraints 이용)
# RUN pip install --no-cache-dir apache-airflow==2.7.2 \
#     --constraint "https://raw.githubusercontent.com/apache/airflow/constraints-2.7.2/constraints-3.11.txt"
# RUN pip install --cache-dir=/root/.cache/pip -r /tmp/requirements.txt
FROM apache/airflow:2.7.2-python3.11
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt


# ===== Airflow 초기화는 CMD 단계에서 실행 =====
# → 빌드 중에 `airflow db init` 실행 시 레이어 캐시가 깨지고 이미지 빌드 느려짐
# → 실제 컨테이너 시작할 때 init 여부 확인하는 게 더 나음
RUN mkdir -p "$AIRFLOW_HOME/dags"
WORKDIR /app
# RUN airflow db init

# ===== DAG 복사 (선택 사항) =====
# CI/CD에서 volume mount 한다면 COPY 생략 가능
# COPY dags/my_dag.py $AIRFLOW_HOME/dags/

# airflow port
EXPOSE 8080

# 웹서버 + 스케줄러 실행
# - 컨테이너 시작 시 DB 상태를 점검하고(없으면 init) 구동
# - scheduler는 PID 1로 실행되도록 exec 사용

# ===== 컨테이너 시작 시 실행 =====
# airflow db check → 없으면 init
CMD bash -lc " airflow db check || airflow db init; \
    airflow users create --username airflow --password airflow --firstname air --lastname flow --role Admin --email airflow@example.com || true; \
    airflow webserver -p 8080 & \
    exec airflow scheduler"