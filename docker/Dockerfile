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
RUN pip install --no-cache-dir -r ./requirements.txt
		
# apache 설치 - airflow 2.7.2 (공식 constraints 이용)
RUN pip install --no-cache-dir \
    "apache-airflow==2.7.2" \
    --constraint "https://raw.githubusercontent.com/apache/airflow/constraints-2.7.2/constraints-3.11.txt"

# 디렉토리 준비, 작업 디렉토리 설정
RUN mkdir "$AIRFLOW_HOME/dags"
WORKDIR /app
RUN airflow db init

# Dag 배포
COPY my_dag.py $AIRFLOW_HOME/dags/

# airflow port
EXPOSE 8080

# 웹서버 + 스케줄러 실행
# - 컨테이너 시작 시 DB 상태를 점검하고(없으면 init) 구동
# - scheduler는 PID 1로 실행되도록 exec 사용
CMD bash -lc 'airflow db check || airflow db init; airflow webserver -p 8080 & exec airflow scheduler'