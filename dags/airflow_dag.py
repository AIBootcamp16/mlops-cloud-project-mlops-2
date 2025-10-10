import os
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.bash import BashOperator
from datetime import datetime


# uri 확인
def tracking_test():
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
    print(tracking_uri)

# mlflow 로그 확인
def log_to_mlflow():
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment("airflow_demo")
    with mlflow.start_run(run_name="airflow_log"):
        mlflow.log_param("p1", 42)
        mlflow.log_metric("m1", 0.99)

# default arguments
default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'start_date': datetime(2024, 1, 1),
    'retries': 1,
}

with DAG(
    dag_id="mlflow",
    description="모델 학습",
    start_date=datetime(2025, 10, 1),
    schedule_interval="@daily",
    catchup=False,
) as dag:
    #컨테이너 실행
    compose_up = BashOperator(
        task_id='compose_up',
        bash_command='docker-compose -f /opt/airflow/mlflow_compose/docker-compose.yaml up -d',
    )

    #프로그램 실행 (컨테이너 내부)
    run_program = BashOperator(
        task_id='run_modelTrain',
        bash_command='docker exec train_test python /app/mlops/src/modelTrain.py',
    )

    #컨테이너 종료
    compose_down = BashOperator(
        task_id='compose_down',
        bash_command='docker-compose -f /opt/airflow/mlflow_compose/docker-compose.yaml down',
    )

    # 실행 순서 지정
    compose_up >> run_program >> compose_down


    # 기본 ml 실행 파이프라인 DAG
with DAG(
    dag_id="ml_pipeline_python_operator",
    description="기본 ML 파이프라인",
    start_date=datetime(2025, 10, 1),
    schedule_interval="@daily",
    catchup=False,
) as dag:

    preprocess = BashOperator(
        task_id="preprocess",
        bash_command="python /opt/airflow/src/data/main.py"
    )
    
    preprocess