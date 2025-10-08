from airflow import DAG
from airflow.operators.bash import BashOperator
from datetime import datetime


default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'retries': 1,
}

# DAG 정의
with DAG(
    dag_id='ml_pipeline_dag',
    default_args=default_args,
    description='End-to-End ML pipeline with data collection, preprocessing, training, and evaluation',
    start_date=datetime(2025, 10, 5),
    schedule_interval='@daily',  # 매일 1회 자동 실행
    catchup=False,
    tags=['mlops', 'pipeline']
) as dag:

    # 1. 데이터 수집
    collect_data = BashOperator(
        task_id='collect_data',
        bash_command='echo "1. 데이터 수집"'
        # bash_command='python /app/src/data/spotify_datacollecter.py'
    )

    # 2. 전처리
    preprocess_data = BashOperator(
        task_id='preprocess_data',
        bash_command='echo "2. 전처리"'
        # bash_command='python /app/src/data/main.py'
    )

    # 3. 모델 학습
    train_model = BashOperator(
        task_id='train_model',
        bash_command='echo "3. 모델 학습"'
        # bash_command='python /app/src/modelTrain.py'
    )

    # 4. 평가
    # evaluate_model = BashOperator(
    #     task_id='evaluate_model',
    #     bash_command='python /app/src/evaluate.py'
    # )

    # 실행 순서 정의
    # collect_data >> preprocess_data >> train_model >> evaluate_model
    collect_data >> preprocess_data >> train_model
