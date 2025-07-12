from airflow import DAG # type: ignore
from airflow.operators.bash import BashOperator # type: ignore
from datetime import datetime, timedelta

default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

with DAG(
    dag_id='evidently_monitoring_dag',
    default_args=default_args,
    description='Run Evidently monitoring every hour',
    schedule_interval='@hourly',
    start_date=datetime(2025, 7, 11),
    catchup=False,
    tags=['mlops', 'monitoring', 'evidently'],
) as dag:

    run_monitoring = BashOperator(
        task_id='run_evidently_monitoring',
        bash_command='python /app/Monitoring/evidently_metrics.py'
    )

    run_monitoring