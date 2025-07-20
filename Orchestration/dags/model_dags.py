from airflow import DAG  # type: ignore
from airflow.operators.bash import BashOperator  # type: ignore
from airflow.utils.dates import days_ago  # type: ignore

with DAG(
    dag_id="model_dags",
    start_date=days_ago(1),
    schedule_interval="@daily",
    catchup=False,
    tags=["mlops", "dag"],
    description="MLOPS pipeline in Airflow for training and registering best model",
) as dag:
    preprocess_task = BashOperator(
        task_id="preprocess_data",
        bash_command="python /app/Model/preprocessing_data.py --force-download",
    )
    train_task = BashOperator(
        task_id="train_models", bash_command="python /app/Model/train_model.py"
    )
    register_task = BashOperator(
        task_id="register_models", bash_command="python /app/Model/register_model.py"
    )

preprocess_task >> train_task >> register_task
