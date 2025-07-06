from airflow import DAG
from airflow.operators.bash import BashOperator
from datetime import datetime, timedelta

default_args = {
    'owner': 'Ivan',
    'depends_on_past': False,
    'start_date': datetime(2025, 7, 1),
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

with DAG(
    'delivery_time_etl',
    default_args=default_args,
    schedule_interval='@daily',
    catchup=False
) as dag:

    clean_data_task = BashOperator(
        task_id='execute_data_clean_script',
        bash_command='spark-submit ../pyspark_script/clean_data_csv_to_s3.py'
    )

    regression_task = BashOperator(
        task_id='execute_regression_script',
        bash_command='spark-submit ../pyspark_script/random_forest_regression.py'
    )

    clean_data_task >> regression_task