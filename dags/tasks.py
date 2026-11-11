from airflow import DAG
from datetime import datetime
from airflow.operators.python import PythonOperator

from src import eda, preprocess, modelisation, evaluate, export


with DAG(
    dag_id="student_mental_health_ml",
    start_date=datetime(2026, 11, 10),
    schedule="@daily",
    catchup=False,
) as dag:

    t1 = PythonOperator(
        task_id="EDA",
        python_callable=eda.EDA
    )

    t2 = PythonOperator(
        task_id="preprocessing",
        python_callable=preprocess.preprocess
    )

    t3 = PythonOperator(
        task_id="modelisation",
        python_callable=modelisation.modelisation
    )

    t4 = PythonOperator(
        task_id="evaluation",
        python_callable=evaluate.evaluation
    )

  

    t1 >> t2 >> t3 >> t4