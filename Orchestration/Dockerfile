FROM apache/airflow:2.8.1-python3.10

USER root

RUN apt-get update && apt-get install -y build-essential python3-pip python3-venv

USER airflow

RUN pip install apache-airflow==2.8.1 \
    --constraint "https://raw.githubusercontent.com/apache/airflow/constraints-2.8.1/constraints-3.10.txt"

RUN python -m pip install --upgrade pip

WORKDIR /app

COPY Pipfile Pipfile.lock ./

RUN pip install pipenv

RUN pipenv install --deploy

COPY . .