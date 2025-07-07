FROM apache/airflow:2.8.1-python3.10

USER root

RUN apt-get update && apt-get install -y \
    build-essential \
    python3-pip \
    python3-venv && apt-get clean

USER airflow

RUN python -m pip install --upgrade pip setuptools wheel

RUN pip install apache-airflow==2.8.1 \
    --constraint "https://raw.githubusercontent.com/apache/airflow/constraints-2.8.1/constraints-3.10.txt"

WORKDIR /app

COPY Orchestration/requirements.txt .


RUN pip install --no-cache-dir -r requirements.txt

COPY . .