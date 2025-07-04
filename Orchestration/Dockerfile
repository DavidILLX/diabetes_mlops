FROM apache/airflow:2.8.1-python3.10

USER root

RUN apt-get update && apt-get install -y build-essential

USER airflow

RUN pip install \
    "SQLAlchemy>=1.4.0,<2.0.0" \
    "psycopg2-binary>=2.8.6,<2.10.0"

RUN pip uninstall -y pathlib || true

WORKDIR /app

COPY Pipfile Pipfile.lock ./

RUN python -m pip install --upgrade pip
RUN pip install pipenv && pipenv install --dev --system --deploy
RUN pipenv run pip install apache-airflow==2.8.1 \
    --constraint https://raw.githubusercontent.com/apache/airflow/constraints-2.8.1/constraints-3.10.txt

COPY . /app

RUN pipenv install --deploy --system