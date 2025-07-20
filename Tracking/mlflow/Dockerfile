FROM python:3.10-slim

RUN apt-get update && apt-get install -y gcc libpq-dev curl && \
    pip install --no-cache-dir pipenv setuptools

WORKDIR /app

COPY Pipfile Pipfile.lock /app/

RUN pipenv install --deploy --system
