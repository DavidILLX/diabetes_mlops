FROM python:3.10

RUN apt-get update && apt-get install -y \
    gcc \
    libpq-dev \
    curl \
    build-essential \
    gfortran \
    libopenblas-dev \
    liblapack-dev \
    && pip install --no-cache-dir pipenv setuptools

WORKDIR /app

COPY Pipfile Pipfile.lock /app/

COPY . .

RUN pipenv install --deploy --system
