FROM python:3.12-slim

RUN pip install setuptools

RUN pip install mlflow==2.12.1

EXPOSE 5000

CMD [ \
    "mlflow", "server", \
    "--backend-store-uri", "sqlite:///home/mlflow_data/mlflow.db", \
    "--host", "0.0.0.0", \
    "--port", "5000" \
]