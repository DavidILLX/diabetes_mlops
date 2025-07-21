#!/bin/bash
echo "Starting new deployment"

cd /home/ubuntu/diabetes_mlops || exit 1

echo "Downloading new chnages from main....."
git pull origin main || exit 1

echo "Fixing ownership..."
sudo chown -R ubuntu:ubuntu .

echo "Restartign Airflow"
sudo docker-compose down
sudo docker-compose up -d --build airflow-webserver airflow-scheduler airflow-worker airflow-triggerer || exit 1

echo "Fixing ownership..."
sudo chown -R ubuntu:ubuntu .

echo "Triggering main DAG (model_dags)..."
sudo docker-compose exec airflow-webserver airflow dags trigger -c '{}' model_dags || exit 1

echo "Deployment finished"
