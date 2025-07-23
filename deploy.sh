#!/bin/bash

set -e  # exit on error

echo "Starting new deployment"

cd /home/ubuntu/diabetes_mlops || exit 1

echo "Downloading new chnages from main....."
git pull origin main || exit 1

echo "Fixing ownership..."
sudo chown -R ubuntu:ubuntu .

echo "Restartign Airflow"
sudo docker-compose down
sudo docker-compose up -d --build

echo "Fixing ownership..."
sudo chown -R ubuntu:ubuntu .

echo "Deployment finished"
