#!/bin/bash
set -e

echo "Starting user-data script"

# Update and install packages
sudo apt-get update -y || { echo "apt-get update failed"; exit 1; }
sudo apt-get install -y docker.io docker-compose git software-properties-common jq || { echo "Package installation failed"; exit 1; }

# Get Python 3.10
sudo apt-get install -y python3.10 python3-pip python3.10-venv python3.10-distutils || { echo "Python 3.10 installation failed"; exit 1; }

# Changing prefered poython to 3.10
sudo update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 100

# Chnaging so pip uses 3
sudo update-alternatives --install /usr/bin/pip3 pip3 /usr/bin/pip3 100 || true

# Update pip to the newest version
/usr/bin/python3.10 -m pip install --upgrade pip || { echo "Failed to upgrade pip for Python 3.10"; exit 1; }

# Install Jupyter notebook
sudo apt install -y jupyter || { echo "Jupyter installation failed"; exit 1; }

# Get to the user folder
cd /home/ubuntu/

# Install pipenv and add PATH
sudo -u ubuntu -i bash <<'EOF'
python3.10 -m pip install --user pipenv || { echo "Pipenv installation failed"; exit 1; }
grep -qxF 'export PATH=$HOME/.local/bin:$PATH' ~/.bashrc || echo 'export PATH=$HOME/.local/bin:$PATH' >> ~/.bashrc
echo "Pipenv installed and PATH updated for ubuntu user"
EOF

# Get repository from git and create env file
echo "Attempting to clone repository"
for i in {1..3}; do
  if git clone https://github.com/DavidILLX/diabetes_mlops.git /home/ubuntu/diabetes_mlops; then
    success=1
    break
  else
    echo "Git clone failed, retrying ($i/3)"
    sleep 5
  fi
done

if [ $success -ne 1 ]; then
  echo "Git clone failed after 3 attempts"
  exit 1
fi
echo "Repository cloned successfully"

# Get to the repository
cd diabetes_mlops/

# Create .env file
cat <<EOT > .env
AWS_ACCESS_KEY_ID=
AWS_SECRET_ACCESS_KEY=
AWS_REGION=eu-north-1
KAGGLE_USERNAME=
KAGGLE_KEY=
AIRFLOW_UID=50000
AIRFLOW_PROJ_DIR=./Orchestration
AIRFLOW_IMAGE_NAME=apache/airflow:2.8.1
EOT

# Change ownership to make it wr
chown ubuntu:ubuntu /home/ubuntu/diabetes_mlops/.env
chown -R ubuntu:ubuntu /home/ubuntu/diabetes_mlops

echo "Script completed successfully"