#!/bin/bash
sudo apt-get update -y
sudo apt-get install -y docker.io docker-compose git
sudo apt install -y software-properties-common

# Updating Python to 3.11 version
sudo apt install -y python-is-python3
sudo apt install -y python3.11 python3.11-venv python3.11-distutils python3-pip

# Changing Ubuntu python 3.10
sudo update-alternatives --remove-all python3 || true
sudo update-alternatives --remove-all pip3 || true

# Changing prefered poython to 3.11
sudo update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 100
sudo update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 50 

# Chnaging so pip uses 3.11
sudo update-alternatives --install /usr/bin/pip3 pip3 /usr/bin/pip3 100 




