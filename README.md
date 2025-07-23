# Diabetes MLOps Project

**Project goal:** Build a fully automated MLOps pipeline deployed in AWS cloud, following best practices in CI/CD, monitoring, orchestration, infrastructure-as-code (IaC), and model management. The project uses a diabetes dataset for binary classification.

This project was created as part of the [DataTalks Club MLOps Zoomcamp](https://github.com/DataTalksClub/mlops-zoomcamp).

### 🩺 Problem Statement

Diabetes is a chronic disease affecting millions worldwide, and early prediction is critical to reducing severe complications. Healthcare providers need robust, scalable, and reliable systems that can automate the prediction process based on patient data. However, most machine learning models never make it into production due to the lack of proper infrastructure, versioning, monitoring, and automation. This project addresses that gap by implementing an end-to-end MLOps pipeline that not only trains and deploys predictive models for diabetes but also automates the entire lifecycle — from data ingestion to model monitoring — using modern DevOps and DataOps practices. It solves the real-world problem of operationalizing ML solutions in a reproducible and cloud-native way.

---

## 🔧 Tools and Technologies Used

* **Infrastructure**: Terraform, AWS (EC2, RDS, S3, VPC)
* **CI/CD**: GitHub Actions, Makefile, pre-commit hooks (Black, isort, flake8)
* **Orchestration**: Apache Airflow
* **Experiment Tracking**: MLflow
* **Tests**: Pytests
* **Monitoring**: Evidently + Grafana
* **Packaging**: Docker, Docker Compose
* **Model Training**: XGBoost, Logistic Regression, CatBoost, Random Forest
* **App**: Flask API for predictions

---

## 🌍 Project Architecture

![Project Architecture](.assets/Project_achitecture.jpg)

The infrastructure is provisioned using Terraform. Everything runs within a private AWS VPC. Only the Flask API is publicly accessible. All components communicate internally over Docker Compose.

### 📁 Repository Structure

```
├── Infrastructure        # Terraform scripts
├── Analysis              # Basic EDA of the dataset
├── Model                 # Training and preprocessing logic
├── Data                  # Downloaded and transformed data
├── App                   # Flask prediction service
├── Monitoring            # Evidently monitoring scripts
├── Tracking              # MLflow integration
├── Orchestration         # Airflow intergration
├── Tests                 # Unit tests for preprocessing
├── docker-compose.yml
├── Makefile
├── .pre-commit-config    # Pre-commit hooks for linting, isort, formatting
├── pyproject.toml        # Config for pre-commit
├── Pipfile               # Pytrhon enviroment
├── Pipfile.lock          # Locked enviroment with versions packages
└── .github/workflows     # CI/CD pipelines and manual Terraform destroy
```

---

## ⚙️ How to Run the Project

### 1. ✅ Requirements

You only need:

* AWS account (Creator access)
* Kaggle account
* GitHub Secrets:

  * `KAGGLE_USERNAME`
  * `KAGGLE_KEY`
  * `AWS_ACCESS_KEY_ID`
  * `AWS_SECRET_ACCESS_KEY`
  * `MLOPS_SSH_PUBLIC_KEY`
  * `EC2_SSH_KEY_PRIVATE`

IAM roles are not explicitly defined (for simplicity), but production-ready setup should define roles for EC2, S3, RDS and CloudWatch.

---

### 🔑 Creating SSH Key Pair

To connect to the EC2 instance, you’ll need an SSH key pair. If you don't have one, generate it using the command below:

```bash
ssh-keygen -t rsa -b 4096 -f ~/.ssh/mlops
```

### 2. 🚀 Launch via GitHub Actions

1. Fork or clone this repository.
2. Upload secrets mentioned above into GitHub Secrets.
3. Trigger the GitHub Action manually (`Deploy` workflow).
4. GitHub Actions will:

   * Run tests, linting, formatting
   * Build infrastructure with Terraform
   * Deploy Docker containers to EC2
   * Set up .env, volumes and S3 connections

---

### 3. 🔐 Connect to EC2 Server

Copy your public EC2 DNS or IP from Terraform output, then create a local SSH config file like this:

```bash
Host mlops-server
    HostName <YOUR_EC2_PUBLIC_DNS>
    User ubuntu
    IdentityFile ~/.ssh/mlops
    IdentitiesOnly yes
```

Then connect with:

```bash
ssh mlops-server
```

---

### 4. 🔄 Port Forwarding (for UI Access)

Use VSCode Remote SSH or GitHub Codespaces to forward these ports:

| Service      | Port |
| ------------ | ---- |
| Flask API    | 9696 |
| Airflow      | 8080 |
| MLflow       | 5000 |
| Evidently UI | 8000 |
| Grafana      | 3000 |

> All services run inside the VPC, port forwarding is required to view them in browser.

---

## 🤖 Model Logic

* Trains 4 models (XGBoost, Logistic Regression, CatBoost, Random Forest)
* Uses `fmin` to tune hyperparameters.
* Selects best model based on F1-score, accuracy and recall.
* Registers best model as **Champion** in MLflow.
* If better model is trained, previous version is deprecated.

---

## 📊 Monitoring and Reporting

* **Evidently** generates HTML reports stored in S3 and visible in its UI.
* Tracks metrics like F1-score, recall, accuracy, precision, ROC-AUC.
* Important metrics are pushed into PostgreSQL and visualized in **Grafana**.

> 📌 *Placeholder for Grafana dashboard screenshot*

---

## 🧪 CI/CD Pipeline

* Linting and formatting (Black, isort)
* Unit tests for preprocessing
* Terraform steps: `init`, `plan`, `apply`
* Docker builds and deploy to EC2
* Auto-provisioning of `.env` file and all services

---

## 📦 Flask API

Predicts diabetes outcome from input features. Accessible on port `9696` after deployment.

![Flask API](.assets/flask.jpg)


---

## 📣 Summary

This project demonstrates:

* End-to-end MLOps workflow
* Infrastructure-as-code
* Cloud-native architecture (AWS)
* Automated CI/CD & monitoring

It’s designed to be production-like and hands-off – deployable via GitHub with minimal local setup.
