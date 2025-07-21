# AWS provider
terraform {
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }
}

# Configure the AWS Provider
provider "aws" {
  region     = var.aws_region
}

# S3 Buckets
resource "aws_s3_bucket" "mlflow_bucket_diabetes" {
  bucket = var.mlflow_bucket_diabetes

  tags = {
    Name        = "MLFLOW Bucket"
    Environment = "Dev"
  }
}

resource "aws_s3_bucket" "diabetes_data_bucket" {
  bucket = var.diabetes_data_bucket

  tags = {
    Name        = "Diabetes data Bucket"
    Environment = "Dev"
  }
}

resource "aws_s3_bucket" "evidently_data_bucket" {
  bucket = var.evidently_data_bucket

  tags = {
    Name        = "Evidently reports Bucket"
    Environment = "Dev"
  }
}

# VPC
resource "aws_vpc" "mlops_vpc" {
  cidr_block           = var.vpc_cidr
  enable_dns_support   = true
  enable_dns_hostnames = true

  tags = {
    Name = var.mlops_vpc_tag
  }
}

# RDS instance requires 2 different subnets to work, creating 2 subnnets and db_subnet
resource "aws_subnet" "mlops_subnet_a" {
  vpc_id            = aws_vpc.mlops_vpc.id
  cidr_block        = "10.0.1.0/24"
  availability_zone = format("%sa", var.aws_region)

  tags = {
    Name = var.mlops_subnet_a
  }
}

resource "aws_subnet" "mlops_subnet_b" {
  vpc_id            = aws_vpc.mlops_vpc.id
  cidr_block        = "10.0.2.0/24"
  availability_zone = format("%sb", var.aws_region)

  tags = {
    Name = var.mlops_subnet_b
  }
}

resource "aws_db_subnet_group" "postgres_subnet" {
  name = var.db_subnet_name
  subnet_ids = [
    aws_subnet.mlops_subnet_a.id,
    aws_subnet.mlops_subnet_b.id
  ]
}

# Creating gateway for communication
resource "aws_internet_gateway" "igw" {
  vpc_id = aws_vpc.mlops_vpc.id

  tags = {
    Name = "mlops-igw"
  }
}

resource "aws_route_table" "public_rt" {
  vpc_id = aws_vpc.mlops_vpc.id

  route {
    cidr_block = "0.0.0.0/0"
    gateway_id = aws_internet_gateway.igw.id
  }

  tags = {
    Name = "mlops-public-rt"
  }
}

# Firewall rules
resource "aws_security_group" "diabetes_security_group" {
  name        = var.security_group_name
  vpc_id      = aws_vpc.mlops_vpc.id
  description = "Security group with custom ports and protocols"

  ingress {
    description = "SSH"
    from_port   = 22
    to_port     = 22
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }

  ingress {
    description = "HTTPS"
    from_port   = 443
    to_port     = 443
    protocol    = "tcp"
    cidr_blocks = [var.vpc_cidr]
  }

  ingress {
    description = "MLflow"
    from_port   = 5000
    to_port     = 5000
    protocol    = "tcp"
    cidr_blocks = [var.vpc_cidr]
  }

  ingress {
    description = "Evidently UI custom port"
    from_port   = 8000
    to_port     = 8000
    protocol    = "tcp"
    cidr_blocks = [var.vpc_cidr]
  }

  ingress {
    description = "Airflow UI custom port"
    from_port   = 8080
    to_port     = 8080
    protocol    = "tcp"
    cidr_blocks = [var.vpc_cidr]
  }

  ingress {
    description = "Grafana UI custom port"
    from_port   = 3000
    to_port     = 3000
    protocol    = "tcp"
    cidr_blocks = [var.vpc_cidr]
  }

  ingress {
    description = "Jupyter notebook port"
    from_port   = 8888
    to_port     = 8888
    protocol    = "tcp"
    cidr_blocks = [var.vpc_cidr]
  }

  ingress {
    description = "Adminer custom port"
    from_port   = 8081
    to_port     = 8081
    protocol    = "tcp"
    cidr_blocks = [var.vpc_cidr]
  }

  ingress {
    description = "Flask access"
    from_port   = 9696
    to_port     = 9696
    protocol    = "tcp"
    cidr_blocks = [var.vpc_cidr]
  }

  ingress {
    description = "PostgreSQL access"
    from_port   = 5432
    to_port     = 5432
    protocol    = "tcp"
    cidr_blocks = [var.vpc_cidr]
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = {
    Name = var.mlops_sg_tag
  }
}

#Postgres instance
resource "aws_db_instance" "db_instance_airflow" {
  identifier        = var.db_id_airflow
  instance_class    = var.db_instance
  allocated_storage = 5
  engine            = var.engine
  engine_version    = var.engine_version

  username = var.postgres_username_airflow
  password = var.postgres_password_airflow
  db_name = var.postgres_name_airflow

  db_subnet_group_name   = aws_db_subnet_group.postgres_subnet.name
  vpc_security_group_ids = [aws_security_group.diabetes_security_group.id]

  publicly_accessible = false
  skip_final_snapshot = true
}

resource "aws_db_instance" "db_instance_mlflow" {
  identifier        = var.db_id_mlflow
  instance_class    = var.db_instance
  allocated_storage = 5
  engine            = var.engine
  engine_version    = var.engine_version

  username = var.postgres_username_mlflow
  password = var.postgres_password_mlflow
  db_name = var.postgres_name_mlflow

  db_subnet_group_name   = aws_db_subnet_group.postgres_subnet.name
  vpc_security_group_ids = [aws_security_group.diabetes_security_group.id]

  publicly_accessible = false
  skip_final_snapshot = true
}

resource "aws_db_instance" "db_instance_grafana" {
  identifier        = var.db_id_grafana
  instance_class    = var.db_instance
  allocated_storage = 5
  engine            = var.engine
  engine_version    = var.engine_version

  username = var.postgres_username_grafana
  password = var.postgres_password_grafama
  db_name = var.postgres_name_grafana

  db_subnet_group_name   = aws_db_subnet_group.postgres_subnet.name
  vpc_security_group_ids = [aws_security_group.diabetes_security_group.id]

  publicly_accessible = false
  skip_final_snapshot = true
}

# EC2 Instance
resource "aws_instance" "mlops_server" {
  ami               = var.ami
  instance_type     = var.ec2_instance_name
  key_name          = aws_key_pair.mlops_key.key_name
  subnet_id         = aws_subnet.mlops_subnet_a.id
  availability_zone = format("%sa", var.aws_region)

  vpc_security_group_ids      = [aws_security_group.diabetes_security_group.id]
  associate_public_ip_address = true

  root_block_device {
    volume_size           = 50
    volume_type           = "gp3"
    delete_on_termination = true
  }

  user_data = file("start_run.sh")

  tags = {
    Name = var.mlops_tag
  }
}

# Asssociate EC2 subnet with Internet gateway
resource "aws_route_table_association" "subnet_a_association" {
  subnet_id      = aws_subnet.mlops_subnet_a.id
  route_table_id = aws_route_table.public_rt.id
}

resource "aws_key_pair" "mlops_key" {
  key_name   = var.mlops_key_name
  public_key = file(var.mlops_key_path)
}
