variable "aws_access_key_id" {
  description = "AWS Access Key ID"
  type        = string
  sensitive   = true
}

variable "aws_secret_access_key" {
  description = "AWS Secret Access Key"
  type        = string
  sensitive   = true
}

variable "aws_region" {
  description = "AWS region"
  default     = "eu-north-1"
}

variable "mlflow_bucket_diabetes" {
  description = "Name of the bucket for MLFLOW"
  default     = "mlflow-bucket-diabetes"
}

variable "diabetes_data_bucket" {
  description = "Name of the bucket for raw data"
  default     = "diabetes-data-bucket"
}

variable "evidently_data_bucket" {
  description = "Name of the bucket for Evidently reports"
  default     = "evidently-data-bucket"
}
variable "security_group_name" {
  description = "Name of the aws security group"
  default     = "diabetes-security-group"
}

variable "vpc_id" {
  description = "Virtual private cloud ID"
  default     = "diabetes-vpc"
}

variable "vpc_cidr" {
  description = "Subnet values for VPC"
  default     = "10.0.0.0/16"
}

variable "mlops_vpc_tag" {
  description = "Subnet values for VPC"
  default     = "mlops_vpc"
}

variable "db_subnet_name" {
  description = "Subnet name for db instances"
  default     = "mlops-subnet"
}

variable "mlops_subnet_a" {
  description = "Subnet name for subnet a"
  default     = "mlops_subnet_a"
}

variable "mlops_subnet_b" {
  description = "Subnet name for subnet b"
  default     = "mlops_subnet_b"
}

variable "engine" {
  description = "Engine name for db"
  default     = "postgres"
}

variable "engine_version" {
  description = "Engine version for db"
  default     = "15.10"
}

variable "db_id" {
  description = "ID for the created db for MLFLOW"
  default     = "mlops-1"
}

variable "db_instance" {
  description = "Instance type to use for the db"
  default     = "db.t3.micro"
}

variable "postgres_username" {
  description = "Username for postgres db"
  default     = "mlops"
}

variable "postgres_password" {
  description = "password for postgres db"
  default     = "postgresroot"
  sensitive   = true
}

variable "ami" {
  description = "Template of Ubuntu Cloud image for region eu-north-1 from: https://cloud-images.ubuntu.com/locator/ec2/"
  default     = "ami-03b371d239dfe4af4"
}

variable "ec2_instance_name" {
  description = "Instance for EC2"
  default     = "t3.micro"
}

variable "mlops_tag" {
  description = "Tag for putting together instances in aws"
  default     = "mlops-server"
}

variable "mlops_sg_tag" {
  description = "Tag for putting together instances in aws"
  default     = "mlops-sg"
}

variable "mlops_key_name" {
  description = "Name of the key for SSH"
  default     = "mlops-key"
}

variable "mlops_key_path" {
  description = "Path to the key"
  default     = "~/.ssh/mlops.pub"
}