output "vpc_id" {
  description = "The ID of the created VPC"
  value       = aws_vpc.mlops_vpc.id
}

output "security_group" {
  description = "The ID for security group"
  value       = aws_security_group.diabetes_security_group.id
}

output "ec2_dns_ip" {
  description = "DNS IP adress of EC2 instance"
  value       = aws_instance.mlops_server.public_dns
}

output "ec2_ssh_command" {
  description = "SSH command to connect to EC2"
  value       = "ssh -i ~/.ssh/mlops-key ubuntu@${aws_instance.mlops_server.public_ip}"
}

output "db_output" {
  description = "IP of the db instance"
  value       = aws_db_instance.db_instance_mlflow.address
}

output "db_username" {
    description = "User name for database user"
    value = aws_db_instance.db_instance_mlflow.username
}

output "db_endpoint" {
  value = aws_db_instance.db_instance_mlflow.endpoint
}