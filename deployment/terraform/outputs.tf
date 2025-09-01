# ===============================================================================
# Terraform Outputs for MLflow Infrastructure
# ===============================================================================
# This file defines all the output values that will be displayed after 
# terraform apply and can be used by other Terraform configurations

# ===============================================================================
# General Information
# ===============================================================================

output "project_info" {
  description = "General project information"
  value = {
    project_name = var.project_name
    environment  = var.environment
    aws_region   = var.aws_region
    owner        = var.owner
  }
}

output "deployment_timestamp" {
  description = "Timestamp of the deployment"
  value       = timestamp()
}

# ===============================================================================
# Networking Outputs
# ===============================================================================

output "vpc_info" {
  description = "VPC information"
  value = {
    vpc_id         = aws_vpc.main.id
    vpc_cidr_block = aws_vpc.main.cidr_block
    vpc_arn        = aws_vpc.main.arn
  }
}

output "subnet_info" {
  description = "Subnet information"
  value = {
    public_subnet_ids  = aws_subnet.public[*].id
    private_subnet_ids = aws_subnet.private[*].id
    public_subnet_cidrs = aws_subnet.public[*].cidr_block
    private_subnet_cidrs = aws_subnet.private[*].cidr_block
    availability_zones = aws_subnet.public[*].availability_zone
  }
}

output "security_group_ids" {
  description = "Security group IDs"
  value = {
    alb_security_group      = aws_security_group.alb.id
    ecs_tasks_security_group = aws_security_group.ecs_tasks.id
    rds_security_group      = var.create_rds ? aws_security_group.rds[0].id : null
  }
}

output "internet_gateway_id" {
  description = "Internet Gateway ID"
  value       = aws_internet_gateway.main.id
}

output "nat_gateway_ids" {
  description = "NAT Gateway IDs (if created)"
  value       = var.create_nat_gateway ? aws_nat_gateway.main[*].id : []
}

# ===============================================================================
# Load Balancer Outputs
# ===============================================================================

output "load_balancer_info" {
  description = "Application Load Balancer information"
  value = {
    alb_arn            = aws_lb.main.arn
    alb_dns_name       = aws_lb.main.dns_name
    alb_zone_id        = aws_lb.main.zone_id
    alb_hosted_zone_id = aws_lb.main.zone_id
  }
}

output "target_group_arns" {
  description = "Target Group ARNs for each service"
  value = {
    for service_name, service_config in local.mlflow_services :
    service_name => aws_lb_target_group.services[service_name].arn
  }
}

output "listener_arns" {
  description = "Load Balancer Listener ARNs"
  value = {
    main_listener = aws_lb_listener.main.arn
  }
}

# ===============================================================================
# Service URLs
# ===============================================================================

output "service_urls" {
  description = "URLs for accessing MLflow services"
  value = {
    mlflow_ui    = "http://${aws_lb.main.dns_name}/"
    fastapi_docs = "http://${aws_lb.main.dns_name}/docs"
    fastapi_api  = "http://${aws_lb.main.dns_name}/api"
    streamlit_app = "http://${aws_lb.main.dns_name}/streamlit"
  }
}

output "health_check_urls" {
  description = "Health check URLs for each service"
  value = {
    mlflow_health   = "http://${aws_lb.main.dns_name}/health"
    fastapi_health  = "http://${aws_lb.main.dns_name}/health"
    streamlit_health = "http://${aws_lb.main.dns_name}/streamlit"
  }
}

# ===============================================================================
# Storage Outputs
# ===============================================================================

output "s3_bucket_info" {
  description = "S3 bucket information for MLflow artifacts"
  value = {
    bucket_name   = aws_s3_bucket.mlflow_artifacts.bucket
    bucket_arn    = aws_s3_bucket.mlflow_artifacts.arn
    bucket_domain = aws_s3_bucket.mlflow_artifacts.bucket_domain_name
    bucket_region = aws_s3_bucket.mlflow_artifacts.region
  }
}

output "s3_bucket_url" {
  description = "S3 bucket URL for MLflow artifacts"
  value       = "s3://${aws_s3_bucket.mlflow_artifacts.bucket}"
}

# ===============================================================================
# Database Outputs
# ===============================================================================

output "database_info" {
  description = "Database connection information (if RDS is created)"
  value = var.create_rds ? {
    endpoint = aws_db_instance.mlflow[0].endpoint
    port     = aws_db_instance.mlflow[0].port
    database_name = aws_db_instance.mlflow[0].db_name
    username = aws_db_instance.mlflow[0].username
    # Note: Password is not included in outputs for security
  } : null
  sensitive = true
}

output "database_connection_string" {
  description = "Database connection string for MLflow (if RDS is created)"
  value = var.create_rds ? (
    var.db_engine == "mysql" ?
    "mysql://${aws_db_instance.mlflow[0].username}:<password>@${aws_db_instance.mlflow[0].endpoint}:${aws_db_instance.mlflow[0].port}/${aws_db_instance.mlflow[0].db_name}" :
    "postgresql://${aws_db_instance.mlflow[0].username}:<password>@${aws_db_instance.mlflow[0].endpoint}:${aws_db_instance.mlflow[0].port}/${aws_db_instance.mlflow[0].db_name}"
  ) : null
  sensitive = true
}

# ===============================================================================
# ECS Outputs
# ===============================================================================

output "ecs_cluster_info" {
  description = "ECS cluster information"
  value = {
    cluster_name = aws_ecs_cluster.main.name
    cluster_arn  = aws_ecs_cluster.main.arn
    cluster_id   = aws_ecs_cluster.main.id
  }
}

output "ecs_service_arns" {
  description = "ECS service ARNs"
  value = {
    for service_name, service_config in local.mlflow_services :
    service_name => aws_ecs_service.services[service_name].id
  }
}

output "ecs_task_definition_arns" {
  description = "ECS task definition ARNs"
  value = {
    for service_name, service_config in local.mlflow_services :
    service_name => aws_ecs_task_definition.services[service_name].arn
  }
}

# ===============================================================================
# IAM Outputs
# ===============================================================================

output "iam_role_arns" {
  description = "IAM role ARNs"
  value = {
    ecs_task_execution_role = aws_iam_role.ecs_task_execution_role.arn
    ecs_task_role          = aws_iam_role.ecs_task_role.arn
  }
}

output "iam_policy_arns" {
  description = "IAM policy ARNs"
  value = {
    ecs_task_policy = aws_iam_policy.ecs_task_policy.arn
  }
}

# ===============================================================================
# Monitoring Outputs
# ===============================================================================

output "cloudwatch_log_groups" {
  description = "CloudWatch log group names"
  value = {
    for service_name, service_config in local.mlflow_services :
    service_name => aws_cloudwatch_log_group.mlflow[service_name].name
  }
}

output "cloudwatch_dashboard_url" {
  description = "CloudWatch dashboard URL (if created)"
  value = var.create_cloudwatch_dashboard ? (
    "https://${var.aws_region}.console.aws.amazon.com/cloudwatch/home?region=${var.aws_region}#dashboards:name=${aws_cloudwatch_dashboard.mlflow[0].dashboard_name}"
  ) : null
}

# ===============================================================================
# Cost Information
# ===============================================================================

output "estimated_monthly_cost" {
  description = "Estimated monthly cost breakdown (approximate)"
  value = {
    alb             = "~$16.50/month (basic)"
    ecs_fargate     = "~$${var.container_cpu * 0.04048 * 24 * 30 + var.container_memory * 0.004445 * 24 * 30}/month per task"
    s3_storage      = "~$0.023/GB/month (Standard)"
    cloudwatch_logs = "~$0.50/GB ingested"
    rds_cost        = var.create_rds ? "~$${var.db_instance_class == "db.t3.micro" ? "12.50" : "25.00"}/month" : "Not created"
    nat_gateway     = var.create_nat_gateway ? "~$32.40/month per NAT Gateway" : "Not created"
    total_estimate  = "Varies based on usage and configuration"
  }
}

# ===============================================================================
# Configuration Summary
# ===============================================================================

output "configuration_summary" {
  description = "Summary of key configuration settings"
  value = {
    environment                = var.environment
    container_cpu             = var.container_cpu
    container_memory          = var.container_memory
    desired_count             = var.desired_count
    auto_scaling_enabled      = var.enable_auto_scaling
    rds_enabled              = var.create_rds
    nat_gateway_enabled      = var.create_nat_gateway
    deletion_protection      = var.enable_deletion_protection
    log_retention_days       = var.log_retention_days
    backup_retention_days    = var.create_rds ? var.db_backup_retention_period : "N/A"
  }
}

# ===============================================================================
# Next Steps
# ===============================================================================

output "next_steps" {
  description = "Recommended next steps after deployment"
  value = [
    "1. Access MLflow UI at: http://${aws_lb.main.dns_name}/",
    "2. Check FastAPI documentation at: http://${aws_lb.main.dns_name}/docs",
    "3. Monitor services in CloudWatch logs",
    "4. Configure domain name and SSL certificate for production",
    "5. Set up CI/CD pipeline to deploy applications to ECS",
    "6. Configure backup strategy for S3 and RDS (if applicable)",
    "7. Review and adjust auto-scaling settings based on load"
  ]
}

# ===============================================================================
# Troubleshooting Information
# ===============================================================================

output "troubleshooting_info" {
  description = "Troubleshooting and debugging information"
  value = {
    ecs_cluster_name    = aws_ecs_cluster.main.name
    vpc_id             = aws_vpc.main.id
    security_group_ids = [aws_security_group.alb.id, aws_security_group.ecs_tasks.id]
    log_groups         = [for lg in aws_cloudwatch_log_group.mlflow : lg.name]
    s3_bucket          = aws_s3_bucket.mlflow_artifacts.bucket
    useful_commands = [
      "aws ecs list-services --cluster ${aws_ecs_cluster.main.name}",
      "aws logs describe-log-groups --log-group-name-prefix '/ecs/${local.name_prefix}'",
      "aws s3 ls s3://${aws_s3_bucket.mlflow_artifacts.bucket}/",
      "aws elbv2 describe-load-balancers --names ${aws_lb.main.name}"
    ]
  }
}

# ===============================================================================
# Environment Variables for Applications
# ===============================================================================

output "environment_variables" {
  description = "Environment variables to configure in your applications"
  value = {
    MLFLOW_S3_ENDPOINT_URL      = "https://s3.${var.aws_region}.amazonaws.com"
    MLFLOW_ARTIFACT_ROOT        = "s3://${aws_s3_bucket.mlflow_artifacts.bucket}"
    AWS_DEFAULT_REGION          = var.aws_region
    MLFLOW_TRACKING_URI         = var.create_rds ? (
      var.db_engine == "mysql" ?
      "mysql://${var.db_username}:<password>@${aws_db_instance.mlflow[0].endpoint}:${aws_db_instance.mlflow[0].port}/${var.db_name}" :
      "postgresql://${var.db_username}:<password>@${aws_db_instance.mlflow[0].endpoint}:${aws_db_instance.mlflow[0].port}/${var.db_name}"
    ) : "sqlite:///mlruns.db"
  }
  sensitive = true
}

# ===============================================================================
# Security Information
# ===============================================================================

output "security_info" {
  description = "Security-related information and recommendations"
  value = {
    vpc_cidr               = aws_vpc.main.cidr_block
    private_subnets_isolated = var.create_nat_gateway ? false : true
    encryption_at_rest     = "S3: AES256, EBS: Default encryption"
    iam_roles_created      = ["ecs-task-execution-role", "ecs-task-role"]
    security_groups_created = ["alb-sg", "ecs-tasks-sg", var.create_rds ? "rds-sg" : null]
    recommendations = [
      "Configure WAF for additional protection",
      "Enable GuardDuty for threat detection",
      "Set up AWS Config for compliance monitoring",
      "Implement least privilege IAM policies",
      "Enable VPC Flow Logs for network monitoring"
    ]
  }
}
