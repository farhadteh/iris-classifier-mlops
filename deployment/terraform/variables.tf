# ===============================================================================
# Terraform Variables for MLflow Infrastructure
# ===============================================================================
# This file defines all the configurable parameters for the MLflow infrastructure
# Customize these values in terraform.tfvars or through environment variables

# ===============================================================================
# General Configuration
# ===============================================================================

variable "project_name" {
  description = "Name of the MLflow project (used for resource naming)"
  type        = string
  default     = "mlflow-iris"
  
  validation {
    condition     = length(var.project_name) > 0 && length(var.project_name) <= 30
    error_message = "Project name must be between 1 and 30 characters."
  }
}

variable "environment" {
  description = "Environment name (dev, staging, prod)"
  type        = string
  default     = "dev"
  
  validation {
    condition = contains(["dev", "staging", "prod"], var.environment)
    error_message = "Environment must be one of: dev, staging, prod."
  }
}

variable "owner" {
  description = "Owner of the infrastructure (for tagging)"
  type        = string
  default     = "mlops-team"
}

variable "aws_region" {
  description = "AWS region for all resources"
  type        = string
  default     = "us-west-2"
}

# ===============================================================================
# Network Configuration
# ===============================================================================

variable "vpc_cidr" {
  description = "CIDR block for the VPC"
  type        = string
  default     = "10.0.0.0/16"
  
  validation {
    condition     = can(cidrhost(var.vpc_cidr, 0))
    error_message = "VPC CIDR must be a valid IPv4 CIDR block."
  }
}

variable "create_nat_gateway" {
  description = "Whether to create NAT Gateway for private subnets (incurs costs)"
  type        = bool
  default     = false
}

variable "enable_deletion_protection" {
  description = "Enable deletion protection for ALB (recommended for production)"
  type        = bool
  default     = false
}

# ===============================================================================
# ECS Configuration
# ===============================================================================

variable "ecs_cluster_name" {
  description = "Name of the ECS cluster"
  type        = string
  default     = null # Will use project_name-environment if not specified
}

variable "ecs_capacity_providers" {
  description = "ECS capacity providers to use"
  type        = list(string)
  default     = ["FARGATE", "FARGATE_SPOT"]
}

variable "enable_container_insights" {
  description = "Enable CloudWatch Container Insights for ECS cluster"
  type        = bool
  default     = true
}

# ===============================================================================
# Container Configuration
# ===============================================================================

variable "container_cpu" {
  description = "CPU units for ECS tasks (1 vCPU = 1024 units)"
  type        = number
  default     = 512
  
  validation {
    condition = contains([256, 512, 1024, 2048, 4096], var.container_cpu)
    error_message = "CPU must be one of: 256, 512, 1024, 2048, 4096."
  }
}

variable "container_memory" {
  description = "Memory (MB) for ECS tasks"
  type        = number
  default     = 1024
  
  validation {
    condition = var.container_memory >= 512 && var.container_memory <= 30720
    error_message = "Memory must be between 512 MB and 30720 MB."
  }
}

variable "container_port" {
  description = "Port number for container applications"
  type        = number
  default     = 8000
}

variable "desired_count" {
  description = "Desired number of ECS tasks"
  type        = number
  default     = 1
  
  validation {
    condition = var.desired_count >= 1 && var.desired_count <= 10
    error_message = "Desired count must be between 1 and 10."
  }
}

variable "min_capacity" {
  description = "Minimum number of tasks for auto scaling"
  type        = number
  default     = 1
}

variable "max_capacity" {
  description = "Maximum number of tasks for auto scaling"
  type        = number
  default     = 10
}

variable "auto_scaling_target_cpu" {
  description = "Target CPU utilization for auto scaling (%)"
  type        = number
  default     = 70
  
  validation {
    condition = var.auto_scaling_target_cpu >= 10 && var.auto_scaling_target_cpu <= 90
    error_message = "Auto scaling target CPU must be between 10% and 90%."
  }
}

variable "auto_scaling_target_memory" {
  description = "Target memory utilization for auto scaling (%)"
  type        = number
  default     = 80
  
  validation {
    condition = var.auto_scaling_target_memory >= 10 && var.auto_scaling_target_memory <= 90
    error_message = "Auto scaling target memory must be between 10% and 90%."
  }
}

# ===============================================================================
# Database Configuration
# ===============================================================================

variable "create_rds" {
  description = "Whether to create RDS instance for MLflow backend store"
  type        = bool
  default     = false
}

variable "db_instance_class" {
  description = "RDS instance class"
  type        = string
  default     = "db.t3.micro"
}

variable "db_engine" {
  description = "Database engine (mysql, postgresql)"
  type        = string
  default     = "mysql"
  
  validation {
    condition = contains(["mysql", "postgresql"], var.db_engine)
    error_message = "Database engine must be either 'mysql' or 'postgresql'."
  }
}

variable "db_engine_version" {
  description = "Database engine version"
  type        = string
  default     = "8.0"
}

variable "db_allocated_storage" {
  description = "RDS allocated storage (GB)"
  type        = number
  default     = 20
  
  validation {
    condition = var.db_allocated_storage >= 20 && var.db_allocated_storage <= 1000
    error_message = "Database storage must be between 20 GB and 1000 GB."
  }
}

variable "db_max_allocated_storage" {
  description = "RDS maximum allocated storage for autoscaling (GB)"
  type        = number
  default     = 100
}

variable "db_name" {
  description = "Name of the database to create"
  type        = string
  default     = "mlflow"
}

variable "db_username" {
  description = "Master username for the database"
  type        = string
  default     = "mlflow_user"
  sensitive   = true
}

variable "db_password" {
  description = "Master password for the database"
  type        = string
  default     = null
  sensitive   = true
}

variable "db_backup_retention_period" {
  description = "Backup retention period (days)"
  type        = number
  default     = 7
  
  validation {
    condition = var.db_backup_retention_period >= 0 && var.db_backup_retention_period <= 35
    error_message = "Backup retention period must be between 0 and 35 days."
  }
}

variable "db_backup_window" {
  description = "Preferred backup window"
  type        = string
  default     = "03:00-04:00"
}

variable "db_maintenance_window" {
  description = "Preferred maintenance window"
  type        = string
  default     = "sun:04:00-sun:05:00"
}

variable "db_deletion_protection" {
  description = "Enable deletion protection for RDS instance"
  type        = bool
  default     = false
}

# ===============================================================================
# Storage Configuration
# ===============================================================================

variable "s3_bucket_name" {
  description = "Name of S3 bucket for MLflow artifacts (auto-generated if not specified)"
  type        = string
  default     = null
}

variable "s3_versioning_enabled" {
  description = "Enable versioning for S3 bucket"
  type        = bool
  default     = true
}

variable "s3_lifecycle_enabled" {
  description = "Enable lifecycle management for S3 bucket"
  type        = bool
  default     = true
}

variable "s3_transition_to_ia_days" {
  description = "Days after which objects transition to IA storage class"
  type        = number
  default     = 30
}

variable "s3_transition_to_glacier_days" {
  description = "Days after which objects transition to Glacier storage class"
  type        = number
  default     = 90
}

variable "s3_expiration_days" {
  description = "Days after which objects expire (0 = never expire)"
  type        = number
  default     = 0
}

# ===============================================================================
# Monitoring and Logging
# ===============================================================================

variable "log_retention_days" {
  description = "CloudWatch log retention period (days)"
  type        = number
  default     = 7
  
  validation {
    condition = contains([1, 3, 5, 7, 14, 30, 60, 90, 120, 150, 180, 365, 400, 545, 731, 1827, 3653], var.log_retention_days)
    error_message = "Log retention days must be a valid CloudWatch retention period."
  }
}

variable "enable_xray_tracing" {
  description = "Enable AWS X-Ray tracing for applications"
  type        = bool
  default     = false
}

variable "create_cloudwatch_dashboard" {
  description = "Create CloudWatch dashboard for monitoring"
  type        = bool
  default     = true
}

variable "alarm_email" {
  description = "Email address for CloudWatch alarms"
  type        = string
  default     = null
}

# ===============================================================================
# Security Configuration
# ===============================================================================

variable "allowed_cidr_blocks" {
  description = "CIDR blocks allowed to access the load balancer"
  type        = list(string)
  default     = ["0.0.0.0/0"]
}

variable "enable_waf" {
  description = "Enable AWS WAF for the load balancer"
  type        = bool
  default     = false
}

variable "ssl_certificate_arn" {
  description = "ARN of SSL certificate for HTTPS listener"
  type        = string
  default     = null
}

variable "enable_https" {
  description = "Enable HTTPS listener (requires ssl_certificate_arn)"
  type        = bool
  default     = false
}

# ===============================================================================
# Feature Flags
# ===============================================================================

variable "enable_auto_scaling" {
  description = "Enable auto scaling for ECS services"
  type        = bool
  default     = true
}

variable "enable_load_balancer" {
  description = "Create Application Load Balancer"
  type        = bool
  default     = true
}

variable "enable_service_discovery" {
  description = "Enable AWS Cloud Map service discovery"
  type        = bool
  default     = false
}

variable "create_bastion_host" {
  description = "Create bastion host for secure access to private resources"
  type        = bool
  default     = false
}

# ===============================================================================
# Docker Configuration
# ===============================================================================

variable "docker_image_tag" {
  description = "Docker image tag for MLflow services"
  type        = string
  default     = "latest"
}

variable "ecr_repository_name" {
  description = "ECR repository name for custom images"
  type        = string
  default     = null
}

variable "force_new_deployment" {
  description = "Force new deployment of ECS services"
  type        = bool
  default     = false
}

# ===============================================================================
# Cost Optimization
# ===============================================================================

variable "use_spot_instances" {
  description = "Use Spot instances in ECS (cost optimization)"
  type        = bool
  default     = false
}

variable "spot_instance_percentage" {
  description = "Percentage of capacity to provision using Spot instances"
  type        = number
  default     = 50
  
  validation {
    condition = var.spot_instance_percentage >= 0 && var.spot_instance_percentage <= 100
    error_message = "Spot instance percentage must be between 0 and 100."
  }
}

# ===============================================================================
# Development and Testing
# ===============================================================================

variable "enable_debug_mode" {
  description = "Enable debug mode for applications"
  type        = bool
  default     = false
}

variable "create_test_data" {
  description = "Create test data in S3 bucket"
  type        = bool
  default     = false
}

variable "local_development" {
  description = "Configure for local development (uses Docker provider)"
  type        = bool
  default     = false
}

# ===============================================================================
# Advanced Configuration
# ===============================================================================

variable "custom_tags" {
  description = "Additional custom tags to apply to all resources"
  type        = map(string)
  default     = {}
}

variable "enable_encryption_at_rest" {
  description = "Enable encryption at rest for all supported services"
  type        = bool
  default     = true
}

variable "enable_encryption_in_transit" {
  description = "Enable encryption in transit for all supported services"
  type        = bool
  default     = true
}

variable "compliance_mode" {
  description = "Enable compliance mode (additional security and logging)"
  type        = bool
  default     = false
}

# ===============================================================================
# Service-Specific Configuration
# ===============================================================================

variable "mlflow_server_config" {
  description = "Configuration for MLflow server"
  type = object({
    port              = optional(number, 5000)
    workers           = optional(number, 1)
    artifacts_destination = optional(string, "s3")
    backend_store_uri = optional(string, "sqlite")
  })
  default = {}
}

variable "fastapi_config" {
  description = "Configuration for FastAPI application"
  type = object({
    port     = optional(number, 8000)
    workers  = optional(number, 1)
    log_level = optional(string, "info")
  })
  default = {}
}

variable "streamlit_config" {
  description = "Configuration for Streamlit application"
  type = object({
    port              = optional(number, 8501)
    max_upload_size   = optional(number, 200)
    theme_base        = optional(string, "light")
  })
  default = {}
}

# ===============================================================================
# Environment-Specific Overrides
# ===============================================================================

variable "dev_overrides" {
  description = "Development environment specific overrides"
  type = object({
    instance_type        = optional(string, "t3.micro")
    min_capacity        = optional(number, 1)
    max_capacity        = optional(number, 2)
    enable_deletion_protection = optional(bool, false)
    db_backup_retention = optional(number, 1)
  })
  default = {}
}

variable "staging_overrides" {
  description = "Staging environment specific overrides"
  type = object({
    instance_type        = optional(string, "t3.small")
    min_capacity        = optional(number, 1)
    max_capacity        = optional(number, 3)
    enable_deletion_protection = optional(bool, false)
    db_backup_retention = optional(number, 3)
  })
  default = {}
}

variable "prod_overrides" {
  description = "Production environment specific overrides"
  type = object({
    instance_type        = optional(string, "t3.medium")
    min_capacity        = optional(number, 2)
    max_capacity        = optional(number, 10)
    enable_deletion_protection = optional(bool, true)
    db_backup_retention = optional(number, 30)
  })
  default = {}
}
