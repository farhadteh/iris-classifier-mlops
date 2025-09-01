# ===============================================================================
# Staging Environment Configuration  
# ===============================================================================
# This configuration mimics production but with reduced resources
# Used for pre-production testing and validation

# ===============================================================================
# Basic Configuration
# ===============================================================================

project_name = "mlflow-staging"
environment  = "staging"
aws_region   = "us-west-2"
owner        = "staging-team"

# ===============================================================================
# Infrastructure Settings
# ===============================================================================

# Enable NAT Gateway for better production simulation
# Private subnets have internet access through NAT
create_nat_gateway = true

# Use managed database like production
create_rds                   = true
db_engine                   = "mysql"
db_engine_version           = "8.0"
db_instance_class           = "db.t3.micro"  # Smaller than production
db_allocated_storage        = 20
db_max_allocated_storage    = 50
db_backup_retention_period  = 3              # Shorter than production
db_deletion_protection      = false          # Allow deletion for testing

# ===============================================================================
# Container Configuration
# ===============================================================================

# Moderate resources for staging
container_cpu    = 512   # 0.5 vCPU
container_memory = 1024  # 1 GB RAM

# Single instance with auto-scaling capability
desired_count = 1
min_capacity  = 1
max_capacity  = 3

# Enable auto-scaling for testing
enable_auto_scaling = true
auto_scaling_target_cpu    = 70
auto_scaling_target_memory = 80

# ===============================================================================
# Security Settings
# ===============================================================================

# No deletion protection for staging (easier testing)
enable_deletion_protection = false

# Restrict access to internal networks (example)
allowed_cidr_blocks = ["10.0.0.0/8", "172.16.0.0/12", "192.168.0.0/16"]

# Enable encryption
enable_encryption_at_rest     = true
enable_encryption_in_transit  = true

# ===============================================================================
# Monitoring and Logging
# ===============================================================================

# Standard log retention
log_retention_days = 7

# Enable monitoring features
create_cloudwatch_dashboard = true
enable_container_insights   = true

# Disable X-Ray for cost savings
enable_xray_tracing = false

# ===============================================================================
# Database Configuration
# ===============================================================================

db_name     = "mlflow_staging"
db_username = "mlflow_staging_user"
# Password will be auto-generated

db_backup_window      = "03:00-04:00"
db_maintenance_window = "sun:04:00-sun:05:00"

# ===============================================================================
# Storage Configuration
# ===============================================================================

# Enable S3 lifecycle management
s3_versioning_enabled = true
s3_lifecycle_enabled  = true

# Transition to cheaper storage classes
s3_transition_to_ia_days     = 30
s3_transition_to_glacier_days = 90
s3_expiration_days           = 365  # Clean up old data

# ===============================================================================
# Service Configuration
# ===============================================================================

# MLflow server configuration
mlflow_server_config = {
  port              = 5000
  workers           = 1
  artifacts_destination = "s3"
  backend_store_uri = "mysql"  # Use RDS
}

# FastAPI configuration
fastapi_config = {
  port      = 8000
  workers   = 1
  log_level = "info"
}

# Streamlit configuration
streamlit_config = {
  port              = 8501
  max_upload_size   = 100
  theme_base        = "light"
}

# ===============================================================================
# Feature Flags
# ===============================================================================

# Enable additional features for testing
enable_load_balancer    = true
enable_service_discovery = false  # Test without service discovery first
create_bastion_host     = false   # No bastion needed for staging

# Test without HTTPS first, add later
enable_https = false
ssl_certificate_arn = null

# ===============================================================================
# Cost Management
# ===============================================================================

# No spot instances for staging (predictable performance)
use_spot_instances = false

# Force new deployment for testing
force_new_deployment = false

# ===============================================================================
# Custom Tags
# ===============================================================================

custom_tags = {
  Purpose      = "Staging and Pre-Production Testing"
  Team         = "MLOps Team"
  CostCenter   = "Engineering"
  Environment  = "Staging"
  AutoBackup   = "enabled"
  TestingPhase = "pre-production"
}

# ===============================================================================
# Environment-Specific Overrides
# ===============================================================================

staging_overrides = {
  instance_type        = "t3.small"
  min_capacity        = 1
  max_capacity        = 3
  enable_deletion_protection = false
  db_backup_retention = 3
}

# ===============================================================================
# Comments and Notes
# ===============================================================================

# Estimated monthly cost: $60-90
# - ECS Fargate: ~$25-35
# - ALB: ~$16.50
# - RDS db.t3.micro: ~$13
# - NAT Gateway: ~$32
# - S3: ~$2-5
# - CloudWatch: ~$3-8
#
# Total: Approximately $70-100/month
#
# Staging environment goals:
# 1. Test production-like configuration
# 2. Validate deployment procedures
# 3. Performance testing under load
# 4. Security and compliance validation
# 5. Data migration testing
#
# Testing checklist:
# - [ ] Deploy application containers
# - [ ] Test auto-scaling triggers
# - [ ] Validate database connectivity
# - [ ] Test backup and restore procedures
# - [ ] Load testing with realistic data
# - [ ] Security scanning and validation
# - [ ] Disaster recovery procedures
