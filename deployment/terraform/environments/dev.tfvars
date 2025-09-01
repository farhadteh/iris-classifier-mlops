# ===============================================================================
# Development Environment Configuration
# ===============================================================================
# This configuration is optimized for development and testing
# Cost-optimized with minimal resources for learning and experimentation

# ===============================================================================
# Basic Configuration
# ===============================================================================

project_name = "mlflow-dev"
environment  = "dev"
aws_region   = "us-west-2"
owner        = "dev-team"

# ===============================================================================
# Cost Optimization Settings
# ===============================================================================

# Disable NAT Gateway to save ~$32/month
# Tasks in private subnets won't have internet access
create_nat_gateway = false

# Use SQLite instead of RDS to save ~$13/month
# MLflow will store metadata locally in container
create_rds = false

# Minimal container resources
container_cpu    = 256   # 0.25 vCPU
container_memory = 512   # 512 MB RAM

# Single instance to minimize costs
desired_count = 1
min_capacity  = 1
max_capacity  = 2

# Disable deletion protection for easy cleanup
enable_deletion_protection = false

# ===============================================================================
# Development-Specific Settings
# ===============================================================================

# Enable debug mode for applications
enable_debug_mode = true

# Shorter log retention to save costs
log_retention_days = 3

# Single availability zone for development
# (This is handled automatically by the infrastructure)

# Allow access from anywhere for development (not recommended for production)
allowed_cidr_blocks = ["0.0.0.0/0"]

# ===============================================================================
# Optional Features (Disabled for Cost)
# ===============================================================================

# CloudWatch dashboard
create_cloudwatch_dashboard = true

# Service discovery (not needed for dev)
enable_service_discovery = false

# Auto scaling (minimal for dev)
enable_auto_scaling = false

# Container insights (adds cost)
enable_container_insights = false

# ===============================================================================
# Storage Configuration
# ===============================================================================

# Basic S3 configuration
s3_versioning_enabled = true
s3_lifecycle_enabled  = false  # Disabled to keep all data

# No transition to save costs in dev
s3_transition_to_ia_days     = 0
s3_transition_to_glacier_days = 0
s3_expiration_days           = 0

# ===============================================================================
# Service Configuration
# ===============================================================================

# MLflow server configuration
mlflow_server_config = {
  port              = 5000
  workers           = 1
  artifacts_destination = "s3"
  backend_store_uri = "sqlite"  # Local SQLite since no RDS
}

# FastAPI configuration  
fastapi_config = {
  port      = 8000
  workers   = 1
  log_level = "debug"  # Verbose logging for development
}

# Streamlit configuration
streamlit_config = {
  port              = 8501
  max_upload_size   = 50    # Smaller upload limit
  theme_base        = "light"
}

# ===============================================================================
# Custom Tags
# ===============================================================================

custom_tags = {
  Purpose     = "Development and Testing"
  Team        = "MLOps Development"
  CostCenter  = "Development"
  AutoShutdown = "true"  # Could be used by automation scripts
}

# ===============================================================================
# Comments and Notes
# ===============================================================================

# Estimated monthly cost: $15-25
# - ECS Fargate: ~$10-15
# - ALB: ~$16.50
# - S3: ~$1-3
# - CloudWatch: ~$2-5
#
# Total: Approximately $20-30/month for continuous operation
#
# Cost-saving tips:
# 1. Stop services when not in use: terraform apply -var desired_count=0
# 2. Destroy environment completely: terraform destroy
# 3. Use scheduled Lambda to stop/start services automatically
