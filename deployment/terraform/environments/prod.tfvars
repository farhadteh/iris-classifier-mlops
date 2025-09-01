# ===============================================================================
# Production Environment Configuration
# ===============================================================================
# This configuration is optimized for production workloads
# High availability, security, and performance focused

# ===============================================================================
# Basic Configuration
# ===============================================================================

project_name = "mlflow-production"
environment  = "prod"
aws_region   = "us-west-2"
owner        = "mlops-team"

# ===============================================================================
# High Availability Infrastructure
# ===============================================================================

# Full production networking
create_nat_gateway = true

# Production database with high availability
create_rds                   = true
db_engine                   = "mysql"
db_engine_version           = "8.0"
db_instance_class           = "db.t3.medium"  # More powerful for production
db_allocated_storage        = 100
db_max_allocated_storage    = 500
db_backup_retention_period  = 30              # Long retention for compliance
db_deletion_protection      = true            # Protect production data

# ===============================================================================
# Container Configuration
# ===============================================================================

# Production-grade resources
container_cpu    = 1024   # 1 vCPU
container_memory = 2048   # 2 GB RAM

# High availability with multiple instances
desired_count = 2     # Always run at least 2 instances
min_capacity  = 2
max_capacity  = 10

# Aggressive auto-scaling for performance
enable_auto_scaling = true
auto_scaling_target_cpu    = 60    # Scale earlier
auto_scaling_target_memory = 70

# ===============================================================================
# Security Configuration
# ===============================================================================

# Enable all protection mechanisms
enable_deletion_protection = true

# Restrict access to corporate networks only
# Replace with your actual corporate CIDR blocks
allowed_cidr_blocks = [
  "10.0.0.0/8",      # Corporate network
  "192.168.1.0/24"   # Office network - REPLACE WITH YOUR ACTUAL NETWORKS
]

# Enable all encryption
enable_encryption_at_rest     = true
enable_encryption_in_transit  = true

# Enable WAF for additional protection
enable_waf = true

# Production compliance mode
compliance_mode = true

# ===============================================================================
# Database Security and Performance
# ===============================================================================

db_name     = "mlflow_production"
db_username = "mlflow_prod_user"
# Strong password will be auto-generated and stored securely

# Optimized backup windows (during low traffic)
db_backup_window      = "03:00-04:00"
db_maintenance_window = "sun:04:00-sun:05:00"

# ===============================================================================
# Monitoring and Observability
# ===============================================================================

# Full monitoring suite
log_retention_days = 30  # Longer retention for production

create_cloudwatch_dashboard = true
enable_container_insights   = true
enable_xray_tracing        = true

# Set up alerting
alarm_email = "ops-team@company.com"  # REPLACE WITH YOUR EMAIL

# ===============================================================================
# Storage Configuration
# ===============================================================================

# Production S3 configuration with lifecycle management
s3_versioning_enabled = true
s3_lifecycle_enabled  = true

# Optimize costs with intelligent tiering
s3_transition_to_ia_days     = 30
s3_transition_to_glacier_days = 90
s3_expiration_days           = 2555  # 7 years retention for compliance

# ===============================================================================
# Performance Configuration
# ===============================================================================

# Service configuration for production loads
mlflow_server_config = {
  port              = 5000
  workers           = 2                    # Multiple workers
  artifacts_destination = "s3"
  backend_store_uri = "mysql"
}

fastapi_config = {
  port      = 8000
  workers   = 2                           # Multiple workers
  log_level = "warning"                   # Less verbose for production
}

streamlit_config = {
  port              = 8501
  max_upload_size   = 200                 # Higher limit for production
  theme_base        = "light"
}

# ===============================================================================
# Advanced Features
# ===============================================================================

# Enable all production features
enable_load_balancer    = true
enable_service_discovery = true         # For service mesh
create_bastion_host     = true          # Secure access to private resources

# HTTPS configuration (requires SSL certificate)
enable_https = false  # Set to true after obtaining SSL certificate
# ssl_certificate_arn = "arn:aws:acm:us-west-2:123456789012:certificate/12345678-1234-1234-1234-123456789012"

# ===============================================================================
# Cost Optimization for Production
# ===============================================================================

# Use mix of On-Demand and Spot for cost optimization
use_spot_instances = true
spot_instance_percentage = 30  # 30% spot instances for cost savings

# ===============================================================================
# Disaster Recovery and Business Continuity
# ===============================================================================

# Environment-specific overrides for production
prod_overrides = {
  instance_type        = "t3.large"
  min_capacity        = 2
  max_capacity        = 10
  enable_deletion_protection = true
  db_backup_retention = 30
}

# ===============================================================================
# Custom Tags for Production
# ===============================================================================

custom_tags = {
  Purpose         = "Production MLflow Infrastructure"
  Team           = "MLOps Production Team"
  CostCenter     = "Machine Learning"
  Environment    = "Production"
  Compliance     = "SOC2"
  BackupPolicy   = "daily"
  DisasterRecovery = "enabled"
  SLA            = "99.9%"
  DataClassification = "confidential"
  BusinessUnit   = "Data Science"
}

# ===============================================================================
# Production Checklist and Documentation
# ===============================================================================

# Before deploying to production:
# 
# SECURITY:
# [ ] Review and update allowed_cidr_blocks with actual corporate networks
# [ ] Set up SSL certificate in AWS Certificate Manager
# [ ] Configure alarm_email with actual operations team email
# [ ] Enable AWS GuardDuty for threat detection
# [ ] Set up AWS Config for compliance monitoring
# [ ] Configure VPC Flow Logs
# [ ] Review IAM policies for least privilege
# 
# MONITORING:
# [ ] Set up external monitoring (Datadog, New Relic, etc.)
# [ ] Configure log aggregation (ELK stack, Splunk, etc.)
# [ ] Set up alerting for critical metrics
# [ ] Create runbooks for common issues
# [ ] Test alert escalation procedures
# 
# BACKUP AND DR:
# [ ] Test database backup and restore procedures
# [ ] Verify S3 cross-region replication if needed
# [ ] Document disaster recovery procedures
# [ ] Test failover procedures
# [ ] Set up automated backup verification
# 
# PERFORMANCE:
# [ ] Conduct load testing
# [ ] Tune auto-scaling parameters based on actual usage
# [ ] Optimize container resource allocation
# [ ] Review and optimize database performance
# [ ] Set up performance baselines and alerting
# 
# COMPLIANCE:
# [ ] Data encryption verification
# [ ] Access control audit
# [ ] Network security assessment
# [ ] Compliance documentation
# [ ] Regular security scans

# ===============================================================================
# Cost Estimation
# ===============================================================================

# Estimated monthly cost: $200-400
# - ECS Fargate (2 tasks): ~$60-80
# - ALB: ~$16.50
# - RDS db.t3.medium: ~$50-70
# - NAT Gateway: ~$32
# - S3 Storage: ~$10-20 (depends on data volume)
# - CloudWatch: ~$10-20
# - Data Transfer: ~$10-30
# - Backup Storage: ~$5-15
# - WAF (if enabled): ~$5-10
#
# Total: Approximately $200-300/month for moderate usage
# Scale up costs with increased traffic and data volume
#
# Cost optimization strategies:
# 1. Use Reserved Instances for predictable workloads
# 2. Implement intelligent S3 lifecycle policies
# 3. Monitor and optimize container resource usage
# 4. Use Spot instances where appropriate
# 5. Regular cost reviews and optimization
