# 🚀 Terraform Infrastructure Guide for MLflow

> **Complete Infrastructure as Code Setup for MLflow MLOps Pipeline**

Welcome to the comprehensive Terraform guide for your MLflow project! This guide will take you through setting up professional cloud infrastructure step by step, even if you're completely new to Terraform.

## 📋 Table of Contents

1. [What is Terraform?](#-what-is-terraform)
2. [Why Use Terraform for MLflow?](#-why-use-terraform-for-mlflow)
3. [Prerequisites](#-prerequisites)
4. [Quick Start Guide](#-quick-start-guide)
5. [Step-by-Step Setup](#-step-by-step-setup)
6. [Configuration Options](#-configuration-options)
7. [Deployment Scenarios](#-deployment-scenarios)
8. [Troubleshooting](#-troubleshooting)
9. [Best Practices](#-best-practices)
10. [Advanced Topics](#-advanced-topics)

## 🤔 What is Terraform?

**Terraform** is an Infrastructure as Code (IaC) tool that allows you to define and manage cloud infrastructure using code instead of manual clicking in web consoles.

### Key Benefits:
- **🔄 Reproducible**: Same infrastructure every time
- **📝 Version Controlled**: Track changes like source code  
- **🤝 Collaborative**: Team can work together
- **💰 Cost Effective**: Easy to destroy and recreate resources
- **📊 Predictable**: See changes before applying them

### How it Works:
```mermaid
graph LR
    A[Write .tf Files] --> B[terraform plan]
    B --> C[Review Changes]
    C --> D[terraform apply]
    D --> E[Cloud Resources Created]
```

## 🎯 Why Use Terraform for MLflow?

Your MLflow project needs several cloud resources:

| Component | AWS Service | Why Needed |
|-----------|-------------|------------|
| **Container Hosting** | ECS Fargate | Run MLflow, FastAPI, Streamlit |
| **Load Balancer** | Application Load Balancer | Route traffic to services |
| **Storage** | S3 | Store ML models and artifacts |
| **Database** | RDS (Optional) | MLflow metadata storage |
| **Networking** | VPC, Subnets | Secure network isolation |
| **Monitoring** | CloudWatch | Logs and metrics |

**Without Terraform**: Manual setup, inconsistent environments, hard to reproduce
**With Terraform**: Automated, consistent, version-controlled infrastructure

## 🛠️ Prerequisites

### 1. Install Required Tools

#### **Install Terraform**
```bash
# macOS (using Homebrew)
brew tap hashicorp/tap
brew install hashicorp/tap/terraform

# Windows (using Chocolatey)
choco install terraform

# Linux (Ubuntu/Debian)
wget -O- https://apt.releases.hashicorp.com/gpg | gpg --dearmor | sudo tee /usr/share/keyrings/hashicorp-archive-keyring.gpg
echo "deb [signed-by=/usr/share/keyrings/hashicorp-archive-keyring.gpg] https://apt.releases.hashicorp.com $(lsb_release -cs) main" | sudo tee /etc/apt/sources.list.d/hashicorp.list
sudo apt update && sudo apt install terraform

# Verify installation
terraform --version
```

#### **Install AWS CLI**
```bash
# macOS
brew install awscli

# Windows
# Download from: https://awscli.amazonaws.com/AWSCLIV2.msi

# Linux
curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip"
unzip awscliv2.zip
sudo ./aws/install

# Verify installation
aws --version
```

### 2. AWS Account Setup

#### **Create AWS Account**
1. Go to [aws.amazon.com](https://aws.amazon.com)
2. Create account (requires credit card)
3. **Note**: AWS Free Tier includes many services we'll use

#### **Create IAM User with Programmatic Access**
1. Go to AWS Console → IAM → Users → Add User
2. Username: `terraform-user`
3. Access type: ✅ Programmatic access
4. Permissions: Attach existing policies directly
5. Add these policies:
   - `AmazonECS_FullAccess`
   - `AmazonVPCFullAccess`
   - `AmazonS3FullAccess`
   - `AmazonRDSFullAccess`
   - `ElasticLoadBalancingFullAccess`
   - `CloudWatchFullAccess`
   - `IAMFullAccess`
6. Download the `.csv` file with Access Key ID and Secret

#### **Configure AWS CLI**
```bash
aws configure
# AWS Access Key ID: [Your Access Key]
# AWS Secret Access Key: [Your Secret Key]  
# Default region name: us-west-2
# Default output format: json

# Test configuration
aws sts get-caller-identity
```

## 🚀 Quick Start Guide

### **Option 1: Minimal Setup (Recommended for Beginners)**

```bash
# 1. Navigate to terraform directory
cd deployment/terraform

# 2. Create a simple configuration file
cat > terraform.tfvars << EOF
project_name = "my-mlflow"
environment  = "dev"
aws_region   = "us-west-2"
owner        = "your-name"

# Cost optimization for learning
create_nat_gateway = false
create_rds = false
container_cpu = 256
container_memory = 512
desired_count = 1
EOF

# 3. Initialize Terraform
terraform init

# 4. See what will be created
terraform plan

# 5. Create the infrastructure
terraform apply
```

### **Option 2: Full Production Setup**

```bash
# 1. Navigate to terraform directory
cd deployment/terraform

# 2. Create production configuration
cat > terraform.tfvars << EOF
project_name = "mlflow-prod"
environment  = "prod"
aws_region   = "us-west-2"
owner        = "mlops-team"

# Production settings
create_nat_gateway = true
create_rds = true
db_engine = "mysql"
container_cpu = 1024
container_memory = 2048
desired_count = 2
enable_auto_scaling = true
enable_deletion_protection = true
EOF

# 3. Initialize and apply
terraform init
terraform plan
terraform apply
```

## 📚 Step-by-Step Setup

### Step 1: Understanding the File Structure

```
deployment/terraform/
├── main.tf              # Main infrastructure definition
├── variables.tf         # Configuration parameters
├── outputs.tf           # Information displayed after deployment
├── terraform.tfvars     # Your specific values (you create this)
└── environments/        # Environment-specific configs
    ├── dev.tfvars
    ├── staging.tfvars
    └── prod.tfvars
```

### Step 2: Choose Your Configuration

Create a `terraform.tfvars` file with your settings:

#### **Beginner Configuration (Low Cost)**
```hcl
# Basic Settings
project_name = "my-mlflow-demo"
environment  = "dev"
aws_region   = "us-west-2"
owner        = "your-name"

# Cost-saving options
create_nat_gateway = false          # Saves ~$32/month
create_rds = false                  # Use SQLite instead
container_cpu = 256                 # Minimum CPU
container_memory = 512              # Minimum memory
desired_count = 1                   # Single instance
enable_deletion_protection = false  # Easy to delete
```

#### **Production Configuration**
```hcl
# Basic Settings
project_name = "mlflow-production"
environment  = "prod"
aws_region   = "us-west-2"
owner        = "mlops-team"

# Production options
create_nat_gateway = true
create_rds = true
db_engine = "mysql"
db_instance_class = "db.t3.medium"
container_cpu = 1024
container_memory = 2048
desired_count = 2
enable_auto_scaling = true
enable_deletion_protection = true

# Security
allowed_cidr_blocks = ["10.0.0.0/8"]  # Restrict access
enable_encryption_at_rest = true
```

### Step 3: Initialize Terraform

```bash
# Navigate to terraform directory
cd deployment/terraform

# Initialize (downloads required providers)
terraform init

# You should see:
# ✅ Terraform has been successfully initialized!
```

### Step 4: Plan Your Infrastructure

```bash
# See what Terraform will create
terraform plan

# For detailed output
terraform plan -out=tfplan

# You'll see something like:
# Plan: 25 to add, 0 to change, 0 to destroy.
```

**🔍 Understanding the Plan Output:**
- `+` = Will be created
- `~` = Will be modified  
- `-` = Will be destroyed
- Numbers show total changes

### Step 5: Apply the Configuration

```bash
# Apply the changes
terraform apply

# Or apply the saved plan
terraform apply tfplan

# Type 'yes' when prompted
# Wait 5-10 minutes for completion
```

### Step 6: Access Your Infrastructure

After successful deployment, Terraform will output URLs:

```
service_urls = {
  "fastapi_api" = "http://your-alb-dns-name.region.elb.amazonaws.com/api"
  "fastapi_docs" = "http://your-alb-dns-name.region.elb.amazonaws.com/docs"
  "mlflow_ui" = "http://your-alb-dns-name.region.elb.amazonaws.com/"
  "streamlit_app" = "http://your-alb-dns-name.region.elb.amazonaws.com/streamlit"
}
```

## ⚙️ Configuration Options

### Essential Variables

| Variable | Description | Default | Options |
|----------|-------------|---------|---------|
| `project_name` | Name for your resources | `mlflow-iris` | Any string (no spaces) |
| `environment` | Environment type | `dev` | `dev`, `staging`, `prod` |
| `aws_region` | AWS region | `us-west-2` | Any AWS region |
| `container_cpu` | CPU for containers | `512` | `256`, `512`, `1024`, `2048`, `4096` |
| `container_memory` | Memory for containers | `1024` | `512` to `30720` MB |

### Cost Control Variables

| Variable | Description | Cost Impact |
|----------|-------------|-------------|
| `create_nat_gateway` | Private subnet internet access | ~$32/month |
| `create_rds` | Managed database | ~$13-50/month |
| `desired_count` | Number of running tasks | $15/task/month |
| `container_cpu/memory` | Task resources | Higher = more cost |

### Security Variables

| Variable | Description | Recommendation |
|----------|-------------|----------------|
| `allowed_cidr_blocks` | Who can access ALB | Restrict for prod |
| `enable_deletion_protection` | Prevent accidental deletion | `true` for prod |
| `db_deletion_protection` | Protect database | `true` for prod |

## 🎭 Deployment Scenarios

### Scenario 1: Local Development Testing

```hcl
# File: environments/local-test.tfvars
project_name = "mlflow-local-test"
environment = "dev"
create_nat_gateway = false
create_rds = false
container_cpu = 256
container_memory = 512
desired_count = 1
enable_deletion_protection = false
```

```bash
terraform apply -var-file="environments/local-test.tfvars"
```

**Use Case**: Quick testing, learning Terraform
**Cost**: ~$15-20/month
**Duration**: Create/destroy as needed

### Scenario 2: Team Development Environment

```hcl
# File: environments/dev.tfvars
project_name = "mlflow-team-dev"
environment = "dev"
create_nat_gateway = true
create_rds = true
db_instance_class = "db.t3.micro"
container_cpu = 512
container_memory = 1024
desired_count = 1
enable_auto_scaling = false
```

```bash
terraform apply -var-file="environments/dev.tfvars"
```

**Use Case**: Shared team development
**Cost**: ~$50-70/month
**Duration**: Keep running during development

### Scenario 3: Staging Environment

```hcl
# File: environments/staging.tfvars
project_name = "mlflow-staging"
environment = "staging"
create_nat_gateway = true
create_rds = true
db_instance_class = "db.t3.small"
container_cpu = 1024
container_memory = 2048
desired_count = 1
enable_auto_scaling = true
enable_deletion_protection = false
```

**Use Case**: Pre-production testing
**Cost**: ~$80-120/month

### Scenario 4: Production Environment

```hcl
# File: environments/prod.tfvars
project_name = "mlflow-production"
environment = "prod"
create_nat_gateway = true
create_rds = true
db_instance_class = "db.t3.medium"
db_backup_retention_period = 30
container_cpu = 2048
container_memory = 4096
desired_count = 2
min_capacity = 2
max_capacity = 10
enable_auto_scaling = true
enable_deletion_protection = true
db_deletion_protection = true
```

**Use Case**: Production workloads
**Cost**: ~$200-400/month

## 🛠️ Common Commands

### Basic Operations
```bash
# Initialize
terraform init

# Format code
terraform fmt

# Validate configuration
terraform validate

# Plan changes
terraform plan

# Apply changes
terraform apply

# Show current state
terraform show

# List resources
terraform state list

# Destroy everything
terraform destroy
```

### Working with Different Environments
```bash
# Apply specific environment
terraform apply -var-file="environments/dev.tfvars"

# Plan for production
terraform plan -var-file="environments/prod.tfvars"

# Destroy development environment
terraform destroy -var-file="environments/dev.tfvars"
```

### Targeting Specific Resources
```bash
# Create only VPC and subnets
terraform apply -target=aws_vpc.main -target=aws_subnet.public

# Destroy only the database
terraform destroy -target=aws_db_instance.mlflow

# Recreate ECS services
terraform apply -replace=aws_ecs_service.services
```

## 🐛 Troubleshooting

### Common Issues and Solutions

#### **Error: "InvalidUserID.NotFound"**
```
Error: creating EC2 VPC: InvalidUserID.NotFound
```
**Solution**: Check AWS credentials
```bash
aws sts get-caller-identity
aws configure list
```

#### **Error: "UnauthorizedOperation"**
```
Error: UnauthorizedOperation: You are not authorized to perform this operation
```
**Solution**: Add required IAM permissions
- Check IAM user has necessary policies
- Verify region matches your permissions

#### **Error: "AlreadyExistsException"**
```
Error: creating ECS Cluster: ClientException: Cluster already exists
```
**Solution**: Change project name or destroy existing resources
```bash
terraform destroy
# Or change project_name in terraform.tfvars
```

#### **Error: "InvalidParameterValue"**
```
Error: creating ECS Task Definition: ClientException: Invalid parameter value
```
**Solution**: Check CPU/memory combinations
```hcl
# Valid combinations for Fargate:
# CPU: 256,  Memory: 512, 1024, 2048
# CPU: 512,  Memory: 1024-4096 (increments of 1024)
# CPU: 1024, Memory: 2048-8192 (increments of 1024)
```

#### **Error: "Timeout while waiting for state"**
```
Error: timeout while waiting for state to become 'available'
```
**Solution**: Usually resolves itself, but check AWS console for detailed error

### Debugging Commands

```bash
# Enable detailed logging
export TF_LOG=DEBUG
terraform apply

# Check specific resource
terraform state show aws_ecs_cluster.main

# Import existing resource (if created manually)
terraform import aws_vpc.main vpc-12345678

# Force unlock if stuck
terraform force-unlock <lock-id>
```

### Getting Help

```bash
# Help for specific resource
terraform plan -help

# Provider documentation
terraform providers

# Show output values
terraform output

# Check configuration
terraform validate
terraform fmt -check
```

## 📊 Monitoring Your Infrastructure

### Cost Monitoring

1. **AWS Cost Explorer**:
   - Go to AWS Console → Billing → Cost Explorer
   - Filter by tags: `Project = your-project-name`

2. **Resource Tagging**:
   All resources are automatically tagged:
   ```hcl
   Project     = var.project_name
   Environment = var.environment
   ManagedBy   = "terraform"
   Owner       = var.owner
   ```

3. **Cost Estimation**:
   ```bash
   # Use terraform output to see estimated costs
   terraform output estimated_monthly_cost
   ```

### Performance Monitoring

1. **CloudWatch Dashboard**:
   - Automatically created if `create_cloudwatch_dashboard = true`
   - Access URL provided in terraform outputs

2. **Application Logs**:
   ```bash
   # Get log group names
   terraform output cloudwatch_log_groups
   
   # View logs in AWS Console or CLI
   aws logs describe-log-groups --log-group-name-prefix "/ecs/your-project"
   ```

3. **Health Checks**:
   ```bash
   # Get health check URLs
   terraform output health_check_urls
   
   # Test manually
   curl -f http://your-alb-dns/health
   ```

## 🔒 Security Best Practices

### 1. Secure Your Terraform State

**Local State (Development)**:
```bash
# State contains sensitive data - don't commit to git
echo "*.tfstate*" >> .gitignore
echo "terraform.tfvars" >> .gitignore
```

**Remote State (Production)**:
```hcl
# In main.tf, uncomment and configure:
terraform {
  backend "s3" {
    bucket         = "your-terraform-state-bucket"
    key            = "mlflow/terraform.tfstate"
    region         = "us-west-2"
    encrypt        = true
    dynamodb_table = "terraform-locks"
  }
}
```

### 2. Secure Your AWS Credentials

```bash
# Use IAM roles instead of access keys (when possible)
# Rotate access keys regularly
# Never commit credentials to code

# For production, use:
export AWS_PROFILE=production
# Or use AWS IAM roles for EC2/Lambda
```

### 3. Network Security

```hcl
# Restrict access to your IP only
allowed_cidr_blocks = ["YOUR_IP/32"]

# Use private subnets with NAT Gateway
create_nat_gateway = true

# Enable VPC Flow Logs (add to main.tf)
resource "aws_flow_log" "vpc_flow_log" {
  iam_role_arn    = aws_iam_role.flow_log.arn
  log_destination = aws_cloudwatch_log_group.vpc_flow_log.arn
  traffic_type    = "ALL"
  vpc_id          = aws_vpc.main.id
}
```

### 4. Secrets Management

**For Database Passwords**:
```hcl
# Let Terraform generate secure passwords
db_password = null  # Uses random_password resource

# Or use AWS Secrets Manager (recommended for production)
variable "db_password_secret_arn" {
  description = "ARN of secret containing database password"
  type        = string
  default     = null
}
```

## 🔄 CI/CD Integration

### GitHub Actions Example

Create `.github/workflows/terraform.yml`:

```yaml
name: 'Terraform Infrastructure'

on:
  push:
    paths:
      - 'deployment/terraform/**'
    branches:
      - main
  pull_request:
    paths:
      - 'deployment/terraform/**'

jobs:
  terraform:
    name: 'Terraform'
    runs-on: ubuntu-latest
    
    defaults:
      run:
        working-directory: deployment/terraform
    
    steps:
    - name: Checkout
      uses: actions/checkout@v3
    
    - name: Setup Terraform
      uses: hashicorp/setup-terraform@v2
      with:
        terraform_version: 1.5.0
    
    - name: Configure AWS credentials
      uses: aws-actions/configure-aws-credentials@v2
      with:
        aws-access-key-id: ${{ secrets.AWS_ACCESS_KEY_ID }}
        aws-secret-access-key: ${{ secrets.AWS_SECRET_ACCESS_KEY }}
        aws-region: us-west-2
    
    - name: Terraform Init
      run: terraform init
    
    - name: Terraform Plan
      run: terraform plan -var-file="environments/staging.tfvars"
      if: github.event_name == 'pull_request'
    
    - name: Terraform Apply
      run: terraform apply -auto-approve -var-file="environments/prod.tfvars"
      if: github.ref == 'refs/heads/main'
```

### GitLab CI Example

Create `.gitlab-ci.yml`:

```yaml
stages:
  - validate
  - plan
  - apply

variables:
  TF_ROOT: deployment/terraform
  TF_ADDRESS: ${CI_API_V4_URL}/projects/${CI_PROJECT_ID}/terraform/state/production

terraform:validate:
  stage: validate
  image: hashicorp/terraform:latest
  script:
    - cd $TF_ROOT
    - terraform init -backend=false
    - terraform validate
    - terraform fmt -check
  only:
    changes:
      - deployment/terraform/**/*

terraform:plan:
  stage: plan
  image: hashicorp/terraform:latest
  script:
    - cd $TF_ROOT
    - terraform init
    - terraform plan -var-file="environments/prod.tfvars"
  only:
    - merge_requests

terraform:apply:
  stage: apply
  image: hashicorp/terraform:latest
  script:
    - cd $TF_ROOT
    - terraform init
    - terraform apply -auto-approve -var-file="environments/prod.tfvars"
  only:
    - main
  when: manual
```

## 📈 Scaling Your Infrastructure

### Horizontal Scaling (More Instances)
```hcl
# Increase desired count
desired_count = 3

# Enable auto scaling
enable_auto_scaling = true
min_capacity = 2
max_capacity = 10

# Adjust scaling triggers
auto_scaling_target_cpu = 60
auto_scaling_target_memory = 70
```

### Vertical Scaling (Bigger Instances)
```hcl
# Increase container resources
container_cpu = 2048    # 2 vCPU
container_memory = 4096 # 4 GB RAM

# Upgrade database
db_instance_class = "db.t3.large"
```

### Multi-Region Deployment
```hcl
# Create multiple instances with different regions
# File: environments/us-east-1.tfvars
aws_region = "us-east-1"
project_name = "mlflow-east"

# File: environments/us-west-2.tfvars  
aws_region = "us-west-2"
project_name = "mlflow-west"
```

```bash
# Deploy to multiple regions
terraform apply -var-file="environments/us-east-1.tfvars"
terraform apply -var-file="environments/us-west-2.tfvars"
```

## 🧪 Testing Your Infrastructure

### Infrastructure Testing

**1. Terraform Validate**
```bash
terraform validate
terraform fmt -check
```

**2. Checkov (Security Scanning)**
```bash
pip install checkov
checkov -f main.tf
```

**3. TFLint (Linting)**
```bash
# Install tflint
curl -s https://raw.githubusercontent.com/terraform-linters/tflint/master/install_linux.sh | bash

# Run linting
tflint
```

### Application Testing After Deployment

**1. Health Checks**
```bash
# Get URLs from terraform output
terraform output service_urls

# Test each service
curl -f http://your-alb-dns/health
curl -f http://your-alb-dns/docs
curl -f http://your-alb-dns/streamlit
```

**2. Load Testing**
```bash
# Install artillery
npm install -g artillery

# Create test config
cat > load-test.yml << EOF
config:
  target: 'http://your-alb-dns'
  phases:
    - duration: 60
      arrivalRate: 5
scenarios:
  - name: "Health check"
    requests:
      - get:
          url: "/health"
EOF

# Run load test
artillery run load-test.yml
```

## 💡 Tips and Tricks

### 1. Working with Terraform State

```bash
# List all resources
terraform state list

# Show specific resource
terraform state show aws_ecs_cluster.main

# Move resource (rename)
terraform state mv aws_instance.old aws_instance.new

# Remove resource from state (without destroying)
terraform state rm aws_instance.example

# Import existing resource
terraform import aws_vpc.main vpc-12345678
```

### 2. Terraform Workspaces

```bash
# Create workspace for different environments
terraform workspace new development
terraform workspace new staging
terraform workspace new production

# Switch workspaces
terraform workspace select development

# List workspaces
terraform workspace list

# Current workspace
terraform workspace show
```

### 3. Useful Aliases

Add to your `~/.bashrc` or `~/.zshrc`:

```bash
# Terraform shortcuts
alias tf='terraform'
alias tfinit='terraform init'
alias tfplan='terraform plan'
alias tfapply='terraform apply'
alias tfdestroy='terraform destroy'
alias tfshow='terraform show'
alias tfoutput='terraform output'

# Environment-specific
alias tfdev='terraform apply -var-file="environments/dev.tfvars"'
alias tfstaging='terraform apply -var-file="environments/staging.tfvars"'
alias tfprod='terraform apply -var-file="environments/prod.tfvars"'
```

### 4. Emergency Procedures

**Stop All Services (Cost Saving)**:
```bash
# Scale down to 0 without destroying infrastructure
terraform apply -var desired_count=0
```

**Quick Rollback**:
```bash
# Revert to previous state
terraform apply -target=aws_ecs_service.services -var desired_count=1
```

**Emergency Destroy**:
```bash
# Force destroy everything (be careful!)
terraform destroy -auto-approve
```

## 🎓 Learning Resources

### Terraform Learning Path

1. **Beginner**:
   - [Terraform Introduction](https://learn.hashicorp.com/terraform)
   - [AWS Provider Documentation](https://registry.terraform.io/providers/hashicorp/aws/latest/docs)

2. **Intermediate**:
   - [Terraform Modules](https://learn.hashicorp.com/tutorials/terraform/module-create)
   - [State Management](https://learn.hashicorp.com/tutorials/terraform/state-import)

3. **Advanced**:
   - [Terraform Cloud](https://app.terraform.io/)
   - [Policy as Code with Sentinel](https://learn.hashicorp.com/sentinel)

### AWS Learning Resources

1. **Free Tier Guide**: [AWS Free Tier](https://aws.amazon.com/free/)
2. **Architecture Center**: [AWS Architecture Center](https://aws.amazon.com/architecture/)
3. **Cost Calculator**: [AWS Pricing Calculator](https://calculator.aws/)

### MLOps and Infrastructure

1. **MLOps Principles**: [Google MLOps Guide](https://cloud.google.com/architecture/mlops-continuous-delivery-and-automation-pipelines-in-machine-learning)
2. **Infrastructure as Code**: [IaC Best Practices](https://docs.aws.amazon.com/whitepapers/latest/introduction-devops-aws/infrastructure-as-code.html)

## ❓ FAQ

### **Q: How much will this cost?**
**A**: Depends on configuration:
- **Development**: $15-30/month
- **Staging**: $50-100/month  
- **Production**: $150-400/month

Use `terraform output estimated_monthly_cost` for estimates.

### **Q: Can I use this for other ML frameworks?**
**A**: Yes! Just modify the container definitions in `main.tf` to use different Docker images.

### **Q: How do I backup my data?**
**A**: 
- **S3**: Versioning enabled by default
- **RDS**: Automated backups enabled  
- **Terraform State**: Use remote backend

### **Q: Can I run this locally for testing?**
**A**: Yes, but you'll still need AWS credentials. Consider using [LocalStack](https://localstack.cloud/) for offline development.

### **Q: How do I update the infrastructure?**
**A**: 
1. Modify `terraform.tfvars` or `.tf` files
2. Run `terraform plan` to see changes
3. Run `terraform apply` to apply changes

### **Q: What if I accidentally delete something?**
**A**: 
- Terraform can recreate most resources
- S3 has versioning for data protection
- RDS has automated backups
- Always test in development first!

### **Q: How do I add SSL/HTTPS?**
**A**:
1. Get SSL certificate from AWS Certificate Manager
2. Set `ssl_certificate_arn` variable
3. Set `enable_https = true`
4. Run `terraform apply`

## 🆘 Getting Help

### Community Support
- **Terraform Discord**: [discord.gg/terraform](https://discord.gg/terraform)
- **AWS Community**: [re:Post](https://repost.aws/)
- **Stack Overflow**: Tag questions with `terraform` and `aws`

### Professional Support
- **HashiCorp Support**: For Terraform Enterprise customers
- **AWS Support**: Various tiers available
- **Consulting**: Many DevOps consulting companies available

### Documentation
- **Terraform Docs**: [terraform.io/docs](https://terraform.io/docs)
- **AWS Docs**: [docs.aws.amazon.com](https://docs.aws.amazon.com)
- **This Project**: Check the `docs/` directory

---

## 🎉 Congratulations!

You've successfully set up professional infrastructure for your MLflow project using Terraform! You now have:

✅ **Scalable container orchestration** with ECS  
✅ **Load balancing** for high availability  
✅ **Secure networking** with VPC and security groups  
✅ **Managed storage** with S3 and optional RDS  
✅ **Monitoring and logging** with CloudWatch  
✅ **Infrastructure as Code** for consistency  

### Next Steps:
1. 🚀 **Deploy your MLflow containers** to the ECS cluster
2. 🔒 **Set up SSL certificates** for HTTPS
3. 📊 **Configure monitoring dashboards**
4. 🔄 **Integrate with CI/CD pipeline**
5. 📈 **Scale based on usage patterns**

Happy infrastructure building! 🛠️✨

---

*Made with ❤️ for the MLOps community*
