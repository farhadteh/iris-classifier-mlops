# ===============================================================================
# MLflow Infrastructure - Main Configuration
# ===============================================================================
# This file defines the core infrastructure for the MLflow MLOps pipeline
# including compute instances, networking, storage, and container orchestration

terraform {
  required_version = ">= 1.0"
  
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
    docker = {
      source  = "kreuzwerker/docker"
      version = "~> 3.0"
    }
  }

  # Backend for storing Terraform state (uncomment and configure for production)
  # backend "s3" {
  #   bucket         = "your-terraform-state-bucket"
  #   key            = "mlflow/terraform.tfstate"
  #   region         = var.aws_region
  #   encrypt        = true
  #   dynamodb_table = "terraform-locks"
  # }
}

# ===============================================================================
# Provider Configuration
# ===============================================================================

provider "aws" {
  region = var.aws_region
  
  default_tags {
    tags = {
      Project     = var.project_name
      Environment = var.environment
      ManagedBy   = "terraform"
      Owner       = var.owner
    }
  }
}

provider "docker" {
  # Configuration for local Docker development
}

# ===============================================================================
# Data Sources
# ===============================================================================

# Get current AWS caller identity
data "aws_caller_identity" "current" {}

# Get current AWS region
data "aws_region" "current" {}

# Get latest Amazon Linux 2 AMI
data "aws_ami" "amazon_linux" {
  most_recent = true
  owners      = ["amazon"]

  filter {
    name   = "name"
    values = ["amzn2-ami-hvm-*-x86_64-gp2"]
  }

  filter {
    name   = "virtualization-type"
    values = ["hvm"]
  }
}

# Get available availability zones
data "aws_availability_zones" "available" {
  state = "available"
}

# ===============================================================================
# Local Values
# ===============================================================================

locals {
  # Common tags for all resources
  common_tags = {
    Project     = var.project_name
    Environment = var.environment
    ManagedBy   = "terraform"
    Owner       = var.owner
    Timestamp   = formatdate("YYYY-MM-DD", timestamp())
  }

  # Generate unique names for resources
  name_prefix = "${var.project_name}-${var.environment}"
  
  # MLflow services configuration
  mlflow_services = {
    mlflow_server = {
      name           = "mlflow-server"
      port           = 5000
      health_path    = "/health"
      container_port = 5000
    }
    fastapi_app = {
      name           = "fastapi-app"
      port           = 8000
      health_path    = "/health"
      container_port = 8000
    }
    streamlit_app = {
      name           = "streamlit-app"
      port           = 8501
      health_path    = "/"
      container_port = 8501
    }
  }

  # Subnets configuration
  vpc_cidr = var.vpc_cidr
  azs      = slice(data.aws_availability_zones.available.names, 0, min(length(data.aws_availability_zones.available.names), 3))
  
  public_subnet_cidrs  = [for i, az in local.azs : cidrsubnet(local.vpc_cidr, 8, i)]
  private_subnet_cidrs = [for i, az in local.azs : cidrsubnet(local.vpc_cidr, 8, i + 10)]
}

# ===============================================================================
# VPC and Networking
# ===============================================================================

resource "aws_vpc" "main" {
  cidr_block           = local.vpc_cidr
  enable_dns_hostnames = true
  enable_dns_support   = true

  tags = merge(local.common_tags, {
    Name = "${local.name_prefix}-vpc"
  })
}

# Internet Gateway
resource "aws_internet_gateway" "main" {
  vpc_id = aws_vpc.main.id

  tags = merge(local.common_tags, {
    Name = "${local.name_prefix}-igw"
  })
}

# Public Subnets
resource "aws_subnet" "public" {
  count = length(local.azs)

  vpc_id                  = aws_vpc.main.id
  cidr_block              = local.public_subnet_cidrs[count.index]
  availability_zone       = local.azs[count.index]
  map_public_ip_on_launch = true

  tags = merge(local.common_tags, {
    Name = "${local.name_prefix}-public-subnet-${count.index + 1}"
    Type = "public"
  })
}

# Private Subnets
resource "aws_subnet" "private" {
  count = length(local.azs)

  vpc_id            = aws_vpc.main.id
  cidr_block        = local.private_subnet_cidrs[count.index]
  availability_zone = local.azs[count.index]

  tags = merge(local.common_tags, {
    Name = "${local.name_prefix}-private-subnet-${count.index + 1}"
    Type = "private"
  })
}

# Elastic IPs for NAT Gateways
resource "aws_eip" "nat" {
  count = var.create_nat_gateway ? length(local.azs) : 0

  domain = "vpc"
  depends_on = [aws_internet_gateway.main]

  tags = merge(local.common_tags, {
    Name = "${local.name_prefix}-nat-eip-${count.index + 1}"
  })
}

# NAT Gateways
resource "aws_nat_gateway" "main" {
  count = var.create_nat_gateway ? length(local.azs) : 0

  allocation_id = aws_eip.nat[count.index].id
  subnet_id     = aws_subnet.public[count.index].id

  tags = merge(local.common_tags, {
    Name = "${local.name_prefix}-nat-${count.index + 1}"
  })

  depends_on = [aws_internet_gateway.main]
}

# Route Tables
resource "aws_route_table" "public" {
  vpc_id = aws_vpc.main.id

  route {
    cidr_block = "0.0.0.0/0"
    gateway_id = aws_internet_gateway.main.id
  }

  tags = merge(local.common_tags, {
    Name = "${local.name_prefix}-public-rt"
  })
}

resource "aws_route_table" "private" {
  count = length(local.azs)

  vpc_id = aws_vpc.main.id

  dynamic "route" {
    for_each = var.create_nat_gateway ? [1] : []
    content {
      cidr_block     = "0.0.0.0/0"
      nat_gateway_id = aws_nat_gateway.main[count.index].id
    }
  }

  tags = merge(local.common_tags, {
    Name = "${local.name_prefix}-private-rt-${count.index + 1}"
  })
}

# Route Table Associations
resource "aws_route_table_association" "public" {
  count = length(aws_subnet.public)

  subnet_id      = aws_subnet.public[count.index].id
  route_table_id = aws_route_table.public.id
}

resource "aws_route_table_association" "private" {
  count = length(aws_subnet.private)

  subnet_id      = aws_subnet.private[count.index].id
  route_table_id = aws_route_table.private[count.index].id
}

# ===============================================================================
# Security Groups
# ===============================================================================

# Security Group for Load Balancer
resource "aws_security_group" "alb" {
  name        = "${local.name_prefix}-alb-sg"
  description = "Security group for Application Load Balancer"
  vpc_id      = aws_vpc.main.id

  ingress {
    description = "HTTP"
    from_port   = 80
    to_port     = 80
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }

  ingress {
    description = "HTTPS"
    from_port   = 443
    to_port     = 443
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = merge(local.common_tags, {
    Name = "${local.name_prefix}-alb-sg"
  })
}

# Security Group for ECS Tasks
resource "aws_security_group" "ecs_tasks" {
  name        = "${local.name_prefix}-ecs-tasks-sg"
  description = "Security group for ECS tasks"
  vpc_id      = aws_vpc.main.id

  ingress {
    description     = "HTTP from ALB"
    from_port       = 0
    to_port         = 65535
    protocol        = "tcp"
    security_groups = [aws_security_group.alb.id]
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = merge(local.common_tags, {
    Name = "${local.name_prefix}-ecs-tasks-sg"
  })
}

# Security Group for RDS
resource "aws_security_group" "rds" {
  count = var.create_rds ? 1 : 0

  name        = "${local.name_prefix}-rds-sg"
  description = "Security group for RDS instance"
  vpc_id      = aws_vpc.main.id

  ingress {
    description     = "MySQL/Aurora"
    from_port       = 3306
    to_port         = 3306
    protocol        = "tcp"
    security_groups = [aws_security_group.ecs_tasks.id]
  }

  tags = merge(local.common_tags, {
    Name = "${local.name_prefix}-rds-sg"
  })
}

# ===============================================================================
# IAM Roles and Policies
# ===============================================================================

# ECS Task Execution Role
resource "aws_iam_role" "ecs_task_execution_role" {
  name = "${local.name_prefix}-ecs-task-execution-role"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Action = "sts:AssumeRole"
        Effect = "Allow"
        Principal = {
          Service = "ecs-tasks.amazonaws.com"
        }
      }
    ]
  })

  tags = local.common_tags
}

resource "aws_iam_role_policy_attachment" "ecs_task_execution_role_policy" {
  role       = aws_iam_role.ecs_task_execution_role.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AmazonECSTaskExecutionRolePolicy"
}

# ECS Task Role
resource "aws_iam_role" "ecs_task_role" {
  name = "${local.name_prefix}-ecs-task-role"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Action = "sts:AssumeRole"
        Effect = "Allow"
        Principal = {
          Service = "ecs-tasks.amazonaws.com"
        }
      }
    ]
  })

  tags = local.common_tags
}

# Policy for ECS tasks to access S3 and other AWS services
resource "aws_iam_policy" "ecs_task_policy" {
  name        = "${local.name_prefix}-ecs-task-policy"
  description = "Policy for ECS tasks to access AWS services"

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Action = [
          "s3:GetObject",
          "s3:PutObject",
          "s3:DeleteObject",
          "s3:ListBucket"
        ]
        Resource = [
          "${aws_s3_bucket.mlflow_artifacts.arn}",
          "${aws_s3_bucket.mlflow_artifacts.arn}/*"
        ]
      },
      {
        Effect = "Allow"
        Action = [
          "logs:CreateLogGroup",
          "logs:CreateLogStream",
          "logs:PutLogEvents"
        ]
        Resource = "arn:aws:logs:*:*:*"
      }
    ]
  })

  tags = local.common_tags
}

resource "aws_iam_role_policy_attachment" "ecs_task_policy" {
  role       = aws_iam_role.ecs_task_role.name
  policy_arn = aws_iam_policy.ecs_task_policy.arn
}

# ===============================================================================
# S3 Buckets
# ===============================================================================

# S3 bucket for MLflow artifacts
resource "aws_s3_bucket" "mlflow_artifacts" {
  bucket = "${local.name_prefix}-mlflow-artifacts-${random_string.bucket_suffix.result}"

  tags = merge(local.common_tags, {
    Name        = "MLflow Artifacts Bucket"
    Purpose     = "MLflow model and experiment artifacts storage"
  })
}

resource "random_string" "bucket_suffix" {
  length  = 8
  special = false
  upper   = false
}

resource "aws_s3_bucket_versioning" "mlflow_artifacts" {
  bucket = aws_s3_bucket.mlflow_artifacts.id
  versioning_configuration {
    status = "Enabled"
  }
}

resource "aws_s3_bucket_server_side_encryption_configuration" "mlflow_artifacts" {
  bucket = aws_s3_bucket.mlflow_artifacts.id

  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm = "AES256"
    }
  }
}

resource "aws_s3_bucket_public_access_block" "mlflow_artifacts" {
  bucket = aws_s3_bucket.mlflow_artifacts.id

  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

# ===============================================================================
# CloudWatch Log Groups
# ===============================================================================

resource "aws_cloudwatch_log_group" "mlflow" {
  for_each = local.mlflow_services

  name              = "/ecs/${local.name_prefix}/${each.value.name}"
  retention_in_days = var.log_retention_days

  tags = merge(local.common_tags, {
    Name    = "${local.name_prefix}-${each.value.name}-logs"
    Service = each.value.name
  })
}

# ===============================================================================
# Application Load Balancer
# ===============================================================================

resource "aws_lb" "main" {
  name               = "${local.name_prefix}-alb"
  internal           = false
  load_balancer_type = "application"
  security_groups    = [aws_security_group.alb.id]
  subnets            = aws_subnet.public[*].id

  enable_deletion_protection = var.enable_deletion_protection

  tags = merge(local.common_tags, {
    Name = "${local.name_prefix}-alb"
  })
}

# Target Groups
resource "aws_lb_target_group" "services" {
  for_each = local.mlflow_services

  name        = "${local.name_prefix}-${each.value.name}-tg"
  port        = each.value.port
  protocol    = "HTTP"
  vpc_id      = aws_vpc.main.id
  target_type = "ip"

  health_check {
    enabled             = true
    healthy_threshold   = 2
    unhealthy_threshold = 2
    timeout             = 5
    interval            = 30
    path                = each.value.health_path
    matcher             = "200"
  }

  tags = merge(local.common_tags, {
    Name    = "${local.name_prefix}-${each.value.name}-tg"
    Service = each.value.name
  })
}

# Listeners
resource "aws_lb_listener" "main" {
  load_balancer_arn = aws_lb.main.arn
  port              = "80"
  protocol          = "HTTP"

  default_action {
    type             = "forward"
    target_group_arn = aws_lb_target_group.services["mlflow_server"].arn
  }

  tags = local.common_tags
}

# Listener Rules for different services
resource "aws_lb_listener_rule" "fastapi" {
  listener_arn = aws_lb_listener.main.arn
  priority     = 100

  action {
    type             = "forward"
    target_group_arn = aws_lb_target_group.services["fastapi_app"].arn
  }

  condition {
    path_pattern {
      values = ["/api/*", "/docs", "/docs/*", "/redoc", "/openapi.json"]
    }
  }

  tags = local.common_tags
}

resource "aws_lb_listener_rule" "streamlit" {
  listener_arn = aws_lb_listener.main.arn
  priority     = 200

  action {
    type             = "forward"
    target_group_arn = aws_lb_target_group.services["streamlit_app"].arn
  }

  condition {
    path_pattern {
      values = ["/streamlit/*"]
    }
  }

  tags = local.common_tags
}

# ===============================================================================
# ECS Cluster and Services
# ===============================================================================

resource "aws_ecs_cluster" "main" {
  name = var.ecs_cluster_name != null ? var.ecs_cluster_name : "${local.name_prefix}-cluster"

  capacity_providers = var.ecs_capacity_providers

  default_capacity_provider_strategy {
    capacity_provider = var.use_spot_instances ? "FARGATE_SPOT" : "FARGATE"
    weight            = var.use_spot_instances ? var.spot_instance_percentage : 100
  }

  dynamic "default_capacity_provider_strategy" {
    for_each = var.use_spot_instances ? [1] : []
    content {
      capacity_provider = "FARGATE"
      weight            = 100 - var.spot_instance_percentage
    }
  }

  setting {
    name  = "containerInsights"
    value = var.enable_container_insights ? "enabled" : "disabled"
  }

  tags = merge(local.common_tags, {
    Name = "${local.name_prefix}-ecs-cluster"
  })
}

# ECS Task Definitions
resource "aws_ecs_task_definition" "services" {
  for_each = local.mlflow_services

  family                   = "${local.name_prefix}-${each.value.name}"
  network_mode            = "awsvpc"
  requires_compatibilities = ["FARGATE"]
  cpu                     = var.container_cpu
  memory                  = var.container_memory
  execution_role_arn      = aws_iam_role.ecs_task_execution_role.arn
  task_role_arn          = aws_iam_role.ecs_task_role.arn

  container_definitions = jsonencode([
    {
      name  = each.value.name
      image = var.ecr_repository_name != null ? 
        "${data.aws_caller_identity.current.account_id}.dkr.ecr.${var.aws_region}.amazonaws.com/${var.ecr_repository_name}:${var.docker_image_tag}" :
        "${each.value.name}:${var.docker_image_tag}"
      
      essential = true
      
      portMappings = [
        {
          containerPort = each.value.container_port
          hostPort      = each.value.container_port
          protocol      = "tcp"
        }
      ]

      environment = concat([
        {
          name  = "AWS_DEFAULT_REGION"
          value = var.aws_region
        },
        {
          name  = "MLFLOW_S3_ENDPOINT_URL"
          value = "https://s3.${var.aws_region}.amazonaws.com"
        },
        {
          name  = "MLFLOW_ARTIFACT_ROOT"
          value = "s3://${aws_s3_bucket.mlflow_artifacts.bucket}"
        },
        {
          name  = "ENVIRONMENT"
          value = var.environment
        },
        {
          name  = "SERVICE_NAME"
          value = each.value.name
        }
      ], var.create_rds && each.key == "mlflow_server" ? [
        {
          name  = "MLFLOW_BACKEND_STORE_URI"
          value = var.db_engine == "mysql" ?
            "mysql://${var.db_username}:${random_password.db_password[0].result}@${aws_db_instance.mlflow[0].endpoint}:${aws_db_instance.mlflow[0].port}/${var.db_name}" :
            "postgresql://${var.db_username}:${random_password.db_password[0].result}@${aws_db_instance.mlflow[0].endpoint}:${aws_db_instance.mlflow[0].port}/${var.db_name}"
        }
      ] : [])

      logConfiguration = {
        logDriver = "awslogs"
        options = {
          "awslogs-group"         = aws_cloudwatch_log_group.mlflow[each.key].name
          "awslogs-region"        = var.aws_region
          "awslogs-stream-prefix" = "ecs"
        }
      }

      healthCheck = {
        command = [
          "CMD-SHELL",
          "curl -f http://localhost:${each.value.container_port}${each.value.health_path} || exit 1"
        ]
        interval    = 30
        timeout     = 5
        retries     = 3
        startPeriod = 60
      }
    }
  ])

  tags = merge(local.common_tags, {
    Name    = "${local.name_prefix}-${each.value.name}-task"
    Service = each.value.name
  })
}

# ECS Services
resource "aws_ecs_service" "services" {
  for_each = local.mlflow_services

  name            = "${local.name_prefix}-${each.value.name}-service"
  cluster         = aws_ecs_cluster.main.id
  task_definition = aws_ecs_task_definition.services[each.key].arn
  desired_count   = var.desired_count
  launch_type     = "FARGATE"

  force_new_deployment = var.force_new_deployment

  network_configuration {
    security_groups  = [aws_security_group.ecs_tasks.id]
    subnets         = aws_subnet.private[*].id
    assign_public_ip = !var.create_nat_gateway
  }

  load_balancer {
    target_group_arn = aws_lb_target_group.services[each.key].arn
    container_name   = each.value.name
    container_port   = each.value.container_port
  }

  dynamic "service_registries" {
    for_each = var.enable_service_discovery ? [1] : []
    content {
      registry_arn = aws_service_discovery_service.services[each.key].arn
    }
  }

  depends_on = [aws_lb_listener.main]

  tags = merge(local.common_tags, {
    Name    = "${local.name_prefix}-${each.value.name}-service"
    Service = each.value.name
  })
}

# ===============================================================================
# Auto Scaling
# ===============================================================================

resource "aws_appautoscaling_target" "ecs_target" {
  for_each = var.enable_auto_scaling ? local.mlflow_services : {}

  max_capacity       = var.max_capacity
  min_capacity       = var.min_capacity
  resource_id        = "service/${aws_ecs_cluster.main.name}/${aws_ecs_service.services[each.key].name}"
  scalable_dimension = "ecs:service:DesiredCount"
  service_namespace  = "ecs"

  tags = merge(local.common_tags, {
    Name    = "${local.name_prefix}-${each.value.name}-scaling-target"
    Service = each.value.name
  })
}

resource "aws_appautoscaling_policy" "ecs_cpu_policy" {
  for_each = var.enable_auto_scaling ? local.mlflow_services : {}

  name               = "${local.name_prefix}-${each.value.name}-cpu-scaling"
  policy_type        = "TargetTrackingScaling"
  resource_id        = aws_appautoscaling_target.ecs_target[each.key].resource_id
  scalable_dimension = aws_appautoscaling_target.ecs_target[each.key].scalable_dimension
  service_namespace  = aws_appautoscaling_target.ecs_target[each.key].service_namespace

  target_tracking_scaling_policy_configuration {
    predefined_metric_specification {
      predefined_metric_type = "ECSServiceAverageCPUUtilization"
    }
    target_value = var.auto_scaling_target_cpu
  }
}

resource "aws_appautoscaling_policy" "ecs_memory_policy" {
  for_each = var.enable_auto_scaling ? local.mlflow_services : {}

  name               = "${local.name_prefix}-${each.value.name}-memory-scaling"
  policy_type        = "TargetTrackingScaling"
  resource_id        = aws_appautoscaling_target.ecs_target[each.key].resource_id
  scalable_dimension = aws_appautoscaling_target.ecs_target[each.key].scalable_dimension
  service_namespace  = aws_appautoscaling_target.ecs_target[each.key].service_namespace

  target_tracking_scaling_policy_configuration {
    predefined_metric_specification {
      predefined_metric_type = "ECSServiceAverageMemoryUtilization"
    }
    target_value = var.auto_scaling_target_memory
  }
}

# ===============================================================================
# RDS Database (Optional)
# ===============================================================================

resource "random_password" "db_password" {
  count = var.create_rds && var.db_password == null ? 1 : 0

  length  = 16
  special = true
}

resource "aws_db_subnet_group" "mlflow" {
  count = var.create_rds ? 1 : 0

  name       = "${local.name_prefix}-db-subnet-group"
  subnet_ids = aws_subnet.private[*].id

  tags = merge(local.common_tags, {
    Name = "${local.name_prefix}-db-subnet-group"
  })
}

resource "aws_db_instance" "mlflow" {
  count = var.create_rds ? 1 : 0

  identifier     = "${local.name_prefix}-db"
  engine         = var.db_engine
  engine_version = var.db_engine_version
  instance_class = var.db_instance_class

  allocated_storage     = var.db_allocated_storage
  max_allocated_storage = var.db_max_allocated_storage
  storage_type          = "gp2"
  storage_encrypted     = var.enable_encryption_at_rest

  db_name  = var.db_name
  username = var.db_username
  password = var.db_password != null ? var.db_password : random_password.db_password[0].result

  vpc_security_group_ids = [aws_security_group.rds[0].id]
  db_subnet_group_name   = aws_db_subnet_group.mlflow[0].name

  backup_retention_period = var.db_backup_retention_period
  backup_window          = var.db_backup_window
  maintenance_window     = var.db_maintenance_window

  deletion_protection = var.db_deletion_protection
  skip_final_snapshot = !var.db_deletion_protection

  tags = merge(local.common_tags, {
    Name = "${local.name_prefix}-database"
  })
}

# ===============================================================================
# Service Discovery (Optional)
# ===============================================================================

resource "aws_service_discovery_private_dns_namespace" "mlflow" {
  count = var.enable_service_discovery ? 1 : 0

  name = "${local.name_prefix}.local"
  vpc  = aws_vpc.main.id

  tags = merge(local.common_tags, {
    Name = "${local.name_prefix}-service-discovery"
  })
}

resource "aws_service_discovery_service" "services" {
  for_each = var.enable_service_discovery ? local.mlflow_services : {}

  name = each.value.name

  dns_config {
    namespace_id = aws_service_discovery_private_dns_namespace.mlflow[0].id

    dns_records {
      ttl  = 10
      type = "A"
    }

    routing_policy = "MULTIVALUE"
  }

  health_check_grace_period_seconds = 30

  tags = merge(local.common_tags, {
    Name    = "${local.name_prefix}-${each.value.name}-discovery"
    Service = each.value.name
  })
}

# ===============================================================================
# CloudWatch Dashboard (Optional)
# ===============================================================================

resource "aws_cloudwatch_dashboard" "mlflow" {
  count = var.create_cloudwatch_dashboard ? 1 : 0

  dashboard_name = "${local.name_prefix}-dashboard"

  dashboard_body = jsonencode({
    widgets = [
      {
        type   = "metric"
        x      = 0
        y      = 0
        width  = 12
        height = 6

        properties = {
          metrics = [
            for service_name in keys(local.mlflow_services) : [
              "AWS/ECS",
              "CPUUtilization",
              "ServiceName",
              "${local.name_prefix}-${local.mlflow_services[service_name].name}-service",
              "ClusterName",
              aws_ecs_cluster.main.name
            ]
          ]
          view    = "timeSeries"
          stacked = false
          region  = var.aws_region
          title   = "ECS CPU Utilization"
          period  = 300
        }
      },
      {
        type   = "metric"
        x      = 0
        y      = 6
        width  = 12
        height = 6

        properties = {
          metrics = [
            for service_name in keys(local.mlflow_services) : [
              "AWS/ECS",
              "MemoryUtilization",
              "ServiceName",
              "${local.name_prefix}-${local.mlflow_services[service_name].name}-service",
              "ClusterName",
              aws_ecs_cluster.main.name
            ]
          ]
          view    = "timeSeries"
          stacked = false
          region  = var.aws_region
          title   = "ECS Memory Utilization"
          period  = 300
        }
      },
      {
        type   = "metric"
        x      = 0
        y      = 12
        width  = 12
        height = 6

        properties = {
          metrics = [
            [
              "AWS/ApplicationELB",
              "RequestCount",
              "LoadBalancer",
              aws_lb.main.arn_suffix
            ],
            [
              "AWS/ApplicationELB",
              "TargetResponseTime",
              "LoadBalancer",
              aws_lb.main.arn_suffix
            ]
          ]
          view    = "timeSeries"
          stacked = false
          region  = var.aws_region
          title   = "Load Balancer Metrics"
          period  = 300
        }
      }
    ]
  })

  tags = local.common_tags
}
