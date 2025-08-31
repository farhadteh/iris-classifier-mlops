# MLflow Iris Classification - Makefile
# Simple commands to manage the MLOps pipeline

.PHONY: help install train serve streamlit mlflow docker-build docker-up docker-down test clean demo

# Default target
help:
	@echo "MLflow Iris Classification Pipeline"
	@echo "=================================="
	@echo ""
	@echo "Available commands:"
	@echo "  install        Install dependencies"
	@echo "  train          Train models"
	@echo "  serve          Start FastAPI server"
	@echo "  streamlit      Start Streamlit app"
	@echo "  mlflow         Start MLflow server"
	@echo "  all            Start all services"
	@echo "  docker-build   Build Docker images"
	@echo "  docker-up      Start with Docker Compose"
	@echo "  docker-down    Stop Docker services"
	@echo "  test           Run tests"
	@echo "  test-coverage  Run tests with coverage"
	@echo "  lint           Run linting"
	@echo "  format         Format code"
	@echo "  clean          Clean generated files"
	@echo "  demo           Run demo script"
	@echo "  setup          Initial project setup"

# Installation and setup
install:
	pip install -r requirements.txt

setup: install
	mkdir -p logs models data
	cp .env.example .env
	@echo "✅ Project setup completed!"
	@echo "📝 Edit .env file to customize configuration"

# Training
train:
	python train.py

train-fast:
	python train.py --no-tune --models logistic_regression

train-all:
	python train.py --models logistic_regression random_forest svm

# Services
mlflow:
	mlflow server --host 0.0.0.0 --port 5000 --backend-store-uri file:./mlruns

serve:
	uvicorn fastapi_app:app --host 0.0.0.0 --port 8000 --reload

streamlit:
	streamlit run streamlit_app.py --server.port 8501 --server.address 0.0.0.0

# Start all services
all:
	@echo "🚀 Starting all services..."
	./start_services.sh

# Docker commands
docker-build:
	cd deployment/docker && docker-compose build

docker-up:
	cd deployment/docker && docker-compose up -d

docker-down:
	cd deployment/docker && docker-compose down

docker-logs:
	cd deployment/docker && docker-compose logs -f

# Development
test:
	pytest tests/ -v

test-coverage:
	pytest tests/ -v --cov=. --cov-report=html --cov-report=term

lint:
	flake8 . --count --select=E9,F63,F7,F82 --show-source --statistics
	flake8 . --count --exit-zero --max-complexity=10 --max-line-length=127 --statistics

format:
	black .
	isort .

format-check:
	black --check .
	isort --check-only .

# Utilities
demo:
	python demo.py

clean:
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	rm -rf .pytest_cache
	rm -rf htmlcov
	rm -rf .coverage
	rm -f *.png *.csv model_comparison.*

clean-all: clean
	rm -rf mlruns models logs

# Health checks
health:
	@echo "🔍 Checking service health..."
	@curl -s http://localhost:5000 > /dev/null && echo "✅ MLflow server is running" || echo "❌ MLflow server is not running"
	@curl -s http://localhost:8000/health > /dev/null && echo "✅ FastAPI server is running" || echo "❌ FastAPI server is not running"
	@curl -s http://localhost:8501/_stcore/health > /dev/null && echo "✅ Streamlit app is running" || echo "❌ Streamlit app is not running"

# Quick start
quickstart: setup train all
	@echo "🎉 Quick start completed!"
	@echo "🌐 Access the services:"
	@echo "   MLflow UI: http://localhost:5000"
	@echo "   FastAPI: http://localhost:8000"
	@echo "   Streamlit: http://localhost:8501"

# Development workflow
dev: format lint test

# CI/CD simulation
ci: format-check lint test-coverage

# Show URLs
urls:
	@echo "🌐 Service URLs:"
	@echo "   MLflow UI: http://localhost:5000"
	@echo "   FastAPI: http://localhost:8000"
	@echo "   FastAPI Docs: http://localhost:8000/docs"
	@echo "   Streamlit: http://localhost:8501"
