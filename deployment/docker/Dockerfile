# Multi-stage Dockerfile for MLflow project
FROM python:3.9-slim AS base

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    MLFLOW_TRACKING_URI=file:./mlruns

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Set work directory
WORKDIR /app

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY src/ ./src/
COPY setup.py .

# Install the package
RUN pip install -e .

# Create mlruns directory with proper permissions
RUN mkdir -p ./mlruns && chmod 755 ./mlruns

# Training stage
FROM base AS training
CMD ["python", "src/iris_classifier/training/train.py"]

# FastAPI stage
FROM base AS fastapi
EXPOSE 8000
CMD ["uvicorn", "src.iris_classifier.api.fastapi_app:app", "--host", "0.0.0.0", "--port", "8000"]

# Streamlit stage  
FROM base AS streamlit
EXPOSE 8501
CMD ["streamlit", "run", "src/iris_classifier/api/streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0"]

# MLflow server stage
FROM base AS mlflow-server
EXPOSE 5000
CMD ["mlflow", "server", "--host", "0.0.0.0", "--port", "5000", "--backend-store-uri", "file:./mlruns"]

# Production stage (FastAPI + MLflow server)
FROM base AS production
EXPOSE 8000 5000
CMD ["uvicorn", "src.iris_classifier.api.fastapi_app:app", "--host", "0.0.0.0", "--port", "8000"]
