# Multi-stage Dockerfile for MLflow project
FROM python:3.9-slim as base

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

# Training stage
FROM base as training
COPY train.py .
COPY model.py .
COPY mlflow_quickstart_local.py .
CMD ["python", "train.py"]

# FastAPI stage
FROM base as fastapi
COPY fastapi_app.py .
COPY model.py .
COPY mlruns/ ./mlruns/
EXPOSE 8000
CMD ["uvicorn", "fastapi_app:app", "--host", "0.0.0.0", "--port", "8000"]

# Streamlit stage  
FROM base as streamlit
COPY streamlit_app.py .
EXPOSE 8501
CMD ["streamlit", "run", "streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0"]

# MLflow server stage
FROM base as mlflow-server
COPY mlruns/ ./mlruns/
EXPOSE 5000
CMD ["mlflow", "server", "--host", "0.0.0.0", "--port", "5000", "--backend-store-uri", "file:./mlruns"]

# Production stage (FastAPI + MLflow server)
FROM base as production
COPY fastapi_app.py .
COPY model.py .
COPY mlruns/ ./mlruns/
COPY start_mlflow_server.sh .
RUN chmod +x start_mlflow_server.sh
EXPOSE 8000 5000
CMD ["./start_mlflow_server.sh"]
