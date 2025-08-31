# MLflow Iris Classification - Complete MLOps Pipeline

A comprehensive, production-ready MLOps pipeline for Iris flower classification featuring standard industry structure, modular design, and enterprise-grade practices.

## 🎯 Overview

This project demonstrates a complete machine learning operations (MLOps) pipeline with professional software engineering practices:

- **🏗️ Standard MLOps Architecture**: Industry-standard folder structure and modular design
- **📊 Model Training**: Multi-algorithm training pipeline with automated hyperparameter tuning
- **🚀 Model Serving**: Enterprise-grade FastAPI service with comprehensive validation
- **🎨 Web Interface**: Interactive Streamlit application for model interaction and monitoring
- **🐳 Containerization**: Multi-stage Docker setup with production optimizations
- **🔄 CI/CD Pipeline**: Automated testing, training, and deployment workflows
- **📈 Monitoring**: Real-time model performance tracking and alerting
- **🧪 Testing**: Comprehensive test suite with coverage reporting
- **📚 Documentation**: Detailed API docs and architecture documentation

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        MLOps Pipeline Architecture              │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐         │
│  │ Streamlit   │────│   FastAPI   │────│   MLflow    │         │
│  │ Frontend    │    │ Model API   │    │ Tracking    │         │
│  │   :8501     │    │   :8000     │    │   :5000     │         │
│  └─────────────┘    └─────────────┘    └─────────────┘         │
│         │                    │                    │             │
│         └────────────────────┼────────────────────┘             │
│                              │                                  │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐         │
│  │   Training  │    │   Data      │    │ Inference   │         │
│  │   Pipeline  │    │ Processing  │    │   Service   │         │
│  └─────────────┘    └─────────────┘    └─────────────┘         │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

## 🚀 Quick Start

### Prerequisites

- Python 3.9+
- Docker (for containerized deployment)
- Git

### Option 1: Native Setup

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd MLFLOW
   ```

2. **Setup project**
   ```bash
   make setup  # Creates virtual environment and installs dependencies
   ```
   
   Or manually:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. **Train initial models**
   ```bash
   make train  # Train models with default settings
   # or: python src/iris_classifier/training/train.py
   ```

4. **Start all services**
   ```bash
   make all  # Start all services
   # or: ./scripts/deployment/start_services.sh
   ```

### Option 2: Docker Setup

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd MLFLOW
   ```

2. **Start with Docker Compose**
   ```bash
   make docker-up
   # or: docker-compose -f deployment/docker/docker-compose.yml up -d
   ```

## 📦 Components

### 1. Training Pipeline (`train.py`)

Enhanced training pipeline that:
- Supports multiple algorithms (Logistic Regression, Random Forest, SVM)
- Performs hyperparameter tuning with GridSearchCV
- Logs comprehensive metrics and artifacts to MLflow
- Generates visualization plots
- Automatically promotes the best model to production

```bash
# Train all models with hyperparameter tuning
python src/iris_classifier/training/train.py

# Train specific models
python src/iris_classifier/training/train.py --models logistic_regression random_forest

# Skip hyperparameter tuning for faster training
python src/iris_classifier/training/train.py --no-tune

# Using Makefile shortcuts
make train           # Train all models
make train-fast      # Quick training without tuning
make train-all       # Train all algorithms
```

### 2. FastAPI Model Server (`fastapi_app.py`)

RESTful API for model serving:
- **Single predictions**: `POST /predict`
- **Batch predictions**: `POST /predict/batch`
- **Model information**: `GET /model/info`
- **Health checks**: `GET /health`
- **Model reloading**: `POST /model/reload`

```bash
# Start FastAPI server
uvicorn src.iris_classifier.api.fastapi_app:app --host 0.0.0.0 --port 8000

# Using Makefile
make serve

# API Documentation available at: http://localhost:8000/docs
```

### 3. Streamlit Web Interface (`streamlit_app.py`)

Interactive web application featuring:
- Single flower classification
- Batch prediction with CSV upload
- Model performance exploration
- Real-time API monitoring
- 3D visualizations with Plotly

```bash
# Start Streamlit app
streamlit run src/iris_classifier/api/streamlit_app.py --server.port 8501

# Using Makefile
make streamlit
```

### 4. MLflow Tracking Server

Experiment tracking and model registry:
- Track training experiments
- Compare model performance
- Manage model versions
- Model deployment workflows

```bash
# Start MLflow server
mlflow server --host 0.0.0.0 --port 5000
```

## 🐳 Docker Deployment

### Multi-stage Dockerfile

The project includes a multi-stage Dockerfile supporting different deployment scenarios:

```bash
# Build specific service
docker build --target fastapi -t iris-fastapi .
docker build --target streamlit -t iris-streamlit .
docker build --target mlflow-server -t iris-mlflow .
```

### Docker Compose

Complete stack deployment:

```bash
# Start all services
docker-compose up -d

# Start specific services
docker-compose up -d mlflow-server fastapi-app

# Run training
docker-compose --profile training run training

# Development environment with Jupyter
docker-compose --profile dev up development
```

## 🔄 CI/CD Pipeline

GitHub Actions workflows for:

### Main CI/CD Pipeline (`.github/workflows/ci-cd.yml`)

- **Code Quality**: Black formatting, isort, flake8 linting
- **Testing**: Pytest with coverage reporting
- **Model Training**: Automated retraining on schedule or trigger
- **Docker Builds**: Multi-stage image building and pushing
- **Security Scanning**: Trivy vulnerability scanning
- **Deployment**: Staging and production deployments
- **Monitoring**: Post-deployment health checks

### Manual Deployment (`.github/workflows/manual-deploy.yml`)

- On-demand deployment to staging/production
- Model version specification
- Configurable test skipping
- Environment-specific configuration

### Triggering Workflows

```bash
# Automatic triggers
git push origin main              # Full CI/CD pipeline
git commit -m "feat: [retrain]"   # Trigger model retraining

# Manual triggers
# Use GitHub Actions UI to trigger manual deployment
```

## 📊 Model Performance

The pipeline trains three different models:

| Algorithm | Typical Accuracy | Use Case |
|-----------|------------------|----------|
| Logistic Regression | ~95% | Fast, interpretable baseline |
| Random Forest | ~97% | Robust, feature importance |
| SVM | ~96% | High-dimensional data |

### Metrics Tracked

- Accuracy, Precision, Recall, F1-Score
- Cross-validation scores
- Feature importance (where applicable)
- Confusion matrices
- Training/validation curves

## 🛠️ Configuration

### Environment Variables

Copy `.env.example` to `.env` and customize:

```bash
# MLflow Configuration
MLFLOW_TRACKING_URI=file:./mlruns
MLFLOW_EXPERIMENT_NAME=Iris Classification

# API Configuration
FASTAPI_HOST=0.0.0.0
FASTAPI_PORT=8000
STREAMLIT_PORT=8501

# Model Configuration
DEFAULT_MODEL_NAME=iris-logistic_regression
DEFAULT_MODEL_STAGE=Production
```

### Service Management

```bash
# Start all services
make all
# or: ./scripts/deployment/start_services.sh

# Start specific service
./scripts/deployment/start_services.sh -s fastapi

# Use Docker
make docker-up
# or: ./scripts/deployment/start_services.sh -d

# Stop all services
make docker-down
# or: ./stop_services.sh
```

## 🧪 Testing

Comprehensive test suite:

```bash
# Run all tests
pytest src/tests/ -v

# Run with coverage
pytest src/tests/ -v --cov=src --cov-report=html

# Run specific test file
pytest src/tests/test_api.py -v

# Using Makefile shortcuts
make test              # Run all tests
make test-coverage     # Run with coverage report
make lint              # Run code quality checks
make format            # Format code with black and isort
```

### Test Coverage

- API endpoint testing
- Model utility function testing
- Data validation testing
- Integration testing

## 📁 Project Structure

Following industry-standard MLOps project organization:

```
MLFLOW/
├── 📁 src/                          # Source code package
│   ├── 📁 iris_classifier/          # Main package
│   │   ├── 📁 models/               # Model definitions
│   │   │   ├── base.py              # Abstract base model class
│   │   │   ├── sklearn_models.py    # Scikit-learn implementations
│   │   │   ├── ensemble.py          # Ensemble model combinations
│   │   │   └── legacy_model.py      # Legacy model utilities
│   │   ├── 📁 data/                 # Data processing modules
│   │   │   ├── loader.py            # Data loading utilities
│   │   │   ├── preprocessor.py      # Data preprocessing
│   │   │   └── validator.py         # Data validation
│   │   ├── 📁 features/             # Feature engineering
│   │   │   ├── engineering.py       # Feature creation
│   │   │   └── selection.py         # Feature selection
│   │   ├── 📁 training/             # Training pipeline
│   │   │   ├── trainer.py           # Model training logic
│   │   │   ├── pipeline.py          # Training orchestration
│   │   │   ├── validator.py         # Model validation
│   │   │   └── train.py             # Training script
│   │   ├── 📁 inference/            # Model serving
│   │   │   ├── predictor.py         # Prediction logic
│   │   │   └── batch_predictor.py   # Batch processing
│   │   ├── 📁 api/                  # API components
│   │   │   ├── fastapi_app.py       # FastAPI application
│   │   │   ├── streamlit_app.py     # Streamlit interface
│   │   │   ├── schemas.py           # API schemas
│   │   │   └── middleware.py        # API middleware
│   │   ├── 📁 utils/                # Utility functions
│   │   │   ├── logging.py           # Logging configuration
│   │   │   ├── mlflow_utils.py      # MLflow helpers
│   │   │   └── metrics.py           # Metrics calculation
│   │   └── config.py                # Configuration management
│   └── 📁 tests/                    # Test suite
│       ├── test_api.py              # API tests
│       ├── test_model.py            # Model tests
│       └── test_data.py             # Data tests
├── 📁 configs/                      # Configuration files
│   ├── 📁 environments/             # Environment-specific configs
│   ├── 📁 logging/                  # Logging configurations
│   └── 📁 model/                    # Model configurations
├── 📁 data/                         # Data storage
│   ├── 📁 raw/                      # Raw, immutable data
│   ├── 📁 processed/                # Cleaned, processed data
│   └── 📁 external/                 # External data sources
├── 📁 models/                       # Model storage
│   ├── 📁 trained/                  # Trained model artifacts
│   ├── 📁 staging/                  # Staging models
│   └── 📁 production/               # Production models
├── 📁 notebooks/                    # Jupyter notebooks
├── 📁 scripts/                      # Utility scripts
│   ├── 📁 deployment/               # Deployment scripts
│   │   └── start_services.sh        # Service startup
│   ├── 📁 data_processing/          # Data processing scripts
│   ├── 📁 training/                 # Training scripts
│   └── demo.py                      # Demo script
├── 📁 deployment/                   # Deployment configurations
│   ├── 📁 docker/                   # Docker files
│   │   ├── Dockerfile               # Multi-stage build
│   │   ├── docker-compose.yml       # Service orchestration
│   │   └── .dockerignore            # Docker ignore rules
│   ├── 📁 kubernetes/               # K8s manifests
│   ├── 📁 terraform/                # Infrastructure as code
│   └── 📁 ci-cd/                    # CI/CD configurations
│       └── github-actions/          # GitHub Actions workflows
├── 📁 docs/                         # Documentation
│   ├── 📁 api/                      # API documentation
│   ├── 📁 architecture/             # Architecture docs
│   └── 📁 user_guide/               # User guides
├── 📁 monitoring/                   # Monitoring setup
│   ├── 📁 metrics/                  # Metrics collection
│   ├── 📁 logs/                     # Log aggregation
│   └── 📁 alerts/                   # Alerting rules
├── 📁 mlruns/                       # MLflow tracking data
├── 📄 requirements.txt              # Python dependencies
├── 📄 Makefile                      # Build automation
├── 📄 .env.example                  # Environment template
└── 📄 README.md                     # This documentation
```

## 🧩 Module Documentation

### Core Package (`src/iris_classifier/`)

#### 🤖 Models (`src/iris_classifier/models/`)

- **`base.py`**: Abstract base class defining the interface for all ML models
  - Standardizes model lifecycle methods (fit, predict, save, load)
  - Provides common functionality like parameter management and model info
  - Ensures consistent behavior across all model implementations

- **`sklearn_models.py`**: Concrete implementations of scikit-learn models
  - `LogisticRegressionModel`: Fast, interpretable baseline model
  - `RandomForestModel`: Robust ensemble method with feature importance
  - `SVMModel`: Support Vector Machine with kernel tricks
  - `HyperparameterTunedModel`: Automated hyperparameter optimization wrapper

- **`ensemble.py`**: Advanced ensemble modeling capabilities
  - Combines multiple models using voting strategies
  - Supports both hard and soft voting mechanisms
  - Weighted ensemble combinations for optimal performance

- **`legacy_model.py`**: Legacy model utilities and backward compatibility

#### 📊 Data (`src/iris_classifier/data/`)

- **`loader.py`**: Data loading and ingestion utilities
  - Built-in Iris dataset loading
  - CSV file handling with validation
  - Synthetic data generation for testing
  - Batch and streaming data support

- **`preprocessor.py`**: Data preprocessing and transformation
  - Feature scaling and normalization
  - Label encoding for categorical targets
  - Train/test splitting with stratification
  - Pipeline-friendly transformations

- **`validator.py`**: Comprehensive data validation
  - Schema validation for input data
  - Range checking and anomaly detection
  - Batch validation with detailed error reporting
  - Data quality assessment and reporting

#### 🔧 Features (`src/iris_classifier/features/`)

- **`engineering.py`**: Feature engineering and creation
  - Automated feature generation
  - Domain-specific transformations
  - Feature interaction detection

- **`selection.py`**: Feature selection and dimensionality reduction
  - Statistical feature selection
  - Model-based feature importance
  - Recursive feature elimination

#### 🏋️ Training (`src/iris_classifier/training/`)

- **`trainer.py`**: Core model training logic
  - Multi-algorithm training support
  - Automated hyperparameter tuning
  - Cross-validation and model selection

- **`pipeline.py`**: Training pipeline orchestration
  - End-to-end training workflows
  - Experiment tracking and logging
  - Artifact management

- **`validator.py`**: Model validation and testing
  - Performance metric calculation
  - Model comparison and selection
  - Statistical significance testing

- **`train.py`**: Command-line training interface
  - Configurable training parameters
  - MLflow integration
  - Automated model registration

#### 🔮 Inference (`src/iris_classifier/inference/`)

- **`predictor.py`**: Single prediction interface
  - Real-time prediction serving
  - Input validation and preprocessing
  - Confidence scoring and uncertainty quantification

- **`batch_predictor.py`**: Batch prediction processing
  - High-throughput batch inference
  - Parallel processing capabilities
  - Progress tracking and error handling

#### 🌐 API (`src/iris_classifier/api/`)

- **`fastapi_app.py`**: Production-ready REST API
  - RESTful endpoints with OpenAPI documentation
  - Automatic request/response validation
  - Error handling and logging
  - Health checks and monitoring

- **`streamlit_app.py`**: Interactive web interface
  - Real-time model interaction
  - Visualization and analytics dashboard
  - Batch processing interface
  - Model monitoring and diagnostics

- **`schemas.py`**: API data models and validation schemas
- **`middleware.py`**: Custom middleware for security, logging, and monitoring

#### 🛠️ Utils (`src/iris_classifier/utils/`)

- **`logging.py`**: Centralized logging configuration
  - Structured logging with JSON output
  - Log level management
  - File and console handlers

- **`mlflow_utils.py`**: MLflow integration utilities
  - Experiment management
  - Model registry operations
  - Artifact tracking and retrieval

- **`metrics.py`**: Model performance metrics
  - Classification metrics calculation
  - Statistical analysis
  - Performance monitoring

#### ⚙️ Configuration (`src/iris_classifier/config.py`)

- Environment-specific configuration management
- Parameter validation and defaults
- Configuration inheritance and overrides
- Integration with environment variables

### Infrastructure & Operations

#### 🐳 Deployment (`deployment/`)

- **Docker**: Multi-stage containerization
  - Development, staging, and production images
  - Optimized layer caching
  - Security scanning integration

- **Kubernetes**: Container orchestration
  - Helm charts for deployment
  - Horizontal pod autoscaling
  - Service mesh integration

- **CI/CD**: Automated workflows
  - GitHub Actions pipelines
  - Automated testing and validation
  - Progressive deployment strategies

#### 📈 Monitoring (`monitoring/`)

- **Metrics**: Performance tracking
  - Model accuracy drift detection
  - Latency and throughput monitoring
  - Business metrics correlation

- **Logging**: Centralized log management
  - Structured logging aggregation
  - Error tracking and alerting
  - Audit trail maintenance

- **Alerts**: Proactive notification system
  - Threshold-based alerting
  - Anomaly detection
  - Escalation policies

### Testing & Quality Assurance

#### 🧪 Tests (`src/tests/`)

- **Unit Tests**: Component-level testing
  - Model behavior validation
  - Data processing verification
  - API endpoint testing

- **Integration Tests**: End-to-end validation
  - Workflow testing
  - Service interaction validation
  - Performance benchmarking

- **Load Tests**: Scalability validation
  - Stress testing under load
  - Performance baseline establishment
  - Capacity planning

## 🔗 Service URLs

When running locally:

- **MLflow UI**: http://localhost:5000
- **FastAPI Server**: http://localhost:8000
- **FastAPI Docs**: http://localhost:8000/docs
- **Streamlit App**: http://localhost:8501

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/new-feature`
3. Make your changes and add tests
4. Run the test suite: `pytest tests/`
5. Commit your changes: `git commit -m "feat: add new feature"`
6. Push to the branch: `git push origin feature/new-feature`
7. Submit a pull request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🆘 Troubleshooting

### Common Issues

1. **Port conflicts**: Check if ports 5000, 8000, 8501 are available
2. **Docker issues**: Ensure Docker daemon is running
3. **Model loading errors**: Verify MLflow tracking URI and experiment data
4. **Import errors**: Check virtual environment activation and dependencies

### Getting Help

- Check the [Issues](https://github.com/your-repo/issues) page
- Review the logs: `docker-compose logs -f [service-name]`
- Run health checks: `curl http://localhost:8000/health`

## 🔮 Future Enhancements

- [ ] Kubernetes deployment manifests
- [ ] Model A/B testing framework
- [ ] Advanced monitoring with Prometheus/Grafana
- [ ] Data drift detection
- [ ] Model interpretability dashboard
- [ ] Multi-cloud deployment support
- [ ] Advanced feature engineering pipeline

---

Built with ❤️ using MLflow, FastAPI, Streamlit, and Docker.
