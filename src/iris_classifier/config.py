"""
Configuration management for MLflow Iris Classification project
"""

import os
from typing import Optional

from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


class Config:
    """Configuration class for the application"""

    # MLflow Configuration
    MLFLOW_TRACKING_URI: str = os.getenv("MLFLOW_TRACKING_URI", "file:./mlruns")
    MLFLOW_DEFAULT_ARTIFACT_ROOT: str = os.getenv(
        "MLFLOW_DEFAULT_ARTIFACT_ROOT", "./mlruns"
    )
    MLFLOW_EXPERIMENT_NAME: str = os.getenv(
        "MLFLOW_EXPERIMENT_NAME", "Iris Classification"
    )

    # FastAPI Configuration
    FASTAPI_HOST: str = os.getenv("FASTAPI_HOST", "0.0.0.0")
    FASTAPI_PORT: int = int(os.getenv("FASTAPI_PORT", "8000"))
    FASTAPI_RELOAD: bool = os.getenv("FASTAPI_RELOAD", "False").lower() == "true"

    # Streamlit Configuration
    STREAMLIT_HOST: str = os.getenv("STREAMLIT_HOST", "0.0.0.0")
    STREAMLIT_PORT: int = int(os.getenv("STREAMLIT_PORT", "8501"))
    API_BASE_URL: str = os.getenv("API_BASE_URL", "http://localhost:8000")

    # Logging Configuration
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
    LOG_FORMAT: str = os.getenv(
        "LOG_FORMAT", "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    # Model Configuration
    DEFAULT_MODEL_NAME: str = os.getenv(
        "DEFAULT_MODEL_NAME", "iris-logistic_regression"
    )
    DEFAULT_MODEL_STAGE: str = os.getenv("DEFAULT_MODEL_STAGE", "Production")

    # Training Configuration
    TRAIN_TEST_SPLIT: float = float(os.getenv("TRAIN_TEST_SPLIT", "0.2"))
    RANDOM_STATE: int = int(os.getenv("RANDOM_STATE", "42"))
    CV_FOLDS: int = int(os.getenv("CV_FOLDS", "5"))

    # Database Configuration (optional)
    MLFLOW_BACKEND_STORE_URI: Optional[str] = os.getenv("MLFLOW_BACKEND_STORE_URI")
    MLFLOW_S3_ENDPOINT_URL: Optional[str] = os.getenv("MLFLOW_S3_ENDPOINT_URL")
    AWS_ACCESS_KEY_ID: Optional[str] = os.getenv("AWS_ACCESS_KEY_ID")
    AWS_SECRET_ACCESS_KEY: Optional[str] = os.getenv("AWS_SECRET_ACCESS_KEY")

    @classmethod
    def validate(cls) -> bool:
        """Validate configuration"""
        errors = []

        # Check required configurations
        if not cls.MLFLOW_TRACKING_URI:
            errors.append("MLFLOW_TRACKING_URI is required")

        if cls.FASTAPI_PORT < 1 or cls.FASTAPI_PORT > 65535:
            errors.append("FASTAPI_PORT must be between 1 and 65535")

        if cls.STREAMLIT_PORT < 1 or cls.STREAMLIT_PORT > 65535:
            errors.append("STREAMLIT_PORT must be between 1 and 65535")

        if cls.TRAIN_TEST_SPLIT < 0.1 or cls.TRAIN_TEST_SPLIT > 0.9:
            errors.append("TRAIN_TEST_SPLIT must be between 0.1 and 0.9")

        if cls.CV_FOLDS < 2:
            errors.append("CV_FOLDS must be at least 2")

        if errors:
            for error in errors:
                print(f"Configuration Error: {error}")
            return False

        return True

    @classmethod
    def print_config(cls):
        """Print current configuration"""
        print("=" * 50)
        print("CURRENT CONFIGURATION")
        print("=" * 50)
        print(f"MLflow Tracking URI: {cls.MLFLOW_TRACKING_URI}")
        print(f"MLflow Experiment: {cls.MLFLOW_EXPERIMENT_NAME}")
        print(f"FastAPI Host:Port: {cls.FASTAPI_HOST}:{cls.FASTAPI_PORT}")
        print(f"Streamlit Host:Port: {cls.STREAMLIT_HOST}:{cls.STREAMLIT_PORT}")
        print(f"API Base URL: {cls.API_BASE_URL}")
        print(f"Default Model: {cls.DEFAULT_MODEL_NAME} ({cls.DEFAULT_MODEL_STAGE})")
        print(f"Train/Test Split: {cls.TRAIN_TEST_SPLIT}")
        print(f"Random State: {cls.RANDOM_STATE}")
        print(f"CV Folds: {cls.CV_FOLDS}")
        print("=" * 50)


class DevelopmentConfig(Config):
    """Development environment configuration"""

    FASTAPI_RELOAD = True
    LOG_LEVEL = "DEBUG"


class ProductionConfig(Config):
    """Production environment configuration"""

    FASTAPI_RELOAD = False
    LOG_LEVEL = "INFO"


class TestingConfig(Config):
    """Testing environment configuration"""

    MLFLOW_TRACKING_URI = "file:./test_mlruns"
    MLFLOW_EXPERIMENT_NAME = "Test Iris Classification"
    LOG_LEVEL = "DEBUG"


def get_config(env: str = "development") -> Config:
    """Get configuration based on environment"""
    configs = {
        "development": DevelopmentConfig,
        "production": ProductionConfig,
        "testing": TestingConfig,
    }

    config_class = configs.get(env.lower(), DevelopmentConfig)
    return config_class()


if __name__ == "__main__":
    # Test configuration
    config = Config()
    config.print_config()

    if config.validate():
        print("✅ Configuration is valid!")
    else:
        print("❌ Configuration validation failed!")
