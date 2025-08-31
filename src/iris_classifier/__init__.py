"""
Iris Classifier MLOps Package

A comprehensive MLOps package for Iris flower classification using MLflow.
"""

__version__ = "1.0.0"
__author__ = "MLOps Team"
__email__ = "mlops@example.com"

from src.iris_classifier.config import Config
from src.iris_classifier.utils.logging import setup_logging

# Package-level configuration
config = Config()
logger = setup_logging()

__all__ = ["config", "logger", "__version__"]
