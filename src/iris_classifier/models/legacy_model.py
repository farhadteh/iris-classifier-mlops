"""
Model utilities and helper functions for MLflow Iris Classification project
"""

import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelManager:
    """Utility class for managing MLflow models"""

    def __init__(self, tracking_uri: str = "file:./mlruns"):
        """Initialize ModelManager"""
        self.tracking_uri = tracking_uri
        mlflow.set_tracking_uri(tracking_uri)
        self.client = mlflow.tracking.MlflowClient()

    def list_experiments(self) -> List[Dict[str, Any]]:
        """List all experiments"""
        experiments = mlflow.search_experiments()
        return [
            {
                "experiment_id": exp.experiment_id,
                "name": exp.name,
                "lifecycle_stage": exp.lifecycle_stage,
                "artifact_location": exp.artifact_location,
            }
            for exp in experiments
        ]

    def list_models(self) -> List[Dict[str, Any]]:
        """List all registered models"""
        try:
            models = self.client.search_registered_models()
            return [
                {
                    "name": model.name,
                    "creation_timestamp": model.creation_timestamp,
                    "last_updated_timestamp": model.last_updated_timestamp,
                    "description": model.description,
                }
                for model in models
            ]
        except Exception as e:
            logger.error(f"Error listing models: {e}")
            return []

    def get_model_versions(self, model_name: str) -> List[Dict[str, Any]]:
        """Get all versions of a model"""
        try:
            versions = self.client.get_latest_versions(model_name)
            return [
                {
                    "version": version.version,
                    "stage": version.current_stage,
                    "run_id": version.run_id,
                    "creation_timestamp": version.creation_timestamp,
                    "last_updated_timestamp": version.last_updated_timestamp,
                }
                for version in versions
            ]
        except Exception as e:
            logger.error(f"Error getting model versions for {model_name}: {e}")
            return []

    def load_model_by_name(self, model_name: str, stage: str = "Production") -> Any:
        """Load a model by name and stage"""
        try:
            model_uri = f"models:/{model_name}/{stage}"
            model = mlflow.pyfunc.load_model(model_uri)
            logger.info(f"Loaded model {model_name} from stage {stage}")
            return model
        except Exception as e:
            logger.error(f"Error loading model {model_name} from stage {stage}: {e}")
            return None

    def load_model_by_run_id(self, run_id: str, artifact_path: str = "model") -> Any:
        """Load a model by run ID"""
        try:
            model_uri = f"runs:/{run_id}/{artifact_path}"
            model = mlflow.pyfunc.load_model(model_uri)
            logger.info(f"Loaded model from run {run_id}")
            return model
        except Exception as e:
            logger.error(f"Error loading model from run {run_id}: {e}")
            return None

    def get_latest_model(
        self, experiment_name: str = "Iris Classification"
    ) -> Optional[Any]:
        """Get the latest trained model from an experiment"""
        try:
            experiment = mlflow.get_experiment_by_name(experiment_name)
            if not experiment:
                logger.error(f"Experiment {experiment_name} not found")
                return None

            runs = mlflow.search_runs(
                experiment_ids=[experiment.experiment_id],
                order_by=["start_time DESC"],
                max_results=1,
            )

            if runs.empty:
                logger.error(f"No runs found in experiment {experiment_name}")
                return None

            run_id = runs.iloc[0]["run_id"]
            return self.load_model_by_run_id(run_id)

        except Exception as e:
            logger.error(f"Error getting latest model: {e}")
            return None

    def promote_model(self, model_name: str, version: str, stage: str) -> bool:
        """Promote a model version to a new stage"""
        try:
            self.client.transition_model_version_stage(
                name=model_name, version=version, stage=stage
            )
            logger.info(f"Promoted model {model_name} version {version} to {stage}")
            return True
        except Exception as e:
            logger.error(f"Error promoting model: {e}")
            return False


class IrisPredictor:
    """Specialized predictor for Iris dataset"""

    def __init__(self, model=None):
        """Initialize predictor"""
        self.model = model
        self.class_names = ["setosa", "versicolor", "virginica"]
        self.feature_names = [
            "sepal_length",
            "sepal_width",
            "petal_length",
            "petal_width",
        ]

    def load_model(self, model_source: str):
        """Load model from various sources"""
        try:
            if model_source.startswith("models:"):
                self.model = mlflow.pyfunc.load_model(model_source)
            elif model_source.startswith("runs:"):
                self.model = mlflow.pyfunc.load_model(model_source)
            elif model_source.endswith(".pkl"):
                self.model = joblib.load(model_source)
            else:
                # Try as model name
                manager = ModelManager()
                self.model = manager.load_model_by_name(model_source)

            if self.model:
                logger.info(f"Model loaded successfully from {model_source}")
                return True
            else:
                logger.error(f"Failed to load model from {model_source}")
                return False

        except Exception as e:
            logger.error(f"Error loading model: {e}")
            return False

    def predict_single(
        self,
        sepal_length: float,
        sepal_width: float,
        petal_length: float,
        petal_width: float,
    ) -> Dict[str, Any]:
        """Make a single prediction"""
        if not self.model:
            raise ValueError("Model not loaded")

        # Prepare input
        input_data = pd.DataFrame(
            [[sepal_length, sepal_width, petal_length, petal_width]],
            columns=self.feature_names,
        )

        # Make prediction
        prediction = self.model.predict(input_data)[0]

        # Get probabilities if available
        probabilities = {}
        confidence = 1.0

        try:
            if hasattr(self.model, "predict_proba"):
                probs = self.model.predict_proba(input_data)[0]
            elif hasattr(self.model._model_impl, "predict_proba"):
                probs = self.model._model_impl.predict_proba(input_data)[0]
            else:
                probs = None

            if probs is not None:
                probabilities = {
                    self.class_names[i]: float(prob) for i, prob in enumerate(probs)
                }
                confidence = float(max(probs))
            else:
                probabilities = {self.class_names[int(prediction)]: 1.0}

        except Exception as e:
            logger.warning(f"Could not get probabilities: {e}")
            probabilities = {self.class_names[int(prediction)]: 1.0}

        return {
            "prediction": int(prediction),
            "class_name": self.class_names[int(prediction)],
            "confidence": confidence,
            "probabilities": probabilities,
            "input_features": {
                "sepal_length": sepal_length,
                "sepal_width": sepal_width,
                "petal_length": petal_length,
                "petal_width": petal_width,
            },
        }

    def predict_batch(self, data: List[Dict[str, float]]) -> List[Dict[str, Any]]:
        """Make batch predictions"""
        if not self.model:
            raise ValueError("Model not loaded")

        # Prepare input DataFrame
        df = pd.DataFrame(data)

        # Ensure correct column order
        df = df[self.feature_names]

        # Make predictions
        predictions = self.model.predict(df)

        # Get probabilities if available
        try:
            if hasattr(self.model, "predict_proba"):
                probabilities = self.model.predict_proba(df)
            elif hasattr(self.model._model_impl, "predict_proba"):
                probabilities = self.model._model_impl.predict_proba(df)
            else:
                probabilities = None
        except:
            probabilities = None

        # Format results
        results = []
        for i, prediction in enumerate(predictions):
            result = {
                "prediction": int(prediction),
                "class_name": self.class_names[int(prediction)],
                "input_features": data[i],
            }

            if probabilities is not None:
                probs = probabilities[i]
                result["probabilities"] = {
                    self.class_names[j]: float(prob) for j, prob in enumerate(probs)
                }
                result["confidence"] = float(max(probs))
            else:
                result["probabilities"] = {self.class_names[int(prediction)]: 1.0}
                result["confidence"] = 1.0

            results.append(result)

        return results

    def evaluate_model(
        self, X_test: Optional[np.ndarray] = None, y_test: Optional[np.ndarray] = None
    ) -> Dict[str, float]:
        """Evaluate model performance"""
        if not self.model:
            raise ValueError("Model not loaded")

        # Use Iris test data if not provided
        if X_test is None or y_test is None:
            iris = datasets.load_iris()
            X, y = iris.data, iris.target
            from sklearn.model_selection import train_test_split

            _, X_test, _, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )

        # Make predictions
        if isinstance(X_test, np.ndarray):
            X_test_df = pd.DataFrame(X_test, columns=self.feature_names)
        else:
            X_test_df = X_test

        y_pred = self.model.predict(X_test_df)

        # Calculate metrics
        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred, average="weighted"),
            "recall": recall_score(y_test, y_pred, average="weighted"),
            "f1_score": f1_score(y_test, y_pred, average="weighted"),
        }

        return metrics


class DataProcessor:
    """Utility class for data processing"""

    @staticmethod
    def load_iris_data() -> Tuple[np.ndarray, np.ndarray, List[str], List[str]]:
        """Load the Iris dataset"""
        iris = datasets.load_iris()
        return iris.data, iris.target, iris.feature_names, iris.target_names

    @staticmethod
    def validate_input(data: Dict[str, float]) -> bool:
        """Validate input data format"""
        required_fields = ["sepal_length", "sepal_width", "petal_length", "petal_width"]

        # Check if all required fields are present
        if not all(field in data for field in required_fields):
            return False

        # Check if all values are numeric and positive
        try:
            for field in required_fields:
                value = float(data[field])
                if value < 0:
                    return False
        except (ValueError, TypeError):
            return False

        return True

    @staticmethod
    def prepare_features(data: Dict[str, float]) -> np.ndarray:
        """Prepare features for prediction"""
        feature_names = ["sepal_length", "sepal_width", "petal_length", "petal_width"]
        return np.array([[data[name] for name in feature_names]])


def create_sample_data(n_samples: int = 10) -> List[Dict[str, float]]:
    """Create sample data for testing"""
    np.random.seed(42)

    samples = []
    for _ in range(n_samples):
        # Generate realistic Iris measurements
        sample = {
            "sepal_length": np.random.uniform(4.3, 7.9),
            "sepal_width": np.random.uniform(2.0, 4.4),
            "petal_length": np.random.uniform(1.0, 6.9),
            "petal_width": np.random.uniform(0.1, 2.5),
        }
        samples.append(sample)

    return samples


def setup_mlflow_tracking(tracking_uri: str = "file:./mlruns"):
    """Set up MLflow tracking"""
    mlflow.set_tracking_uri(tracking_uri)
    logger.info(f"MLflow tracking URI set to: {tracking_uri}")

    # Create mlruns directory if it doesn't exist
    if tracking_uri.startswith("file:"):
        mlruns_path = tracking_uri.replace("file:", "")
        os.makedirs(mlruns_path, exist_ok=True)
        logger.info(f"Created mlruns directory: {mlruns_path}")


def get_model_info(model_name: str) -> Dict[str, Any]:
    """Get detailed information about a registered model"""
    try:
        client = mlflow.tracking.MlflowClient()
        model = client.get_registered_model(model_name)
        versions = client.get_latest_versions(model_name)

        return {
            "name": model.name,
            "creation_timestamp": model.creation_timestamp,
            "last_updated_timestamp": model.last_updated_timestamp,
            "description": model.description,
            "versions": [
                {
                    "version": v.version,
                    "stage": v.current_stage,
                    "run_id": v.run_id,
                    "creation_timestamp": v.creation_timestamp,
                }
                for v in versions
            ],
        }
    except Exception as e:
        logger.error(f"Error getting model info: {e}")
        return {}


# Example usage and testing functions
if __name__ == "__main__":
    # Example usage
    logger.info("Testing model utilities...")

    # Test data processing
    processor = DataProcessor()
    X, y, feature_names, class_names = processor.load_iris_data()
    logger.info(f"Loaded Iris data: {X.shape} samples, {len(feature_names)} features")

    # Test sample data creation
    samples = create_sample_data(5)
    logger.info(f"Created {len(samples)} sample data points")

    # Test model manager
    manager = ModelManager()
    experiments = manager.list_experiments()
    logger.info(f"Found {len(experiments)} experiments")

    models = manager.list_models()
    logger.info(f"Found {len(models)} registered models")

    logger.info("Model utilities test completed!")
