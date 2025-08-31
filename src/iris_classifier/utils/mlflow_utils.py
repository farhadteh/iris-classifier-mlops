"""
MLflow utilities for experiment tracking and model management
"""

import logging
from typing import Any, Dict, List, Optional

import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient

logger = logging.getLogger(__name__)


class MLflowManager:
    """Utility class for managing MLflow operations"""

    def __init__(
        self,
        tracking_uri: str = "file:./mlruns",
        experiment_name: str = "iris_classification",
    ):
        """
        Initialize MLflow manager

        Args:
            tracking_uri: MLflow tracking URI
            experiment_name: Name of the MLflow experiment
        """
        self.tracking_uri = tracking_uri
        self.experiment_name = experiment_name

        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(experiment_name)

        self.client = MlflowClient()
        logger.info(f"MLflow tracking URI set to: {tracking_uri}")
        logger.info(f"Experiment set to: {experiment_name}")

    def get_experiment(self) -> Optional[mlflow.entities.Experiment]:
        """Get the current experiment"""
        return mlflow.get_experiment_by_name(self.experiment_name)

    def list_registered_models(self) -> List[Dict[str, Any]]:
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
            logger.error(f"Error listing registered models: {e}")
            return []

    def get_latest_model_version(
        self, model_name: str, stage: str = "Production"
    ) -> Optional[str]:
        """Get the latest version of a model in a specific stage"""
        try:
            versions = self.client.get_latest_versions(model_name, stages=[stage])
            return versions[0].version if versions else None
        except Exception as e:
            logger.error(f"Error getting latest model version: {e}")
            return None

    def load_model(self, model_name: str, stage: str = "Production"):
        """Load a model from the registry"""
        try:
            model_uri = f"models:/{model_name}/{stage}"
            model = mlflow.pyfunc.load_model(model_uri)
            logger.info(f"Model {model_name} loaded from stage {stage}")
            return model
        except Exception as e:
            logger.error(f"Error loading model {model_name}: {e}")
            return None

    def promote_model(self, model_name: str, version: str, stage: str) -> bool:
        """Promote a model version to a new stage"""
        try:
            self.client.transition_model_version_stage(
                name=model_name, version=version, stage=stage
            )
            logger.info(f"Model {model_name} version {version} promoted to {stage}")
            return True
        except Exception as e:
            logger.error(f"Error promoting model: {e}")
            return False

    def log_model_artifacts(
        self, model, artifact_path: str, signature=None, input_example=None
    ):
        """Log model artifacts to MLflow"""
        return mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path=artifact_path,
            signature=signature,
            input_example=input_example,
        )

    def search_runs(
        self,
        filter_string: str = "",
        order_by: List[str] = None,
        max_results: int = 1000,
    ):
        """Search for runs in the current experiment"""
        experiment = self.get_experiment()
        if not experiment:
            logger.error(f"Experiment {self.experiment_name} not found")
            return []

        return mlflow.search_runs(
            experiment_ids=[experiment.experiment_id],
            filter_string=filter_string,
            order_by=order_by or ["start_time DESC"],
            max_results=max_results,
        )
