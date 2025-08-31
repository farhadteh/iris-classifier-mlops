#!/usr/bin/env python3
"""
Enhanced MLflow Training Pipeline for Iris Classification
"""

import argparse
import json
import logging
import os
from datetime import datetime
from typing import Any, Dict, Tuple

import joblib
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
import seaborn as sns
from mlflow.models import infer_signature
from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import (
    GridSearchCV,
    cross_val_score,
    train_test_split,
)
from sklearn.svm import SVC

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class IrisModelTrainer:
    """Enhanced Iris Model Training Pipeline"""

    def __init__(
        self,
        mlflow_uri: str = "file:./mlruns",
        experiment_name: str = "Iris Classification",
    ):
        """Initialize the trainer"""
        self.mlflow_uri = mlflow_uri
        self.experiment_name = experiment_name
        self.models = {
            "logistic_regression": LogisticRegression,
            "random_forest": RandomForestClassifier,
            "svm": SVC,
        }
        self.best_model = None
        self.best_score = 0

        # Set up MLflow
        mlflow.set_tracking_uri(self.mlflow_uri)
        mlflow.set_experiment(self.experiment_name)

    def load_data(self) -> Tuple[np.ndarray, np.ndarray, list]:
        """Load and prepare the Iris dataset"""
        logger.info("Loading Iris dataset...")

        # Load dataset
        iris = datasets.load_iris()
        X, y = iris.data, iris.target
        feature_names = iris.feature_names

        logger.info(f"Dataset shape: {X.shape}")
        logger.info(f"Features: {feature_names}")
        logger.info(f"Classes: {iris.target_names}")

        return X, y, feature_names

    def prepare_data(
        self,
        X: np.ndarray,
        y: np.ndarray,
        test_size: float = 0.2,
        random_state: int = 42,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Split data into train and test sets"""
        logger.info(f"Splitting data: train_size={1-test_size}, test_size={test_size}")

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )

        logger.info(f"Train set shape: {X_train.shape}")
        logger.info(f"Test set shape: {X_test.shape}")

        return X_train, X_test, y_train, y_test

    def get_model_configs(self) -> Dict[str, Dict[str, Any]]:
        """Get model configurations for hyperparameter tuning"""
        return {
            "logistic_regression": {
                "model_class": LogisticRegression,
                "param_grid": {
                    "solver": ["lbfgs", "liblinear"],
                    "C": [0.1, 1.0, 10.0],
                    "max_iter": [1000, 2000],
                    "random_state": [42],
                },
            },
            "random_forest": {
                "model_class": RandomForestClassifier,
                "param_grid": {
                    "n_estimators": [50, 100, 200],
                    "max_depth": [None, 5, 10],
                    "min_samples_split": [2, 5],
                    "random_state": [42],
                },
            },
            "svm": {
                "model_class": SVC,
                "param_grid": {
                    "C": [0.1, 1.0, 10.0],
                    "kernel": ["linear", "rbf"],
                    "probability": [True],  # For probability predictions
                    "random_state": [42],
                },
            },
        }

    def train_model(
        self,
        model_name: str,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
        feature_names: list,
        tune_hyperparameters: bool = True,
    ) -> Dict[str, Any]:
        """Train a single model with optional hyperparameter tuning"""

        logger.info(f"Training {model_name}...")

        configs = self.get_model_configs()
        if model_name not in configs:
            raise ValueError(f"Unknown model: {model_name}")

        config = configs[model_name]

        with mlflow.start_run(
            run_name=f"{model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        ):
            # Log basic info
            mlflow.set_tag("model_type", model_name)
            mlflow.set_tag("training_date", datetime.now().isoformat())
            mlflow.set_tag("dataset", "iris")

            if tune_hyperparameters:
                logger.info(f"Performing hyperparameter tuning for {model_name}...")

                # Grid search with cross-validation
                grid_search = GridSearchCV(
                    config["model_class"](),
                    config["param_grid"],
                    cv=5,
                    scoring="accuracy",
                    n_jobs=-1,
                )

                grid_search.fit(X_train, y_train)
                model = grid_search.best_estimator_
                best_params = grid_search.best_params_
                cv_score = grid_search.best_score_

                # Log hyperparameters
                mlflow.log_params(best_params)
                mlflow.log_metric("cv_score", cv_score)

                logger.info(f"Best parameters: {best_params}")
                logger.info(f"CV score: {cv_score:.4f}")

            else:
                # Use default parameters
                model = config["model_class"](random_state=42)
                model.fit(X_train, y_train)
                best_params = model.get_params()
                mlflow.log_params(best_params)

            # Make predictions
            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)

            # Calculate metrics
            train_metrics = self.calculate_metrics(y_train, y_train_pred, "train")
            test_metrics = self.calculate_metrics(y_test, y_test_pred, "test")

            # Log metrics
            for metric_name, value in {**train_metrics, **test_metrics}.items():
                mlflow.log_metric(metric_name, value)

            # Cross-validation scores
            cv_scores = cross_val_score(model, X_train, y_train, cv=5)
            mlflow.log_metric("cv_mean", cv_scores.mean())
            mlflow.log_metric("cv_std", cv_scores.std())

            # Generate and log plots
            self.log_plots(model, X_train, y_train, X_test, y_test, feature_names)

            # Log classification report
            report = classification_report(y_test, y_test_pred, output_dict=True)
            mlflow.log_dict(report, "classification_report.json")

            # Model signature
            signature = infer_signature(X_train, model.predict(X_train))

            # Log model
            model_info = mlflow.sklearn.log_model(
                sk_model=model,
                artifact_path="model",
                signature=signature,
                input_example=X_train[:5],
                registered_model_name=f"iris-{model_name}",
            )

            # Save model locally as well
            os.makedirs("models", exist_ok=True)
            joblib.dump(model, f"models/{model_name}_model.pkl")

            # Update best model tracking
            test_accuracy = test_metrics["test_accuracy"]
            if test_accuracy > self.best_score:
                self.best_score = test_accuracy
                self.best_model = {
                    "name": model_name,
                    "model": model,
                    "accuracy": test_accuracy,
                    "run_id": mlflow.active_run().info.run_id,
                }

            results = {
                "model_name": model_name,
                "model": model,
                "train_metrics": train_metrics,
                "test_metrics": test_metrics,
                "cv_scores": cv_scores,
                "run_id": mlflow.active_run().info.run_id,
                "model_uri": model_info.model_uri,
            }

            logger.info(f"Model {model_name} trained successfully!")
            logger.info(f"Test accuracy: {test_accuracy:.4f}")

            return results

    def calculate_metrics(
        self, y_true: np.ndarray, y_pred: np.ndarray, prefix: str
    ) -> Dict[str, float]:
        """Calculate classification metrics"""
        return {
            f"{prefix}_accuracy": accuracy_score(y_true, y_pred),
            f"{prefix}_precision": precision_score(y_true, y_pred, average="weighted"),
            f"{prefix}_recall": recall_score(y_true, y_pred, average="weighted"),
            f"{prefix}_f1": f1_score(y_true, y_pred, average="weighted"),
        }

    def log_plots(
        self,
        model,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
        feature_names: list,
    ):
        """Generate and log visualization plots"""

        # Confusion Matrix
        plt.figure(figsize=(8, 6))
        y_test_pred = model.predict(X_test)
        cm = confusion_matrix(y_test, y_test_pred)
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=["setosa", "versicolor", "virginica"],
            yticklabels=["setosa", "versicolor", "virginica"],
        )
        plt.title("Confusion Matrix")
        plt.ylabel("True Label")
        plt.xlabel("Predicted Label")
        plt.tight_layout()
        mlflow.log_figure(plt.gcf(), "confusion_matrix.png")
        plt.close()

        # Feature Importance (if available)
        if hasattr(model, "feature_importances_"):
            plt.figure(figsize=(10, 6))
            importances = model.feature_importances_
            indices = np.argsort(importances)[::-1]

            plt.bar(range(len(importances)), importances[indices])
            plt.xticks(
                range(len(importances)),
                [feature_names[i] for i in indices],
                rotation=45,
            )
            plt.title("Feature Importances")
            plt.tight_layout()
            mlflow.log_figure(plt.gcf(), "feature_importance.png")
            plt.close()

        # Data distribution
        plt.figure(figsize=(12, 8))
        for i, feature in enumerate(feature_names):
            plt.subplot(2, 2, i + 1)
            for class_idx, class_name in enumerate(
                ["setosa", "versicolor", "virginica"]
            ):
                data = X_train[y_train == class_idx, i]
                plt.hist(data, alpha=0.7, label=class_name, bins=15)
            plt.xlabel(feature)
            plt.ylabel("Frequency")
            plt.legend()
            plt.title(f"Distribution of {feature}")

        plt.tight_layout()
        mlflow.log_figure(plt.gcf(), "feature_distributions.png")
        plt.close()

    def run_experiment(
        self, models_to_train: list = None, tune_hyperparameters: bool = True
    ) -> Dict[str, Any]:
        """Run the complete training experiment"""

        if models_to_train is None:
            models_to_train = list(self.models.keys())

        logger.info(f"Starting experiment: {self.experiment_name}")
        logger.info(f"Models to train: {models_to_train}")

        # Load and prepare data
        X, y, feature_names = self.load_data()
        X_train, X_test, y_train, y_test = self.prepare_data(X, y)

        # Train models
        results = {}
        for model_name in models_to_train:
            try:
                result = self.train_model(
                    model_name,
                    X_train,
                    y_train,
                    X_test,
                    y_test,
                    feature_names,
                    tune_hyperparameters,
                )
                results[model_name] = result
            except Exception as e:
                logger.error(f"Failed to train {model_name}: {e}")
                continue

        # Log comparison metrics
        if len(results) > 1:
            self.log_model_comparison(results)

        # Register best model in production
        if self.best_model:
            self.promote_best_model()

        logger.info("Experiment completed successfully!")
        logger.info(
            f"Best model: {self.best_model['name']} (accuracy: {self.best_model['accuracy']:.4f})"
        )

        return results

    def log_model_comparison(self, results: Dict[str, Any]):
        """Log comparison of different models"""

        comparison_data = []
        for model_name, result in results.items():
            comparison_data.append(
                {
                    "model": model_name,
                    "test_accuracy": result["test_metrics"]["test_accuracy"],
                    "test_f1": result["test_metrics"]["test_f1"],
                    "cv_mean": result["cv_scores"].mean(),
                    "cv_std": result["cv_scores"].std(),
                }
            )

        # Create comparison plot
        df = pd.DataFrame(comparison_data)

        plt.figure(figsize=(12, 6))

        plt.subplot(1, 2, 1)
        plt.bar(df["model"], df["test_accuracy"])
        plt.title("Test Accuracy Comparison")
        plt.ylabel("Accuracy")
        plt.xticks(rotation=45)

        plt.subplot(1, 2, 2)
        plt.errorbar(df["model"], df["cv_mean"], yerr=df["cv_std"], fmt="o", capsize=5)
        plt.title("Cross-Validation Score Comparison")
        plt.ylabel("CV Score")
        plt.xticks(rotation=45)

        plt.tight_layout()

        # Save comparison plot
        plt.savefig("model_comparison.png", dpi=150, bbox_inches="tight")
        plt.close()

        # Save comparison data
        df.to_csv("model_comparison.csv", index=False)

        logger.info(
            "Model comparison saved to model_comparison.png and model_comparison.csv"
        )

    def promote_best_model(self):
        """Promote the best model to production stage"""
        try:
            client = mlflow.tracking.MlflowClient()
            model_name = f"iris-{self.best_model['name']}"

            # Get the latest version
            latest_versions = client.get_latest_versions(model_name, stages=["None"])
            if latest_versions:
                version = latest_versions[0].version

                # Transition to production
                client.transition_model_version_stage(
                    name=model_name, version=version, stage="Production"
                )

                logger.info(
                    f"Model {model_name} version {version} promoted to Production"
                )

        except Exception as e:
            logger.error(f"Failed to promote model to production: {e}")


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Train Iris Classification Models")
    parser.add_argument(
        "--models",
        nargs="+",
        default=None,
        choices=["logistic_regression", "random_forest", "svm"],
        help="Models to train",
    )
    parser.add_argument(
        "--no-tune", action="store_true", help="Skip hyperparameter tuning"
    )
    parser.add_argument(
        "--mlflow-uri", default="file:./mlruns", help="MLflow tracking URI"
    )
    parser.add_argument(
        "--experiment", default="Iris Classification", help="MLflow experiment name"
    )

    args = parser.parse_args()

    # Initialize trainer
    trainer = IrisModelTrainer(
        mlflow_uri=args.mlflow_uri, experiment_name=args.experiment
    )

    # Run experiment
    results = trainer.run_experiment(
        models_to_train=args.models, tune_hyperparameters=not args.no_tune
    )

    # Print summary
    print("\n" + "=" * 50)
    print("TRAINING SUMMARY")
    print("=" * 50)

    for model_name, result in results.items():
        print(f"\n{model_name.upper()}:")
        print(f"  Test Accuracy: {result['test_metrics']['test_accuracy']:.4f}")
        print(f"  Test F1 Score: {result['test_metrics']['test_f1']:.4f}")
        print(
            f"  CV Score: {result['cv_scores'].mean():.4f} ± {result['cv_scores'].std():.4f}"
        )
        print(f"  Run ID: {result['run_id']}")

    if trainer.best_model:
        print(
            f"\nBEST MODEL: {trainer.best_model['name']} (accuracy: {trainer.best_model['accuracy']:.4f})"
        )

    print("\nTraining completed successfully! 🎉")


if __name__ == "__main__":
    main()
