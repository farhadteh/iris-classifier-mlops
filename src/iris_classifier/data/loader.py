"""
Data loading utilities for the Iris dataset
"""

import logging
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from sklearn import datasets

logger = logging.getLogger(__name__)


class DataLoader:
    """Data loader for Iris dataset and external data sources"""

    @staticmethod
    def load_iris_dataset() -> Tuple[np.ndarray, np.ndarray, list, list]:
        """
        Load the built-in Iris dataset from scikit-learn

        Returns:
            Tuple of (features, targets, feature_names, target_names)
        """
        logger.info("Loading Iris dataset from scikit-learn")

        iris = datasets.load_iris()
        X, y = iris.data, iris.target
        feature_names = iris.feature_names
        target_names = iris.target_names.tolist()

        logger.info(f"Loaded dataset with shape: {X.shape}")
        logger.info(f"Features: {feature_names}")
        logger.info(f"Classes: {target_names}")

        return X, y, feature_names, target_names

    @staticmethod
    def load_from_csv(
        file_path: str, target_column: str = "target"
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Load data from CSV file

        Args:
            file_path: Path to the CSV file
            target_column: Name of the target column

        Returns:
            Tuple of (features DataFrame, target Series)
        """
        logger.info(f"Loading data from CSV: {file_path}")

        if not Path(file_path).exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        df = pd.read_csv(file_path)

        if target_column not in df.columns:
            logger.warning(
                f"Target column '{target_column}' not found. Returning full DataFrame and empty Series"
            )
            return df, pd.Series(dtype=object)

        X = df.drop(columns=[target_column])
        y = df[target_column]

        logger.info(f"Loaded {len(df)} samples with {len(X.columns)} features")
        return X, y

    @staticmethod
    def load_prediction_data(file_path: str) -> pd.DataFrame:
        """
        Load data for prediction (without target column)

        Args:
            file_path: Path to the CSV file

        Returns:
            DataFrame with features for prediction
        """
        logger.info(f"Loading prediction data from: {file_path}")

        if not Path(file_path).exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        df = pd.read_csv(file_path)
        logger.info(f"Loaded {len(df)} samples for prediction")

        return df

    @staticmethod
    def save_to_csv(data: pd.DataFrame, file_path: str, index: bool = False) -> None:
        """
        Save DataFrame to CSV file

        Args:
            data: DataFrame to save
            file_path: Output file path
            index: Whether to include index in CSV
        """
        logger.info(f"Saving data to CSV: {file_path}")

        # Create directory if it doesn't exist
        Path(file_path).parent.mkdir(parents=True, exist_ok=True)

        data.to_csv(file_path, index=index)
        logger.info(f"Data saved successfully to {file_path}")

    @staticmethod
    def create_sample_data(
        n_samples: int = 100, save_path: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Create synthetic Iris-like sample data

        Args:
            n_samples: Number of samples to generate
            save_path: Optional path to save the generated data

        Returns:
            DataFrame with synthetic data
        """
        logger.info(f"Creating {n_samples} synthetic samples")

        np.random.seed(42)
        samples = []

        class_specs = {
            "setosa": {
                "sepal_length": (5.0, 0.4),
                "sepal_width": (3.4, 0.4),
                "petal_length": (1.5, 0.2),
                "petal_width": (0.2, 0.1),
            },
            "versicolor": {
                "sepal_length": (6.0, 0.5),
                "sepal_width": (2.8, 0.3),
                "petal_length": (4.3, 0.5),
                "petal_width": (1.3, 0.2),
            },
            "virginica": {
                "sepal_length": (6.5, 0.6),
                "sepal_width": (3.0, 0.3),
                "petal_length": (5.5, 0.6),
                "petal_width": (2.0, 0.3),
            },
        }

        samples_per_class = n_samples // 3

        for class_name, specs in class_specs.items():
            for _ in range(samples_per_class):
                sample = {"class": class_name}

                for feature, (mean, std) in specs.items():
                    value = max(0.1, np.random.normal(mean, std))
                    sample[feature] = round(value, 2)

                samples.append(sample)

        df = pd.DataFrame(samples)

        if save_path:
            DataLoader.save_to_csv(df, save_path)

        logger.info(f"Created synthetic dataset with shape: {df.shape}")
        return df
