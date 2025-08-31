"""
Data preprocessing utilities
"""

import logging
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

logger = logging.getLogger(__name__)


class DataPreprocessor:
    """Data preprocessing utilities for the Iris dataset"""

    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.feature_columns = [
            "sepal_length",
            "sepal_width",
            "petal_length",
            "petal_width",
        ]
        self.is_fitted = False

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> "DataPreprocessor":
        """
        Fit the preprocessor on training data

        Args:
            X: Feature DataFrame
            y: Target Series (optional)

        Returns:
            Self for method chaining
        """
        logger.info("Fitting data preprocessor")

        # Ensure we have the right columns
        if isinstance(X, pd.DataFrame):
            X_processed = self._ensure_feature_columns(X)
        else:
            X_processed = pd.DataFrame(X, columns=self.feature_columns)

        # Fit scaler
        self.scaler.fit(X_processed)

        # Fit label encoder if targets provided
        if y is not None:
            if isinstance(y, pd.Series):
                y_values = y.values
            else:
                y_values = y
            self.label_encoder.fit(y_values)

        self.is_fitted = True
        logger.info("Data preprocessor fitted successfully")
        return self

    def transform(
        self, X: pd.DataFrame, y: Optional[pd.Series] = None
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Transform data using fitted preprocessor

        Args:
            X: Feature DataFrame
            y: Target Series (optional)

        Returns:
            Tuple of (transformed_features, transformed_targets)
        """
        if not self.is_fitted:
            raise ValueError("Preprocessor must be fitted before transform")

        logger.debug(f"Transforming data with shape: {X.shape}")

        # Process features
        if isinstance(X, pd.DataFrame):
            X_processed = self._ensure_feature_columns(X)
        else:
            X_processed = pd.DataFrame(X, columns=self.feature_columns)

        X_scaled = self.scaler.transform(X_processed)

        # Process targets if provided
        y_encoded = None
        if y is not None:
            if isinstance(y, pd.Series):
                y_values = y.values
            else:
                y_values = y
            y_encoded = self.label_encoder.transform(y_values)

        return X_scaled, y_encoded

    def fit_transform(
        self, X: pd.DataFrame, y: Optional[pd.Series] = None
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Fit preprocessor and transform data in one step

        Args:
            X: Feature DataFrame
            y: Target Series (optional)

        Returns:
            Tuple of (transformed_features, transformed_targets)
        """
        return self.fit(X, y).transform(X, y)

    def inverse_transform_labels(self, y_encoded: np.ndarray) -> np.ndarray:
        """
        Convert encoded labels back to original format

        Args:
            y_encoded: Encoded labels

        Returns:
            Original label format
        """
        if not self.is_fitted or not hasattr(self.label_encoder, "classes_"):
            raise ValueError("Label encoder must be fitted before inverse transform")

        return self.label_encoder.inverse_transform(y_encoded)

    def get_feature_names(self) -> list:
        """Get the expected feature column names"""
        return self.feature_columns.copy()

    def get_class_names(self) -> Optional[np.ndarray]:
        """Get the class names from the fitted label encoder"""
        if hasattr(self.label_encoder, "classes_"):
            return self.label_encoder.classes_
        return None

    def _ensure_feature_columns(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Ensure DataFrame has the required feature columns

        Args:
            X: Input DataFrame

        Returns:
            DataFrame with required columns
        """
        # Check if all required columns are present
        missing_cols = set(self.feature_columns) - set(X.columns)
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")

        # Return only the required columns in the correct order
        return X[self.feature_columns]

    @staticmethod
    def split_data(
        X: np.ndarray,
        y: np.ndarray,
        test_size: float = 0.2,
        random_state: int = 42,
        stratify: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Split data into train and test sets

        Args:
            X: Features
            y: Targets
            test_size: Proportion of test set
            random_state: Random seed
            stratify: Whether to stratify the split

        Returns:
            Tuple of (X_train, X_test, y_train, y_test)
        """
        logger.info(f"Splitting data: test_size={test_size}, stratify={stratify}")

        stratify_param = y if stratify else None

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=test_size,
            random_state=random_state,
            stratify=stratify_param,
        )

        logger.info(f"Train set shape: {X_train.shape}, Test set shape: {X_test.shape}")
        return X_train, X_test, y_train, y_test

    def get_preprocessing_info(self) -> Dict[str, Any]:
        """
        Get information about the fitted preprocessor

        Returns:
            Dictionary with preprocessing information
        """
        info = {
            "is_fitted": self.is_fitted,
            "feature_columns": self.feature_columns,
            "scaler_mean": None,
            "scaler_scale": None,
            "classes": None,
        }

        if self.is_fitted:
            info["scaler_mean"] = (
                self.scaler.mean_.tolist() if hasattr(self.scaler, "mean_") else None
            )
            info["scaler_scale"] = (
                self.scaler.scale_.tolist() if hasattr(self.scaler, "scale_") else None
            )
            info["classes"] = (
                self.get_class_names().tolist()
                if self.get_class_names() is not None
                else None
            )

        return info
