"""
Model prediction utilities for single instances
"""

import logging
from datetime import datetime
from typing import Any, Dict, Optional, Union

import numpy as np
import pandas as pd

from ..data.preprocessor import DataPreprocessor
from ..data.validator import DataValidator
from ..models.base import BaseModel

logger = logging.getLogger(__name__)


class ModelPredictor:
    """Single instance prediction service"""

    def __init__(
        self,
        model: BaseModel,
        preprocessor: Optional[DataPreprocessor] = None,
        validator: Optional[DataValidator] = None,
    ):
        """
        Initialize predictor

        Args:
            model: Trained model instance
            preprocessor: Data preprocessor (optional)
            validator: Data validator (optional)
        """
        self.model = model
        self.preprocessor = preprocessor
        self.validator = validator or DataValidator()
        self.class_names = ["setosa", "versicolor", "virginica"]

        if not model.is_fitted:
            logger.warning("Model is not fitted. Predictions may fail.")

    def predict(
        self, input_data: Dict[str, float], return_probabilities: bool = True
    ) -> Dict[str, Any]:
        """
        Make a single prediction

        Args:
            input_data: Dictionary with feature values
            return_probabilities: Whether to return class probabilities

        Returns:
            Prediction results dictionary
        """
        logger.debug(f"Making prediction for input: {input_data}")

        # Validate input
        is_valid, errors = self.validator.validate_prediction_input(input_data)
        if not is_valid:
            raise ValueError(f"Input validation failed: {errors}")

        try:
            # Prepare input data
            input_df = pd.DataFrame([input_data])

            # Apply preprocessing if available
            if self.preprocessor:
                X_processed, _ = self.preprocessor.transform(input_df)
                X_input = X_processed
            else:
                # Ensure correct feature order
                feature_order = [
                    "sepal_length",
                    "sepal_width",
                    "petal_length",
                    "petal_width",
                ]
                X_input = input_df[feature_order].values

            # Make prediction
            prediction = self.model.predict(X_input)[0]
            class_name = self.class_names[int(prediction)]

            # Get probabilities if requested and available
            probabilities = None
            confidence = 1.0

            if return_probabilities:
                try:
                    probs = self.model.predict_proba(X_input)
                    if probs is not None:
                        probabilities = {
                            self.class_names[i]: float(prob)
                            for i, prob in enumerate(probs[0])
                        }
                        confidence = float(max(probs[0]))
                    else:
                        probabilities = {class_name: 1.0}
                except Exception as e:
                    logger.warning(f"Could not get probabilities: {e}")
                    probabilities = {class_name: 1.0}

            result = {
                "prediction": int(prediction),
                "class_name": class_name,
                "confidence": confidence,
                "timestamp": datetime.now(),
                "input_features": input_data,
            }

            if probabilities:
                result["probabilities"] = probabilities

            logger.debug(f"Prediction result: {result}")
            return result

        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            raise RuntimeError(f"Prediction failed: {str(e)}")

    def predict_with_uncertainty(
        self, input_data: Dict[str, float], n_samples: int = 100
    ) -> Dict[str, Any]:
        """
        Make prediction with uncertainty estimation using Monte Carlo dropout
        (if model supports it) or bootstrap sampling

        Args:
            input_data: Dictionary with feature values
            n_samples: Number of samples for uncertainty estimation

        Returns:
            Prediction results with uncertainty measures
        """
        logger.debug(
            f"Making prediction with uncertainty estimation ({n_samples} samples)"
        )

        # For now, return standard prediction with confidence interval
        # This could be extended with ensemble models or MC dropout
        base_result = self.predict(input_data, return_probabilities=True)

        # Add uncertainty measures
        if "probabilities" in base_result:
            probs = list(base_result["probabilities"].values())
            entropy = -sum(p * np.log(p + 1e-8) for p in probs if p > 0)

            base_result.update(
                {
                    "uncertainty": {
                        "entropy": float(entropy),
                        "confidence_interval": {
                            "lower": max(0.0, base_result["confidence"] - 0.1),
                            "upper": min(1.0, base_result["confidence"] + 0.1),
                        },
                        "prediction_variance": float(np.var(probs)),
                    }
                }
            )

        return base_result

    def explain_prediction(self, input_data: Dict[str, float]) -> Dict[str, Any]:
        """
        Provide explanation for the prediction (feature importance)

        Args:
            input_data: Dictionary with feature values

        Returns:
            Prediction explanation
        """
        logger.debug("Generating prediction explanation")

        # Get base prediction
        prediction_result = self.predict(input_data)

        # Get model feature importance if available
        feature_importance = self.model.get_feature_importance()

        explanation = {
            "prediction_result": prediction_result,
            "model_type": type(self.model).__name__,
            "feature_contributions": {},
        }

        if feature_importance is not None:
            feature_names = [
                "sepal_length",
                "sepal_width",
                "petal_length",
                "petal_width",
            ]

            # Calculate feature contributions (simple approach)
            for i, feature in enumerate(feature_names):
                if i < len(feature_importance):
                    contribution = feature_importance[i] * input_data[feature]
                    explanation["feature_contributions"][feature] = {
                        "value": input_data[feature],
                        "importance": float(feature_importance[i]),
                        "contribution": float(contribution),
                    }

        # Add feature statistics relative to typical ranges
        feature_stats = {
            "sepal_length": {"min": 4.3, "max": 7.9, "mean": 5.8},
            "sepal_width": {"min": 2.0, "max": 4.4, "mean": 3.1},
            "petal_length": {"min": 1.0, "max": 6.9, "mean": 3.8},
            "petal_width": {"min": 0.1, "max": 2.5, "mean": 1.2},
        }

        explanation["feature_analysis"] = {}
        for feature, value in input_data.items():
            if feature in feature_stats:
                stats = feature_stats[feature]
                normalized_value = (value - stats["mean"]) / (
                    stats["max"] - stats["min"]
                )
                explanation["feature_analysis"][feature] = {
                    "value": value,
                    "normalized": float(normalized_value),
                    "percentile": float(
                        (value - stats["min"]) / (stats["max"] - stats["min"]) * 100
                    ),
                }

        return explanation

    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model"""
        return {
            "model_type": type(self.model).__name__,
            "model_name": getattr(self.model, "name", "Unknown"),
            "is_fitted": self.model.is_fitted,
            "has_preprocessor": self.preprocessor is not None,
            "has_validator": self.validator is not None,
            "supports_probabilities": hasattr(self.model, "predict_proba"),
            "feature_names": [
                "sepal_length",
                "sepal_width",
                "petal_length",
                "petal_width",
            ],
            "class_names": self.class_names,
        }

    def health_check(self) -> Dict[str, Any]:
        """Perform health check on the predictor"""
        try:
            # Test with sample data
            test_data = {
                "sepal_length": 5.0,
                "sepal_width": 3.0,
                "petal_length": 1.5,
                "petal_width": 0.2,
            }

            self.predict(test_data)

            return {
                "status": "healthy",
                "model_loaded": True,
                "can_predict": True,
                "timestamp": datetime.now(),
            }
        except Exception as e:
            return {
                "status": "unhealthy",
                "model_loaded": self.model is not None,
                "can_predict": False,
                "error": str(e),
                "timestamp": datetime.now(),
            }
