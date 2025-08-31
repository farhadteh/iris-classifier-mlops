"""
Base model class for Iris classification
"""

from abc import ABC, abstractmethod
import numpy as np
from typing import Any, Dict, Optional
import logging

logger = logging.getLogger(__name__)


class BaseModel(ABC):
    """Abstract base class for all Iris classification models"""
    
    def __init__(self, name: str, **kwargs):
        """
        Initialize base model
        
        Args:
            name: Model name
            **kwargs: Additional model parameters
        """
        self.name = name
        self.model = None
        self.is_fitted = False
        self.parameters = kwargs
        self.feature_names = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
        self.class_names = ['setosa', 'versicolor', 'virginica']
        
        logger.info(f"Initialized {self.name} model")
    
    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'BaseModel':
        """
        Train the model
        
        Args:
            X: Training features
            y: Training targets
            
        Returns:
            Self for method chaining
        """
        pass
    
    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions
        
        Args:
            X: Input features
            
        Returns:
            Predicted class labels
        """
        pass
    
    def predict_proba(self, X: np.ndarray) -> Optional[np.ndarray]:
        """
        Predict class probabilities (if supported)
        
        Args:
            X: Input features
            
        Returns:
            Class probabilities or None if not supported
        """
        if hasattr(self.model, 'predict_proba'):
            return self.model.predict_proba(X)
        return None
    
    def get_feature_importance(self) -> Optional[np.ndarray]:
        """
        Get feature importance (if supported)
        
        Returns:
            Feature importance array or None if not supported
        """
        if hasattr(self.model, 'feature_importances_'):
            return self.model.feature_importances_
        elif hasattr(self.model, 'coef_'):
            # For linear models, use absolute coefficients
            return np.abs(self.model.coef_[0]) if len(self.model.coef_.shape) > 1 else np.abs(self.model.coef_)
        return None
    
    def get_parameters(self) -> Dict[str, Any]:
        """
        Get model parameters
        
        Returns:
            Dictionary of model parameters
        """
        if self.model and hasattr(self.model, 'get_params'):
            return self.model.get_params()
        return self.parameters
    
    def set_parameters(self, **params) -> 'BaseModel':
        """
        Set model parameters
        
        Args:
            **params: Parameters to set
            
        Returns:
            Self for method chaining
        """
        self.parameters.update(params)
        if self.model and hasattr(self.model, 'set_params'):
            self.model.set_params(**params)
        
        logger.info(f"Updated parameters for {self.name}: {params}")
        return self
    
    def save_model(self, file_path: str) -> None:
        """
        Save the model to file
        
        Args:
            file_path: Path to save the model
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before saving")
        
        import joblib
        from pathlib import Path
        
        # Create directory if it doesn't exist
        Path(file_path).parent.mkdir(parents=True, exist_ok=True)
        
        # Save model and metadata
        model_data = {
            'model': self.model,
            'name': self.name,
            'parameters': self.parameters,
            'feature_names': self.feature_names,
            'class_names': self.class_names,
            'is_fitted': self.is_fitted
        }
        
        joblib.dump(model_data, file_path)
        logger.info(f"Model saved to {file_path}")
    
    @classmethod
    def load_model(cls, file_path: str) -> 'BaseModel':
        """
        Load a model from file
        
        Args:
            file_path: Path to the saved model
            
        Returns:
            Loaded model instance
        """
        import joblib
        from pathlib import Path
        
        if not Path(file_path).exists():
            raise FileNotFoundError(f"Model file not found: {file_path}")
        
        model_data = joblib.load(file_path)
        
        # Create new instance
        instance = cls(name=model_data['name'], **model_data['parameters'])
        instance.model = model_data['model']
        instance.is_fitted = model_data['is_fitted']
        instance.feature_names = model_data.get('feature_names', instance.feature_names)
        instance.class_names = model_data.get('class_names', instance.class_names)
        
        logger.info(f"Model loaded from {file_path}")
        return instance
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get comprehensive model information
        
        Returns:
            Dictionary with model information
        """
        info = {
            'name': self.name,
            'is_fitted': self.is_fitted,
            'parameters': self.get_parameters(),
            'feature_names': self.feature_names,
            'class_names': self.class_names,
            'model_type': type(self).__name__
        }
        
        # Add feature importance if available
        importance = self.get_feature_importance()
        if importance is not None:
            info['feature_importance'] = dict(zip(self.feature_names, importance))
        
        return info
    
    def __str__(self) -> str:
        """String representation of the model"""
        return f"{self.__class__.__name__}(name='{self.name}', fitted={self.is_fitted})"
    
    def __repr__(self) -> str:
        """Detailed string representation"""
        return f"{self.__class__.__name__}(name='{self.name}', parameters={self.parameters})"
