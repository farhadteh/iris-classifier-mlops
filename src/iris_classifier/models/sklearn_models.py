"""
Scikit-learn model implementations for Iris classification
"""

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from typing import Dict, Any, Optional
import logging

from .base import BaseModel

logger = logging.getLogger(__name__)


class LogisticRegressionModel(BaseModel):
    """Logistic Regression model for Iris classification"""
    
    def __init__(self, **kwargs):
        """Initialize Logistic Regression model"""
        default_params = {
            'solver': 'lbfgs',
            'max_iter': 1000,
            'multi_class': 'auto',
            'random_state': 42
        }
        default_params.update(kwargs)
        
        super().__init__(name="LogisticRegression", **default_params)
        self.model = LogisticRegression(**self.parameters)
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'LogisticRegressionModel':
        """Train the logistic regression model"""
        logger.info(f"Training {self.name} with parameters: {self.parameters}")
        
        self.model.fit(X, y)
        self.is_fitted = True
        
        logger.info(f"{self.name} training completed")
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions with the logistic regression model"""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        return self.model.predict(X)
    
    def get_coefficients(self) -> Dict[str, np.ndarray]:
        """Get model coefficients"""
        if not self.is_fitted:
            raise ValueError("Model must be fitted to get coefficients")
        
        return {
            'coefficients': self.model.coef_,
            'intercept': self.model.intercept_
        }


class RandomForestModel(BaseModel):
    """Random Forest model for Iris classification"""
    
    def __init__(self, **kwargs):
        """Initialize Random Forest model"""
        default_params = {
            'n_estimators': 100,
            'max_depth': None,
            'min_samples_split': 2,
            'min_samples_leaf': 1,
            'random_state': 42
        }
        default_params.update(kwargs)
        
        super().__init__(name="RandomForest", **default_params)
        self.model = RandomForestClassifier(**self.parameters)
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'RandomForestModel':
        """Train the random forest model"""
        logger.info(f"Training {self.name} with parameters: {self.parameters}")
        
        self.model.fit(X, y)
        self.is_fitted = True
        
        logger.info(f"{self.name} training completed")
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions with the random forest model"""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        return self.model.predict(X)
    
    def get_tree_count(self) -> int:
        """Get number of trees in the forest"""
        return self.model.n_estimators if self.is_fitted else self.parameters.get('n_estimators', 0)


class SVMModel(BaseModel):
    """Support Vector Machine model for Iris classification"""
    
    def __init__(self, **kwargs):
        """Initialize SVM model"""
        default_params = {
            'C': 1.0,
            'kernel': 'rbf',
            'probability': True,  # Enable probability predictions
            'random_state': 42
        }
        default_params.update(kwargs)
        
        super().__init__(name="SVM", **default_params)
        self.model = SVC(**self.parameters)
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'SVMModel':
        """Train the SVM model"""
        logger.info(f"Training {self.name} with parameters: {self.parameters}")
        
        self.model.fit(X, y)
        self.is_fitted = True
        
        logger.info(f"{self.name} training completed")
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions with the SVM model"""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        return self.model.predict(X)
    
    def get_support_vectors(self) -> Dict[str, Any]:
        """Get support vector information"""
        if not self.is_fitted:
            raise ValueError("Model must be fitted to get support vectors")
        
        return {
            'support_vectors': self.model.support_vectors_,
            'support_indices': self.model.support_,
            'n_support': self.model.n_support_
        }


class HyperparameterTunedModel:
    """Wrapper for hyperparameter tuning of any model"""
    
    def __init__(self, base_model_class, param_grid: Dict[str, list], cv: int = 5, scoring: str = 'accuracy'):
        """
        Initialize hyperparameter tuned model
        
        Args:
            base_model_class: Base model class to tune
            param_grid: Parameter grid for tuning
            cv: Number of cross-validation folds
            scoring: Scoring metric for evaluation
        """
        self.base_model_class = base_model_class
        self.param_grid = param_grid
        self.cv = cv
        self.scoring = scoring
        self.best_model = None
        self.grid_search = None
        self.is_fitted = False
        
        logger.info(f"Initialized hyperparameter tuning for {base_model_class.__name__}")
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'HyperparameterTunedModel':
        """
        Perform hyperparameter tuning and fit the best model
        
        Args:
            X: Training features
            y: Training targets
            
        Returns:
            Self for method chaining
        """
        logger.info(f"Starting hyperparameter tuning with {self.cv}-fold CV")
        
        # Create base model instance
        base_model = self.base_model_class()
        
        # Perform grid search
        self.grid_search = GridSearchCV(
            estimator=base_model.model,
            param_grid=self.param_grid,
            cv=self.cv,
            scoring=self.scoring,
            n_jobs=-1,
            verbose=1
        )
        
        self.grid_search.fit(X, y)
        
        # Create best model
        self.best_model = self.base_model_class(**self.grid_search.best_params_)
        self.best_model.fit(X, y)
        self.is_fitted = True
        
        logger.info(f"Hyperparameter tuning completed")
        logger.info(f"Best parameters: {self.grid_search.best_params_}")
        logger.info(f"Best CV score: {self.grid_search.best_score_:.4f}")
        
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions with the best model"""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        return self.best_model.predict(X)
    
    def predict_proba(self, X: np.ndarray) -> Optional[np.ndarray]:
        """Predict probabilities with the best model"""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        return self.best_model.predict_proba(X)
    
    def get_best_params(self) -> Dict[str, Any]:
        """Get the best parameters found during tuning"""
        if not self.is_fitted:
            raise ValueError("Model must be fitted to get best parameters")
        
        return self.grid_search.best_params_
    
    def get_cv_results(self) -> Dict[str, Any]:
        """Get detailed cross-validation results"""
        if not self.is_fitted:
            raise ValueError("Model must be fitted to get CV results")
        
        return self.grid_search.cv_results_
    
    def get_best_score(self) -> float:
        """Get the best cross-validation score"""
        if not self.is_fitted:
            raise ValueError("Model must be fitted to get best score")
        
        return self.grid_search.best_score_


# Model factory function
def create_model(model_type: str, **kwargs) -> BaseModel:
    """
    Factory function to create model instances
    
    Args:
        model_type: Type of model to create
        **kwargs: Model parameters
        
    Returns:
        Model instance
    """
    model_classes = {
        'logistic_regression': LogisticRegressionModel,
        'random_forest': RandomForestModel,
        'svm': SVMModel
    }
    
    if model_type not in model_classes:
        raise ValueError(f"Unknown model type: {model_type}. Available: {list(model_classes.keys())}")
    
    return model_classes[model_type](**kwargs)


# Default parameter grids for hyperparameter tuning
DEFAULT_PARAM_GRIDS = {
    'logistic_regression': {
        'solver': ['lbfgs', 'liblinear'],
        'C': [0.1, 1.0, 10.0],
        'max_iter': [1000, 2000]
    },
    'random_forest': {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 5, 10],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2]
    },
    'svm': {
        'C': [0.1, 1.0, 10.0],
        'kernel': ['linear', 'rbf'],
        'gamma': ['scale', 'auto']
    }
}
