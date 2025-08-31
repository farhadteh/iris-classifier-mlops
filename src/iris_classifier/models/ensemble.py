"""
Ensemble model for combining multiple Iris classifiers
"""

import numpy as np
from typing import List, Dict, Any, Optional
from collections import Counter
import logging

from .base import BaseModel

logger = logging.getLogger(__name__)


class EnsembleModel(BaseModel):
    """Ensemble model that combines multiple base models"""
    
    def __init__(self, models: List[BaseModel], voting: str = 'soft', weights: Optional[List[float]] = None):
        """
        Initialize ensemble model
        
        Args:
            models: List of base models to ensemble
            voting: Voting strategy ('soft' or 'hard')
            weights: Weights for each model (optional)
        """
        super().__init__(name="Ensemble")
        
        self.models = models
        self.voting = voting
        self.weights = weights
        
        if voting not in ['soft', 'hard']:
            raise ValueError("Voting must be 'soft' or 'hard'")
        
        if weights is not None and len(weights) != len(models):
            raise ValueError("Number of weights must match number of models")
        
        logger.info(f"Initialized ensemble with {len(models)} models, voting={voting}")
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'EnsembleModel':
        """
        Fit all models in the ensemble
        
        Args:
            X: Training features
            y: Training targets
            
        Returns:
            Self for method chaining
        """
        logger.info("Training ensemble models")
        
        for i, model in enumerate(self.models):
            logger.info(f"Training model {i+1}/{len(self.models)}: {model.name}")
            model.fit(X, y)
        
        self.is_fitted = True
        logger.info("Ensemble training completed")
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions using ensemble voting
        
        Args:
            X: Input features
            
        Returns:
            Predicted class labels
        """
        if not self.is_fitted:
            raise ValueError("Ensemble must be fitted before making predictions")
        
        if self.voting == 'soft':
            return self._predict_soft_voting(X)
        else:
            return self._predict_hard_voting(X)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities using ensemble averaging
        
        Args:
            X: Input features
            
        Returns:
            Class probabilities
        """
        if not self.is_fitted:
            raise ValueError("Ensemble must be fitted before making predictions")
        
        predictions = []
        total_weight = 0
        
        for i, model in enumerate(self.models):
            model_proba = model.predict_proba(X)
            if model_proba is not None:
                weight = self.weights[i] if self.weights else 1.0
                predictions.append(model_proba * weight)
                total_weight += weight
            else:
                logger.warning(f"Model {model.name} does not support probability prediction")
        
        if not predictions:
            raise ValueError("No models support probability prediction")
        
        # Average probabilities
        ensemble_proba = np.sum(predictions, axis=0) / total_weight
        return ensemble_proba
    
    def _predict_soft_voting(self, X: np.ndarray) -> np.ndarray:
        """Soft voting prediction based on probabilities"""
        probabilities = self.predict_proba(X)
        return np.argmax(probabilities, axis=1)
    
    def _predict_hard_voting(self, X: np.ndarray) -> np.ndarray:
        """Hard voting prediction based on majority vote"""
        predictions = []
        
        for model in self.models:
            model_pred = model.predict(X)
            predictions.append(model_pred)
        
        # Convert to numpy array for easier manipulation
        predictions = np.array(predictions).T  # Shape: (n_samples, n_models)
        
        # Apply weights if provided
        if self.weights:
            weighted_predictions = []
            for sample_preds in predictions:
                vote_counts = Counter()
                for pred, weight in zip(sample_preds, self.weights):
                    vote_counts[pred] += weight
                weighted_predictions.append(vote_counts.most_common(1)[0][0])
            return np.array(weighted_predictions)
        else:
            # Simple majority vote
            ensemble_pred = []
            for sample_preds in predictions:
                vote_counts = Counter(sample_preds)
                ensemble_pred.append(vote_counts.most_common(1)[0][0])
            return np.array(ensemble_pred)
    
    def get_model_predictions(self, X: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Get predictions from all individual models
        
        Args:
            X: Input features
            
        Returns:
            Dictionary mapping model names to predictions
        """
        if not self.is_fitted:
            raise ValueError("Ensemble must be fitted before making predictions")
        
        model_predictions = {}
        for model in self.models:
            model_predictions[model.name] = model.predict(X)
        
        return model_predictions
    
    def get_model_probabilities(self, X: np.ndarray) -> Dict[str, Optional[np.ndarray]]:
        """
        Get probability predictions from all individual models
        
        Args:
            X: Input features
            
        Returns:
            Dictionary mapping model names to probabilities
        """
        if not self.is_fitted:
            raise ValueError("Ensemble must be fitted before making predictions")
        
        model_probabilities = {}
        for model in self.models:
            model_probabilities[model.name] = model.predict_proba(X)
        
        return model_probabilities
    
    def get_model_weights(self) -> Optional[List[float]]:
        """Get the weights assigned to each model"""
        return self.weights
    
    def set_weights(self, weights: List[float]) -> 'EnsembleModel':
        """
        Set new weights for the models
        
        Args:
            weights: New weights for each model
            
        Returns:
            Self for method chaining
        """
        if len(weights) != len(self.models):
            raise ValueError("Number of weights must match number of models")
        
        self.weights = weights
        logger.info(f"Updated ensemble weights: {weights}")
        return self
    
    def evaluate_individual_models(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Dict[str, float]]:
        """
        Evaluate each individual model in the ensemble
        
        Args:
            X: Test features
            y: True labels
            
        Returns:
            Dictionary with evaluation results for each model
        """
        from ..utils.metrics import MetricsCalculator
        
        results = {}
        
        for model in self.models:
            predictions = model.predict(X)
            metrics = MetricsCalculator.calculate_classification_metrics(y, predictions)
            results[model.name] = metrics
        
        # Add ensemble results
        ensemble_predictions = self.predict(X)
        ensemble_metrics = MetricsCalculator.calculate_classification_metrics(y, ensemble_predictions)
        results['Ensemble'] = ensemble_metrics
        
        return results
    
    def get_feature_importance(self) -> Optional[Dict[str, np.ndarray]]:
        """
        Get feature importance from all models that support it
        
        Returns:
            Dictionary mapping model names to feature importance arrays
        """
        importance_dict = {}
        
        for model in self.models:
            importance = model.get_feature_importance()
            if importance is not None:
                importance_dict[model.name] = importance
        
        return importance_dict if importance_dict else None
    
    def get_ensemble_info(self) -> Dict[str, Any]:
        """
        Get comprehensive information about the ensemble
        
        Returns:
            Dictionary with ensemble information
        """
        model_info = []
        for i, model in enumerate(self.models):
            info = model.get_model_info()
            if self.weights:
                info['weight'] = self.weights[i]
            model_info.append(info)
        
        return {
            'ensemble_type': 'EnsembleModel',
            'voting_strategy': self.voting,
            'n_models': len(self.models),
            'weights': self.weights,
            'is_fitted': self.is_fitted,
            'models': model_info
        }
    
    def __str__(self) -> str:
        """String representation of the ensemble"""
        model_names = [model.name for model in self.models]
        return f"EnsembleModel(models={model_names}, voting={self.voting})"
    
    def __repr__(self) -> str:
        """Detailed string representation"""
        return (f"EnsembleModel(n_models={len(self.models)}, "
                f"voting={self.voting}, weights={self.weights})")
