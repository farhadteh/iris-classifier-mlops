"""
Metrics calculation utilities
"""

import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)
from typing import Dict, Any
import logging

logger = logging.getLogger(__name__)


class MetricsCalculator:
    """Utility class for calculating model performance metrics"""
    
    @staticmethod
    def calculate_classification_metrics(y_true: np.ndarray, y_pred: np.ndarray, 
                                       average: str = 'weighted') -> Dict[str, float]:
        """
        Calculate standard classification metrics
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            average: Averaging strategy for multiclass metrics
            
        Returns:
            Dictionary of calculated metrics
        """
        try:
            metrics = {
                'accuracy': accuracy_score(y_true, y_pred),
                'precision': precision_score(y_true, y_pred, average=average),
                'recall': recall_score(y_true, y_pred, average=average),
                'f1_score': f1_score(y_true, y_pred, average=average)
            }
            
            logger.debug(f"Calculated metrics: {metrics}")
            return metrics
            
        except Exception as e:
            logger.error(f"Error calculating metrics: {e}")
            return {}
    
    @staticmethod
    def get_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        """Calculate confusion matrix"""
        return confusion_matrix(y_true, y_pred)
    
    @staticmethod
    def get_classification_report(y_true: np.ndarray, y_pred: np.ndarray, 
                                target_names: list = None) -> Dict[str, Any]:
        """Get detailed classification report"""
        return classification_report(
            y_true, y_pred, 
            target_names=target_names,
            output_dict=True
        )
    
    @staticmethod
    def calculate_per_class_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                                  class_names: list = None) -> Dict[str, Dict[str, float]]:
        """
        Calculate per-class metrics
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            class_names: Names of the classes
            
        Returns:
            Dictionary with per-class metrics
        """
        if class_names is None:
            class_names = [f"class_{i}" for i in np.unique(y_true)]
        
        # Calculate per-class precision, recall, and f1-score
        precision = precision_score(y_true, y_pred, average=None)
        recall = recall_score(y_true, y_pred, average=None)
        f1 = f1_score(y_true, y_pred, average=None)
        
        per_class_metrics = {}
        for i, class_name in enumerate(class_names):
            if i < len(precision):
                per_class_metrics[class_name] = {
                    'precision': float(precision[i]),
                    'recall': float(recall[i]),
                    'f1_score': float(f1[i])
                }
        
        return per_class_metrics
    
    @staticmethod
    def log_metrics_to_mlflow(metrics: Dict[str, float], prefix: str = ""):
        """Log metrics to MLflow"""
        import mlflow
        
        for metric_name, value in metrics.items():
            metric_key = f"{prefix}_{metric_name}" if prefix else metric_name
            mlflow.log_metric(metric_key, value)
            logger.debug(f"Logged metric {metric_key}: {value}")
