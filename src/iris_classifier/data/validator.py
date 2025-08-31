"""
Data validation utilities
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
import logging

logger = logging.getLogger(__name__)


class DataValidator:
    """Data validation utilities for Iris dataset"""
    
    def __init__(self):
        self.expected_features = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
        self.expected_classes = ['setosa', 'versicolor', 'virginica']
        self.feature_ranges = {
            'sepal_length': (0.1, 10.0),
            'sepal_width': (0.1, 10.0),
            'petal_length': (0.1, 10.0),
            'petal_width': (0.1, 10.0)
        }
    
    def validate_features(self, data: pd.DataFrame) -> Tuple[bool, List[str]]:
        """
        Validate feature columns in the dataset
        
        Args:
            data: Input DataFrame
            
        Returns:
            Tuple of (is_valid, list_of_errors)
        """
        errors = []
        
        # Check if data is DataFrame
        if not isinstance(data, pd.DataFrame):
            errors.append("Input data must be a pandas DataFrame")
            return False, errors
        
        # Check for required columns
        missing_cols = set(self.expected_features) - set(data.columns)
        if missing_cols:
            errors.append(f"Missing required columns: {missing_cols}")
        
        # Check for extra columns (warning, not error)
        extra_cols = set(data.columns) - set(self.expected_features)
        if extra_cols:
            logger.warning(f"Extra columns found (will be ignored): {extra_cols}")
        
        # Check data types
        for col in self.expected_features:
            if col in data.columns:
                if not pd.api.types.is_numeric_dtype(data[col]):
                    errors.append(f"Column '{col}' must be numeric")
        
        return len(errors) == 0, errors
    
    def validate_feature_values(self, data: pd.DataFrame) -> Tuple[bool, List[str]]:
        """
        Validate feature value ranges
        
        Args:
            data: Input DataFrame
            
        Returns:
            Tuple of (is_valid, list_of_errors)
        """
        errors = []
        
        for col in self.expected_features:
            if col not in data.columns:
                continue
                
            min_val, max_val = self.feature_ranges[col]
            
            # Check for negative values
            if (data[col] < 0).any():
                errors.append(f"Column '{col}' contains negative values")
            
            # Check for extreme values
            if (data[col] < min_val).any():
                errors.append(f"Column '{col}' contains values below minimum ({min_val})")
            
            if (data[col] > max_val).any():
                errors.append(f"Column '{col}' contains values above maximum ({max_val})")
            
            # Check for missing values
            if data[col].isnull().any():
                errors.append(f"Column '{col}' contains missing values")
            
            # Check for infinite values
            if np.isinf(data[col]).any():
                errors.append(f"Column '{col}' contains infinite values")
        
        return len(errors) == 0, errors
    
    def validate_targets(self, targets: pd.Series) -> Tuple[bool, List[str]]:
        """
        Validate target values
        
        Args:
            targets: Target Series
            
        Returns:
            Tuple of (is_valid, list_of_errors)
        """
        errors = []
        
        if targets is None or len(targets) == 0:
            errors.append("Target values are empty")
            return False, errors
        
        # Check for missing values
        if targets.isnull().any():
            errors.append("Target contains missing values")
        
        # Check for valid class names (if string targets)
        if targets.dtype == 'object':
            invalid_classes = set(targets.unique()) - set(self.expected_classes)
            if invalid_classes:
                errors.append(f"Invalid class names found: {invalid_classes}")
        
        # Check for valid numeric targets (if numeric)
        elif pd.api.types.is_numeric_dtype(targets):
            invalid_indices = (targets < 0) | (targets > 2)
            if invalid_indices.any():
                errors.append("Numeric targets must be 0, 1, or 2")
        
        return len(errors) == 0, errors
    
    def validate_prediction_input(self, input_data: Dict[str, float]) -> Tuple[bool, List[str]]:
        """
        Validate single prediction input
        
        Args:
            input_data: Dictionary with feature values
            
        Returns:
            Tuple of (is_valid, list_of_errors)
        """
        errors = []
        
        # Check required features
        missing_features = set(self.expected_features) - set(input_data.keys())
        if missing_features:
            errors.append(f"Missing required features: {missing_features}")
        
        # Validate each feature value
        for feature in self.expected_features:
            if feature not in input_data:
                continue
            
            value = input_data[feature]
            
            # Check if numeric
            try:
                value = float(value)
            except (ValueError, TypeError):
                errors.append(f"Feature '{feature}' must be numeric")
                continue
            
            # Check range
            min_val, max_val = self.feature_ranges[feature]
            if value < min_val or value > max_val:
                errors.append(f"Feature '{feature}' value {value} is outside valid range [{min_val}, {max_val}]")
            
            # Check for special values
            if np.isnan(value) or np.isinf(value):
                errors.append(f"Feature '{feature}' cannot be NaN or infinite")
        
        return len(errors) == 0, errors
    
    def validate_batch_input(self, batch_data: List[Dict[str, float]]) -> Tuple[bool, List[str], List[int]]:
        """
        Validate batch prediction input
        
        Args:
            batch_data: List of dictionaries with feature values
            
        Returns:
            Tuple of (is_valid, list_of_errors, list_of_invalid_indices)
        """
        errors = []
        invalid_indices = []
        
        if not batch_data:
            errors.append("Batch data is empty")
            return False, errors, invalid_indices
        
        for i, sample in enumerate(batch_data):
            is_valid, sample_errors = self.validate_prediction_input(sample)
            if not is_valid:
                invalid_indices.append(i)
                for error in sample_errors:
                    errors.append(f"Sample {i}: {error}")
        
        return len(invalid_indices) == 0, errors, invalid_indices
    
    def get_data_summary(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Get summary statistics for the dataset
        
        Args:
            data: Input DataFrame
            
        Returns:
            Dictionary with data summary
        """
        summary = {
            'shape': data.shape,
            'columns': data.columns.tolist(),
            'missing_values': data.isnull().sum().to_dict(),
            'data_types': data.dtypes.to_dict()
        }
        
        # Add statistics for numeric columns
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            summary['statistics'] = data[numeric_cols].describe().to_dict()
        
        return summary
    
    def validate_full_dataset(self, data: pd.DataFrame, targets: Optional[pd.Series] = None) -> Dict[str, Any]:
        """
        Perform comprehensive dataset validation
        
        Args:
            data: Feature DataFrame
            targets: Target Series (optional)
            
        Returns:
            Dictionary with validation results
        """
        validation_results = {
            'is_valid': True,
            'errors': [],
            'warnings': [],
            'summary': self.get_data_summary(data)
        }
        
        # Validate features
        features_valid, feature_errors = self.validate_features(data)
        if not features_valid:
            validation_results['is_valid'] = False
            validation_results['errors'].extend(feature_errors)
        
        # Validate feature values
        values_valid, value_errors = self.validate_feature_values(data)
        if not values_valid:
            validation_results['is_valid'] = False
            validation_results['errors'].extend(value_errors)
        
        # Validate targets if provided
        if targets is not None:
            targets_valid, target_errors = self.validate_targets(targets)
            if not targets_valid:
                validation_results['is_valid'] = False
                validation_results['errors'].extend(target_errors)
        
        logger.info(f"Dataset validation completed. Valid: {validation_results['is_valid']}")
        if validation_results['errors']:
            logger.warning(f"Validation errors: {validation_results['errors']}")
        
        return validation_results
