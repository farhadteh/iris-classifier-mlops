"""
Feature engineering utilities for Iris classification
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class FeatureEngineer:
    """Feature engineering class for creating derived features"""
    
    def __init__(self):
        self.feature_names = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
        self.engineered_features = []
    
    def create_ratio_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create ratio-based features
        
        Args:
            df: Input DataFrame with original features
            
        Returns:
            DataFrame with additional ratio features
        """
        logger.info("Creating ratio features")
        
        df_new = df.copy()
        
        # Length ratios
        df_new['sepal_ratio'] = df['sepal_length'] / df['sepal_width']
        df_new['petal_ratio'] = df['petal_length'] / df['petal_width']
        df_new['length_ratio'] = df['sepal_length'] / df['petal_length']
        df_new['width_ratio'] = df['sepal_width'] / df['petal_width']
        
        # Area approximations (assuming elliptical shape)
        df_new['sepal_area'] = np.pi * (df['sepal_length'] / 2) * (df['sepal_width'] / 2)
        df_new['petal_area'] = np.pi * (df['petal_length'] / 2) * (df['petal_width'] / 2)
        df_new['area_ratio'] = df_new['sepal_area'] / df_new['petal_area']
        
        new_features = ['sepal_ratio', 'petal_ratio', 'length_ratio', 'width_ratio', 
                       'sepal_area', 'petal_area', 'area_ratio']
        self.engineered_features.extend(new_features)
        
        logger.info(f"Created {len(new_features)} ratio features")
        return df_new
    
    def create_statistical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create statistical features across original dimensions
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with statistical features
        """
        logger.info("Creating statistical features")
        
        df_new = df.copy()
        
        # Basic statistics across all features
        feature_cols = [col for col in df.columns if col in self.feature_names]
        
        df_new['feature_mean'] = df[feature_cols].mean(axis=1)
        df_new['feature_std'] = df[feature_cols].std(axis=1)
        df_new['feature_min'] = df[feature_cols].min(axis=1)
        df_new['feature_max'] = df[feature_cols].max(axis=1)
        df_new['feature_range'] = df_new['feature_max'] - df_new['feature_min']
        
        # Coefficient of variation
        df_new['feature_cv'] = df_new['feature_std'] / df_new['feature_mean']
        
        new_features = ['feature_mean', 'feature_std', 'feature_min', 
                       'feature_max', 'feature_range', 'feature_cv']
        self.engineered_features.extend(new_features)
        
        logger.info(f"Created {len(new_features)} statistical features")
        return df_new
    
    def create_interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create interaction features between original features
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with interaction features
        """
        logger.info("Creating interaction features")
        
        df_new = df.copy()
        
        # Multiplicative interactions
        df_new['sepal_interaction'] = df['sepal_length'] * df['sepal_width']
        df_new['petal_interaction'] = df['petal_length'] * df['petal_width']
        df_new['length_interaction'] = df['sepal_length'] * df['petal_length']
        df_new['width_interaction'] = df['sepal_width'] * df['petal_width']
        
        # Additive interactions
        df_new['total_length'] = df['sepal_length'] + df['petal_length']
        df_new['total_width'] = df['sepal_width'] + df['petal_width']
        df_new['total_size'] = df_new['total_length'] + df_new['total_width']
        
        new_features = ['sepal_interaction', 'petal_interaction', 'length_interaction',
                       'width_interaction', 'total_length', 'total_width', 'total_size']
        self.engineered_features.extend(new_features)
        
        logger.info(f"Created {len(new_features)} interaction features")
        return df_new
    
    def create_polynomial_features(self, df: pd.DataFrame, degree: int = 2) -> pd.DataFrame:
        """
        Create polynomial features
        
        Args:
            df: Input DataFrame
            degree: Polynomial degree
            
        Returns:
            DataFrame with polynomial features
        """
        logger.info(f"Creating polynomial features of degree {degree}")
        
        df_new = df.copy()
        
        for col in self.feature_names:
            if col in df.columns:
                for d in range(2, degree + 1):
                    new_col = f"{col}_power_{d}"
                    df_new[new_col] = df[col] ** d
                    self.engineered_features.append(new_col)
        
        logger.info(f"Created polynomial features for {len(self.feature_names)} base features")
        return df_new
    
    def create_all_features(self, df: pd.DataFrame, include_polynomial: bool = False) -> pd.DataFrame:
        """
        Create all engineered features
        
        Args:
            df: Input DataFrame
            include_polynomial: Whether to include polynomial features
            
        Returns:
            DataFrame with all engineered features
        """
        logger.info("Creating all engineered features")
        
        # Reset engineered features list
        self.engineered_features = []
        
        # Apply all feature engineering methods
        df_engineered = self.create_ratio_features(df)
        df_engineered = self.create_statistical_features(df_engineered)
        df_engineered = self.create_interaction_features(df_engineered)
        
        if include_polynomial:
            df_engineered = self.create_polynomial_features(df_engineered)
        
        logger.info(f"Total engineered features created: {len(self.engineered_features)}")
        return df_engineered
    
    def get_engineered_feature_names(self) -> List[str]:
        """Get list of engineered feature names"""
        return self.engineered_features.copy()
    
    def get_feature_importance_mapping(self) -> Dict[str, str]:
        """Get mapping of engineered features to their categories"""
        mapping = {}
        
        for feature in self.engineered_features:
            if 'ratio' in feature:
                mapping[feature] = 'ratio'
            elif 'area' in feature:
                mapping[feature] = 'geometric'
            elif 'interaction' in feature or 'total' in feature:
                mapping[feature] = 'interaction'
            elif 'power' in feature:
                mapping[feature] = 'polynomial'
            elif 'feature_' in feature:
                mapping[feature] = 'statistical'
            else:
                mapping[feature] = 'other'
        
        return mapping
