"""
Feature selection utilities for Iris classification
"""

import numpy as np
import pandas as pd
from sklearn.feature_selection import (
    SelectKBest, f_classif, mutual_info_classif,
    RFE, SelectFromModel
)
from sklearn.ensemble import RandomForestClassifier
from typing import List, Dict, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class FeatureSelector:
    """Feature selection utilities for reducing dimensionality"""
    
    def __init__(self):
        self.selected_features = []
        self.feature_scores = {}
        self.selection_method = None
    
    def select_k_best_features(self, X: pd.DataFrame, y: pd.Series, k: int = 10, 
                              score_func=f_classif) -> Tuple[pd.DataFrame, List[str]]:
        """
        Select k best features using statistical tests
        
        Args:
            X: Feature DataFrame
            y: Target Series
            k: Number of features to select
            score_func: Scoring function for feature selection
            
        Returns:
            Tuple of (selected_features_df, selected_feature_names)
        """
        logger.info(f"Selecting {k} best features using {score_func.__name__}")
        
        selector = SelectKBest(score_func=score_func, k=k)
        X_selected = selector.fit_transform(X, y)
        
        # Get selected feature names
        selected_indices = selector.get_support(indices=True)
        selected_features = X.columns[selected_indices].tolist()
        
        # Store feature scores
        feature_scores = dict(zip(X.columns, selector.scores_))
        self.feature_scores = {k: v for k, v in sorted(feature_scores.items(), 
                                                      key=lambda item: item[1], reverse=True)}
        
        self.selected_features = selected_features
        self.selection_method = f"SelectKBest_{score_func.__name__}"
        
        logger.info(f"Selected features: {selected_features}")
        return pd.DataFrame(X_selected, columns=selected_features, index=X.index), selected_features
    
    def select_by_mutual_information(self, X: pd.DataFrame, y: pd.Series, 
                                   k: int = 10) -> Tuple[pd.DataFrame, List[str]]:
        """
        Select features using mutual information
        
        Args:
            X: Feature DataFrame
            y: Target Series
            k: Number of features to select
            
        Returns:
            Tuple of (selected_features_df, selected_feature_names)
        """
        return self.select_k_best_features(X, y, k, mutual_info_classif)
    
    def select_by_model_importance(self, X: pd.DataFrame, y: pd.Series, 
                                 estimator=None, threshold: str = 'median') -> Tuple[pd.DataFrame, List[str]]:
        """
        Select features based on model feature importance
        
        Args:
            X: Feature DataFrame
            y: Target Series
            estimator: Model to use for importance (default: RandomForest)
            threshold: Importance threshold for selection
            
        Returns:
            Tuple of (selected_features_df, selected_feature_names)
        """
        logger.info(f"Selecting features using model importance (threshold: {threshold})")
        
        if estimator is None:
            estimator = RandomForestClassifier(n_estimators=100, random_state=42)
        
        selector = SelectFromModel(estimator, threshold=threshold)
        X_selected = selector.fit_transform(X, y)
        
        # Get selected feature names
        selected_indices = selector.get_support(indices=True)
        selected_features = X.columns[selected_indices].tolist()
        
        # Store feature importance scores
        estimator.fit(X, y)
        importance_scores = dict(zip(X.columns, estimator.feature_importances_))
        self.feature_scores = {k: v for k, v in sorted(importance_scores.items(), 
                                                      key=lambda item: item[1], reverse=True)}
        
        self.selected_features = selected_features
        self.selection_method = f"ModelBased_{type(estimator).__name__}"
        
        logger.info(f"Selected {len(selected_features)} features using model importance")
        return pd.DataFrame(X_selected, columns=selected_features, index=X.index), selected_features
    
    def recursive_feature_elimination(self, X: pd.DataFrame, y: pd.Series, 
                                    n_features: int = 10, estimator=None) -> Tuple[pd.DataFrame, List[str]]:
        """
        Perform recursive feature elimination
        
        Args:
            X: Feature DataFrame
            y: Target Series
            n_features: Number of features to select
            estimator: Model to use for elimination
            
        Returns:
            Tuple of (selected_features_df, selected_feature_names)
        """
        logger.info(f"Performing recursive feature elimination to select {n_features} features")
        
        if estimator is None:
            estimator = RandomForestClassifier(n_estimators=50, random_state=42)
        
        selector = RFE(estimator, n_features_to_select=n_features)
        X_selected = selector.fit_transform(X, y)
        
        # Get selected feature names
        selected_indices = selector.get_support(indices=True)
        selected_features = X.columns[selected_indices].tolist()
        
        # Store ranking scores (lower is better)
        ranking_scores = dict(zip(X.columns, selector.ranking_))
        self.feature_scores = {k: 1.0/v for k, v in sorted(ranking_scores.items(), 
                                                           key=lambda item: item[1])}
        
        self.selected_features = selected_features
        self.selection_method = f"RFE_{type(estimator).__name__}"
        
        logger.info(f"RFE selected features: {selected_features}")
        return pd.DataFrame(X_selected, columns=selected_features, index=X.index), selected_features
    
    def select_by_correlation_threshold(self, X: pd.DataFrame, threshold: float = 0.9) -> Tuple[pd.DataFrame, List[str]]:
        """
        Remove highly correlated features
        
        Args:
            X: Feature DataFrame
            threshold: Correlation threshold for removal
            
        Returns:
            Tuple of (filtered_df, remaining_feature_names)
        """
        logger.info(f"Removing features with correlation > {threshold}")
        
        # Calculate correlation matrix
        corr_matrix = X.corr().abs()
        
        # Find pairs of highly correlated features
        upper_triangle = corr_matrix.where(
            np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
        )
        
        # Find features to drop
        to_drop = [column for column in upper_triangle.columns 
                  if any(upper_triangle[column] > threshold)]
        
        # Keep remaining features
        remaining_features = [col for col in X.columns if col not in to_drop]
        X_filtered = X[remaining_features]
        
        self.selected_features = remaining_features
        self.selection_method = f"CorrelationFilter_{threshold}"
        
        logger.info(f"Removed {len(to_drop)} highly correlated features")
        logger.info(f"Remaining features: {remaining_features}")
        
        return X_filtered, remaining_features
    
    def select_by_variance_threshold(self, X: pd.DataFrame, threshold: float = 0.01) -> Tuple[pd.DataFrame, List[str]]:
        """
        Remove low-variance features
        
        Args:
            X: Feature DataFrame
            threshold: Variance threshold
            
        Returns:
            Tuple of (filtered_df, remaining_feature_names)
        """
        logger.info(f"Removing features with variance < {threshold}")
        
        # Calculate variance for each feature
        variances = X.var()
        
        # Select features above threshold
        high_variance_features = variances[variances >= threshold].index.tolist()
        X_filtered = X[high_variance_features]
        
        self.selected_features = high_variance_features
        self.selection_method = f"VarianceThreshold_{threshold}"
        
        logger.info(f"Removed {len(X.columns) - len(high_variance_features)} low-variance features")
        return X_filtered, high_variance_features
    
    def comprehensive_feature_selection(self, X: pd.DataFrame, y: pd.Series, 
                                      max_features: int = 10) -> Tuple[pd.DataFrame, List[str], Dict[str, float]]:
        """
        Perform comprehensive feature selection using multiple methods
        
        Args:
            X: Feature DataFrame
            y: Target Series
            max_features: Maximum number of features to select
            
        Returns:
            Tuple of (selected_df, selected_features, feature_scores)
        """
        logger.info("Performing comprehensive feature selection")
        
        # Step 1: Remove low variance features
        X_variance, _ = self.select_by_variance_threshold(X, threshold=0.01)
        
        # Step 2: Remove highly correlated features
        X_corr, _ = self.select_by_correlation_threshold(X_variance, threshold=0.95)
        
        # Step 3: Select best features using multiple methods
        methods_scores = {}
        
        # Statistical test
        try:
            _, features_stat = self.select_k_best_features(X_corr, y, k=min(max_features, len(X_corr.columns)))
            methods_scores['statistical'] = {f: self.feature_scores.get(f, 0) for f in features_stat}
        except Exception as e:
            logger.warning(f"Statistical selection failed: {e}")
            methods_scores['statistical'] = {}
        
        # Model-based importance
        try:
            _, features_model = self.select_by_model_importance(X_corr, y, threshold='median')
            methods_scores['model_based'] = {f: self.feature_scores.get(f, 0) for f in features_model}
        except Exception as e:
            logger.warning(f"Model-based selection failed: {e}")
            methods_scores['model_based'] = {}
        
        # Combine scores from different methods
        all_features = set()
        for method_features in methods_scores.values():
            all_features.update(method_features.keys())
        
        # Calculate combined scores
        combined_scores = {}
        for feature in all_features:
            score = 0
            count = 0
            for method_scores in methods_scores.values():
                if feature in method_scores:
                    score += method_scores[feature]
                    count += 1
            combined_scores[feature] = score / max(count, 1)
        
        # Select top features
        top_features = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)[:max_features]
        selected_features = [f[0] for f in top_features]
        final_scores = {f[0]: f[1] for f in top_features}
        
        X_final = X[selected_features]
        
        self.selected_features = selected_features
        self.feature_scores = final_scores
        self.selection_method = "Comprehensive"
        
        logger.info(f"Comprehensive selection completed. Selected {len(selected_features)} features")
        return X_final, selected_features, final_scores
    
    def get_feature_selection_report(self) -> Dict[str, any]:
        """
        Generate a report of the feature selection process
        
        Returns:
            Dictionary with selection details
        """
        return {
            'selection_method': self.selection_method,
            'n_selected_features': len(self.selected_features),
            'selected_features': self.selected_features,
            'feature_scores': self.feature_scores,
            'top_5_features': dict(list(self.feature_scores.items())[:5]) if self.feature_scores else {}
        }
