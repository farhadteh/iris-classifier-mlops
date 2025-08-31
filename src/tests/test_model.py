"""
Tests for model utilities and functions
"""

import pytest
import numpy as np
import pandas as pd
import os
import sys
import tempfile
import shutil

# Add the parent directory to the path so we can import our modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model import (
    ModelManager, IrisPredictor, DataProcessor, 
    create_sample_data, get_model_info, setup_mlflow_tracking
)

class TestDataProcessor:
    """Test cases for DataProcessor class"""
    
    def test_load_iris_data(self):
        """Test loading Iris dataset"""
        X, y, feature_names, class_names = DataProcessor.load_iris_data()
        
        # Check data shapes and types
        assert isinstance(X, np.ndarray)
        assert isinstance(y, np.ndarray)
        assert X.shape[0] == y.shape[0]
        assert X.shape[1] == 4  # 4 features
        assert len(feature_names) == 4
        assert len(class_names) == 3
        
        # Check feature names
        expected_features = ['sepal length (cm)', 'sepal width (cm)', 
                           'petal length (cm)', 'petal width (cm)']
        for expected in expected_features:
            assert any(expected in name for name in feature_names)
        
        # Check class names
        expected_classes = ['setosa', 'versicolor', 'virginica']
        for expected in expected_classes:
            assert expected in class_names
    
    def test_validate_input_valid(self):
        """Test input validation with valid data"""
        valid_inputs = [
            {
                "sepal_length": 5.1,
                "sepal_width": 3.5,
                "petal_length": 1.4,
                "petal_width": 0.2
            },
            {
                "sepal_length": 0.1,
                "sepal_width": 0.1,
                "petal_length": 0.1,
                "petal_width": 0.1
            },
            {
                "sepal_length": 10.0,
                "sepal_width": 5.0,
                "petal_length": 7.0,
                "petal_width": 3.0
            }
        ]
        
        for data in valid_inputs:
            assert DataProcessor.validate_input(data) == True
    
    def test_validate_input_invalid(self):
        """Test input validation with invalid data"""
        invalid_inputs = [
            # Missing fields
            {"sepal_length": 5.1, "sepal_width": 3.5, "petal_length": 1.4},
            {"sepal_width": 3.5, "petal_length": 1.4, "petal_width": 0.2},
            
            # Negative values
            {"sepal_length": -1, "sepal_width": 3.5, "petal_length": 1.4, "petal_width": 0.2},
            {"sepal_length": 5.1, "sepal_width": -1, "petal_length": 1.4, "petal_width": 0.2},
            
            # Invalid types
            {"sepal_length": "invalid", "sepal_width": 3.5, "petal_length": 1.4, "petal_width": 0.2},
            {"sepal_length": None, "sepal_width": 3.5, "petal_length": 1.4, "petal_width": 0.2},
            
            # Empty dict
            {},
            
            # Extra fields only (missing required)
            {"extra_field": 1.0}
        ]
        
        for data in invalid_inputs:
            assert DataProcessor.validate_input(data) == False
    
    def test_prepare_features(self):
        """Test feature preparation"""
        data = {
            "sepal_length": 5.1,
            "sepal_width": 3.5,
            "petal_length": 1.4,
            "petal_width": 0.2
        }
        
        features = DataProcessor.prepare_features(data)
        
        assert isinstance(features, np.ndarray)
        assert features.shape == (1, 4)
        assert np.array_equal(features[0], [5.1, 3.5, 1.4, 0.2])

class TestModelManager:
    """Test cases for ModelManager class"""
    
    def setUp(self):
        """Set up test environment"""
        self.temp_dir = tempfile.mkdtemp()
        self.tracking_uri = f"file:{self.temp_dir}/mlruns"
    
    def tearDown(self):
        """Clean up test environment"""
        if hasattr(self, 'temp_dir') and os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    def test_model_manager_initialization(self):
        """Test ModelManager initialization"""
        # Use default tracking URI
        manager = ModelManager()
        assert manager.tracking_uri == "file:./mlruns"
        assert manager.client is not None
        
        # Use custom tracking URI
        custom_uri = "file:./test_mlruns"
        manager = ModelManager(custom_uri)
        assert manager.tracking_uri == custom_uri
    
    def test_list_experiments_empty(self):
        """Test listing experiments when none exist"""
        self.setUp()
        manager = ModelManager(self.tracking_uri)
        experiments = manager.list_experiments()
        
        # Should have at least the default experiment
        assert isinstance(experiments, list)
        assert len(experiments) >= 1
        
        # Check experiment structure
        if experiments:
            exp = experiments[0]
            assert "experiment_id" in exp
            assert "name" in exp
            assert "lifecycle_stage" in exp
        
        self.tearDown()
    
    def test_list_models_empty(self):
        """Test listing models when none are registered"""
        self.setUp()
        manager = ModelManager(self.tracking_uri)
        models = manager.list_models()
        
        # Should return empty list when no models are registered
        assert isinstance(models, list)
        assert len(models) == 0
        
        self.tearDown()

class TestIrisPredictor:
    """Test cases for IrisPredictor class"""
    
    def test_iris_predictor_initialization(self):
        """Test IrisPredictor initialization"""
        predictor = IrisPredictor()
        
        assert predictor.model is None
        assert predictor.class_names == ['setosa', 'versicolor', 'virginica']
        assert predictor.feature_names == ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
    
    def test_predict_without_model(self):
        """Test prediction without loaded model"""
        predictor = IrisPredictor()
        
        with pytest.raises(ValueError, match="Model not loaded"):
            predictor.predict_single(5.1, 3.5, 1.4, 0.2)
    
    def test_batch_predict_without_model(self):
        """Test batch prediction without loaded model"""
        predictor = IrisPredictor()
        
        test_data = [
            {"sepal_length": 5.1, "sepal_width": 3.5, "petal_length": 1.4, "petal_width": 0.2}
        ]
        
        with pytest.raises(ValueError, match="Model not loaded"):
            predictor.predict_batch(test_data)
    
    def test_evaluate_without_model(self):
        """Test evaluation without loaded model"""
        predictor = IrisPredictor()
        
        with pytest.raises(ValueError, match="Model not loaded"):
            predictor.evaluate_model()

class TestUtilityFunctions:
    """Test cases for utility functions"""
    
    def test_create_sample_data(self):
        """Test sample data creation"""
        # Test default parameters
        samples = create_sample_data()
        assert len(samples) == 10
        
        # Test custom number of samples
        samples = create_sample_data(n_samples=5)
        assert len(samples) == 5
        
        # Check sample structure
        for sample in samples:
            assert isinstance(sample, dict)
            assert "sepal_length" in sample
            assert "sepal_width" in sample
            assert "petal_length" in sample
            assert "petal_width" in sample
            
            # Check all values are positive
            for key, value in sample.items():
                assert isinstance(value, (int, float))
                assert value > 0
    
    def test_setup_mlflow_tracking(self):
        """Test MLflow tracking setup"""
        temp_dir = tempfile.mkdtemp()
        try:
            tracking_uri = f"file:{temp_dir}/test_mlruns"
            setup_mlflow_tracking(tracking_uri)
            
            # Check if directory was created
            expected_path = f"{temp_dir}/test_mlruns"
            assert os.path.exists(expected_path)
            
        finally:
            shutil.rmtree(temp_dir)
    
    def test_get_model_info_nonexistent(self):
        """Test getting info for non-existent model"""
        info = get_model_info("nonexistent-model")
        assert isinstance(info, dict)
        assert len(info) == 0  # Should return empty dict on error

class TestIntegration:
    """Integration tests"""
    
    def test_data_flow_integration(self):
        """Test the complete data flow from input to validation"""
        # Create sample data
        samples = create_sample_data(n_samples=3)
        
        # Validate each sample
        for sample in samples:
            assert DataProcessor.validate_input(sample)
            
            # Prepare features
            features = DataProcessor.prepare_features(sample)
            assert features.shape == (1, 4)
    
    def test_predictor_with_sample_data(self):
        """Test predictor initialization with sample data"""
        predictor = IrisPredictor()
        samples = create_sample_data(n_samples=2)
        
        # Should work without errors even without a model
        assert predictor.class_names is not None
        assert predictor.feature_names is not None
        
        # Validation should pass for sample data
        for sample in samples:
            assert DataProcessor.validate_input(sample)

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
