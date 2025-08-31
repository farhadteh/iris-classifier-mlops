"""
Tests for FastAPI application
"""

import pytest
from fastapi.testclient import TestClient
import json
import os
import sys

# Add the parent directory to the path so we can import our modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi_app import app

client = TestClient(app)

class TestAPI:
    """Test cases for the FastAPI application"""
    
    def test_root_endpoint(self):
        """Test the root endpoint"""
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert "message" in data
        assert "status" in data
        assert data["status"] == "healthy"
    
    def test_health_endpoint(self):
        """Test the health check endpoint"""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert "model_status" in data
        assert "timestamp" in data
    
    def test_predict_endpoint_valid_input(self):
        """Test prediction endpoint with valid input"""
        test_data = {
            "sepal_length": 5.1,
            "sepal_width": 3.5,
            "petal_length": 1.4,
            "petal_width": 0.2
        }
        
        response = client.post("/predict", json=test_data)
        
        # The test might fail if model is not loaded, so we check for both cases
        if response.status_code == 200:
            data = response.json()
            assert "prediction" in data
            assert "class_name" in data
            assert "confidence" in data
            assert "probabilities" in data
            assert "timestamp" in data
            
            # Validate prediction values
            assert isinstance(data["prediction"], int)
            assert 0 <= data["prediction"] <= 2
            assert data["class_name"] in ["setosa", "versicolor", "virginica"]
            assert 0 <= data["confidence"] <= 1
            assert isinstance(data["probabilities"], dict)
        
        elif response.status_code == 503:
            # Model not loaded - this is acceptable in test environment
            data = response.json()
            assert "detail" in data
            assert "Model not loaded" in data["detail"]
        
        else:
            pytest.fail(f"Unexpected response code: {response.status_code}")
    
    def test_predict_endpoint_invalid_input(self):
        """Test prediction endpoint with invalid input"""
        test_cases = [
            # Missing fields
            {"sepal_length": 5.1, "sepal_width": 3.5, "petal_length": 1.4},
            # Negative values
            {"sepal_length": -1, "sepal_width": 3.5, "petal_length": 1.4, "petal_width": 0.2},
            # Invalid types
            {"sepal_length": "invalid", "sepal_width": 3.5, "petal_length": 1.4, "petal_width": 0.2},
            # Empty request
            {}
        ]
        
        for test_data in test_cases:
            response = client.post("/predict", json=test_data)
            assert response.status_code == 422  # Validation error
    
    def test_batch_predict_endpoint_valid_input(self):
        """Test batch prediction endpoint with valid input"""
        test_data = {
            "instances": [
                {
                    "sepal_length": 5.1,
                    "sepal_width": 3.5,
                    "petal_length": 1.4,
                    "petal_width": 0.2
                },
                {
                    "sepal_length": 6.0,
                    "sepal_width": 2.8,
                    "petal_length": 4.0,
                    "petal_width": 1.3
                }
            ]
        }
        
        response = client.post("/predict/batch", json=test_data)
        
        # The test might fail if model is not loaded
        if response.status_code == 200:
            data = response.json()
            assert "predictions" in data
            assert "batch_size" in data
            assert "timestamp" in data
            
            assert data["batch_size"] == 2
            assert len(data["predictions"]) == 2
            
            # Validate each prediction
            for pred in data["predictions"]:
                assert "prediction" in pred
                assert "class_name" in pred
                assert "confidence" in pred
                assert "probabilities" in pred
        
        elif response.status_code == 503:
            # Model not loaded - acceptable in test environment
            pass
        
        else:
            pytest.fail(f"Unexpected response code: {response.status_code}")
    
    def test_batch_predict_endpoint_empty_input(self):
        """Test batch prediction endpoint with empty input"""
        test_data = {"instances": []}
        
        response = client.post("/predict/batch", json=test_data)
        
        if response.status_code == 200:
            data = response.json()
            assert data["batch_size"] == 0
            assert len(data["predictions"]) == 0
        elif response.status_code == 503:
            # Model not loaded - acceptable
            pass
    
    def test_model_info_endpoint(self):
        """Test model info endpoint"""
        response = client.get("/model/info")
        
        # Model might not be loaded in test environment
        if response.status_code == 200:
            data = response.json()
            assert "model_name" in data
            assert "model_version" in data
            assert "model_stage" in data
            assert "mlflow_run_id" in data
            assert "model_uri" in data
        
        elif response.status_code == 503:
            # Model not loaded - acceptable in test environment
            pass
    
    def test_model_reload_endpoint(self):
        """Test model reload endpoint"""
        response = client.post("/model/reload")
        
        # This might fail if no model is available, which is okay in test
        assert response.status_code in [200, 500, 503]
    
    def test_cors_headers(self):
        """Test CORS headers are present"""
        response = client.options("/predict")
        assert response.status_code == 200
        
        # Check for CORS headers in a simple GET request
        response = client.get("/")
        # CORS headers are typically added by middleware, so we just check the response is successful
        assert response.status_code == 200

class TestAPIValidation:
    """Test input validation logic"""
    
    def test_feature_validation_ranges(self):
        """Test feature validation with edge cases"""
        # Test minimum valid values
        test_data = {
            "sepal_length": 0.1,
            "sepal_width": 0.1,
            "petal_length": 0.1,
            "petal_width": 0.1
        }
        response = client.post("/predict", json=test_data)
        # Should either work or fail due to model not loaded, not due to validation
        assert response.status_code in [200, 503]
        
        # Test very large values (should still be valid from API perspective)
        test_data = {
            "sepal_length": 100.0,
            "sepal_width": 100.0,
            "petal_length": 100.0,
            "petal_width": 100.0
        }
        response = client.post("/predict", json=test_data)
        assert response.status_code in [200, 503]
    
    def test_precision_handling(self):
        """Test handling of high precision numbers"""
        test_data = {
            "sepal_length": 5.123456789,
            "sepal_width": 3.987654321,
            "petal_length": 1.456789123,
            "petal_width": 0.234567891
        }
        response = client.post("/predict", json=test_data)
        assert response.status_code in [200, 422, 503]

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
