"""
API schemas and data models for FastAPI
"""

from pydantic import BaseModel, Field, validator
from typing import List, Dict, Optional, Union
from datetime import datetime
from enum import Enum


class ModelStage(str, Enum):
    """Model stages in MLflow registry"""
    NONE = "None"
    STAGING = "Staging"
    PRODUCTION = "Production"
    ARCHIVED = "Archived"


class IrisClass(str, Enum):
    """Iris flower classes"""
    SETOSA = "setosa"
    VERSICOLOR = "versicolor"
    VIRGINICA = "virginica"


class IrisFeatures(BaseModel):
    """Input features for Iris prediction"""
    sepal_length: float = Field(..., ge=0.1, le=10.0, description="Sepal length in cm")
    sepal_width: float = Field(..., ge=0.1, le=10.0, description="Sepal width in cm")
    petal_length: float = Field(..., ge=0.1, le=10.0, description="Petal length in cm")
    petal_width: float = Field(..., ge=0.1, le=10.0, description="Petal width in cm")
    
    @validator('*', pre=True)
    def validate_positive(cls, v):
        """Ensure all values are positive"""
        if v <= 0:
            raise ValueError('All measurements must be positive')
        return v

    class Config:
        schema_extra = {
            "example": {
                "sepal_length": 5.1,
                "sepal_width": 3.5,
                "petal_length": 1.4,
                "petal_width": 0.2
            }
        }


class BatchIrisFeatures(BaseModel):
    """Batch input for multiple predictions"""
    instances: List[IrisFeatures] = Field(..., min_items=1, max_items=1000)
    
    class Config:
        schema_extra = {
            "example": {
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
        }


class PredictionResponse(BaseModel):
    """Single prediction response"""
    prediction: int = Field(..., ge=0, le=2, description="Predicted class index")
    class_name: IrisClass = Field(..., description="Predicted class name")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Prediction confidence")
    probabilities: Dict[str, float] = Field(..., description="Class probabilities")
    timestamp: datetime = Field(..., description="Prediction timestamp")
    
    class Config:
        schema_extra = {
            "example": {
                "prediction": 0,
                "class_name": "setosa",
                "confidence": 0.95,
                "probabilities": {
                    "setosa": 0.95,
                    "versicolor": 0.03,
                    "virginica": 0.02
                },
                "timestamp": "2023-12-01T10:30:00"
            }
        }


class BatchPredictionResponse(BaseModel):
    """Batch prediction response"""
    predictions: List[PredictionResponse] = Field(..., description="List of predictions")
    batch_size: int = Field(..., ge=1, description="Number of predictions")
    processing_time_ms: Optional[float] = Field(None, description="Processing time in milliseconds")
    timestamp: datetime = Field(..., description="Batch processing timestamp")


class HealthResponse(BaseModel):
    """Health check response"""
    status: str = Field(..., description="Service status")
    model_status: str = Field(..., description="Model loading status")
    timestamp: datetime = Field(..., description="Health check timestamp")
    version: Optional[str] = Field(None, description="API version")
    uptime_seconds: Optional[float] = Field(None, description="Service uptime in seconds")


class ModelInfo(BaseModel):
    """Model information response"""
    model_name: str = Field(..., description="Model name")
    model_version: str = Field(..., description="Model version")
    model_stage: ModelStage = Field(..., description="Model stage")
    mlflow_run_id: str = Field(..., description="MLflow run ID")
    model_uri: str = Field(..., description="Model URI")
    model_type: Optional[str] = Field(None, description="Model algorithm type")
    training_timestamp: Optional[datetime] = Field(None, description="When model was trained")
    performance_metrics: Optional[Dict[str, float]] = Field(None, description="Model performance metrics")


class ErrorResponse(BaseModel):
    """Error response schema"""
    error: str = Field(..., description="Error type")
    message: str = Field(..., description="Error message")
    details: Optional[Dict[str, any]] = Field(None, description="Additional error details")
    timestamp: datetime = Field(..., description="Error timestamp")
    request_id: Optional[str] = Field(None, description="Request ID for tracking")


class ValidationError(BaseModel):
    """Validation error details"""
    field: str = Field(..., description="Field that failed validation")
    message: str = Field(..., description="Validation error message")
    invalid_value: Union[str, float, int, None] = Field(..., description="The invalid value")


class ValidationErrorResponse(ErrorResponse):
    """Validation error response with field details"""
    validation_errors: List[ValidationError] = Field(..., description="List of validation errors")


class ModelReloadRequest(BaseModel):
    """Request to reload model"""
    model_name: Optional[str] = Field(None, description="Specific model to load")
    model_version: Optional[str] = Field(None, description="Specific version to load")
    model_stage: Optional[ModelStage] = Field(ModelStage.PRODUCTION, description="Model stage to load")


class ModelReloadResponse(BaseModel):
    """Response after model reload"""
    success: bool = Field(..., description="Whether reload was successful")
    message: str = Field(..., description="Reload status message")
    model_info: Optional[ModelInfo] = Field(None, description="New model information")
    timestamp: datetime = Field(..., description="Reload timestamp")


class MetricsResponse(BaseModel):
    """API metrics response"""
    total_requests: int = Field(..., description="Total number of requests")
    successful_requests: int = Field(..., description="Number of successful requests")
    failed_requests: int = Field(..., description="Number of failed requests")
    average_response_time_ms: float = Field(..., description="Average response time")
    requests_per_minute: float = Field(..., description="Current requests per minute")
    model_version: str = Field(..., description="Current model version")
    uptime_seconds: float = Field(..., description="Service uptime")


# Request/Response type aliases for documentation
IrisInputType = Union[IrisFeatures, BatchIrisFeatures]
IrisOutputType = Union[PredictionResponse, BatchPredictionResponse]
