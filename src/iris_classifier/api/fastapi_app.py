import logging
import os
from datetime import datetime
from typing import Any, Dict, List

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Iris Classification API",
    description="MLflow-powered Iris classification model API",
    version="1.0.0",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Pydantic models for request/response
class IrisFeatures(BaseModel):
    sepal_length: float = Field(..., description="Sepal length in cm", ge=0)
    sepal_width: float = Field(..., description="Sepal width in cm", ge=0)
    petal_length: float = Field(..., description="Petal length in cm", ge=0)
    petal_width: float = Field(..., description="Petal width in cm", ge=0)


class BatchIrisFeatures(BaseModel):
    instances: List[IrisFeatures]


class PredictionResponse(BaseModel):
    prediction: int
    class_name: str
    confidence: float
    probabilities: Dict[str, float]
    timestamp: str


class BatchPredictionResponse(BaseModel):
    predictions: List[PredictionResponse]
    batch_size: int
    timestamp: str


class ModelInfo(BaseModel):
    model_name: str
    model_version: str
    model_stage: str
    mlflow_run_id: str
    model_uri: str


# Global variables for model
model = None
model_info = None
class_names = ["setosa", "versicolor", "virginica"]


def load_model():
    """Load the latest model from MLflow Model Registry"""
    global model, model_info

    try:
        # Set MLflow tracking URI
        mlflow_uri = os.getenv("MLFLOW_TRACKING_URI", "file:./mlruns")
        mlflow.set_tracking_uri(mlflow_uri)

        # Try to load from model registry first
        try:
            model_name = "tracking-quickstart"
            model_version = "latest"
            model_uri = f"models:/{model_name}/{model_version}"
            model = mlflow.pyfunc.load_model(model_uri)

            # Get model info
            client = mlflow.tracking.MlflowClient()
            model_version_details = client.get_latest_versions(
                model_name, stages=["None", "Staging", "Production"]
            )
            if model_version_details:
                latest_version = model_version_details[0]
                model_info = {
                    "model_name": model_name,
                    "model_version": latest_version.version,
                    "model_stage": latest_version.current_stage,
                    "mlflow_run_id": latest_version.run_id,
                    "model_uri": model_uri,
                }
            logger.info(f"Model loaded from registry: {model_uri}")

        except Exception as e:
            logger.warning(f"Could not load from model registry: {e}")
            # Fallback: load the latest model from runs
            experiment_name = "MLflow Quickstart"
            try:
                experiment = mlflow.get_experiment_by_name(experiment_name)
                if experiment:
                    runs = mlflow.search_runs(
                        experiment_ids=[experiment.experiment_id],
                        order_by=["start_time DESC"],
                        max_results=1,
                    )
                    if not runs.empty:
                        run_id = runs.iloc[0]["run_id"]
                        model_uri = f"runs:/{run_id}/iris_model"
                        model = mlflow.pyfunc.load_model(model_uri)
                        model_info = {
                            "model_name": "iris_model",
                            "model_version": "latest_run",
                            "model_stage": "None",
                            "mlflow_run_id": run_id,
                            "model_uri": model_uri,
                        }
                        logger.info(f"Model loaded from latest run: {model_uri}")
                    else:
                        raise Exception("No runs found in experiment")
                else:
                    raise Exception(f"Experiment '{experiment_name}' not found")
            except Exception as e2:
                logger.error(f"Could not load model from runs either: {e2}")
                raise Exception("Could not load any model")

    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise e


@app.on_event("startup")
async def startup_event():
    """Load model on startup"""
    try:
        load_model()
        logger.info("Model loaded successfully on startup")
    except Exception as e:
        logger.error(f"Failed to load model on startup: {e}")
        # Don't fail startup, but log the error


@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "message": "Iris Classification API is running!",
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
    }


@app.get("/health")
async def health_check():
    """Detailed health check"""
    model_status = "loaded" if model is not None else "not_loaded"
    return {
        "status": "healthy",
        "model_status": model_status,
        "timestamp": datetime.now().isoformat(),
    }


@app.get("/model/info", response_model=ModelInfo)
async def get_model_info():
    """Get information about the loaded model"""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    if model_info is None:
        raise HTTPException(status_code=503, detail="Model info not available")

    return ModelInfo(**model_info)


@app.post("/predict", response_model=PredictionResponse)
async def predict(features: IrisFeatures):
    """Make a single prediction"""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        # Convert to DataFrame
        input_data = pd.DataFrame(
            [
                [
                    features.sepal_length,
                    features.sepal_width,
                    features.petal_length,
                    features.petal_width,
                ]
            ],
            columns=["sepal_length", "sepal_width", "petal_length", "petal_width"],
        )

        # Make prediction
        prediction = model.predict(input_data)[0]

        # Get probabilities if available
        try:
            probabilities = model.predict_proba(input_data)[0]
            prob_dict = {
                class_names[i]: float(prob) for i, prob in enumerate(probabilities)
            }
            confidence = float(max(probabilities))
        except:
            # Fallback if predict_proba is not available
            prob_dict = {class_names[int(prediction)]: 1.0}
            confidence = 1.0

        return PredictionResponse(
            prediction=int(prediction),
            class_name=class_names[int(prediction)],
            confidence=confidence,
            probabilities=prob_dict,
            timestamp=datetime.now().isoformat(),
        )

    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.post("/predict/batch", response_model=BatchPredictionResponse)
async def predict_batch(batch_features: BatchIrisFeatures):
    """Make batch predictions"""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        # Convert to DataFrame
        input_data = pd.DataFrame(
            [
                [f.sepal_length, f.sepal_width, f.petal_length, f.petal_width]
                for f in batch_features.instances
            ],
            columns=["sepal_length", "sepal_width", "petal_length", "petal_width"],
        )

        # Make predictions
        predictions = model.predict(input_data)

        # Get probabilities if available
        try:
            probabilities = model.predict_proba(input_data)
            prob_available = True
        except:
            prob_available = False

        # Format results
        results = []
        timestamp = datetime.now().isoformat()

        for i, pred in enumerate(predictions):
            if prob_available:
                probs = probabilities[i]
                prob_dict = {
                    class_names[j]: float(prob) for j, prob in enumerate(probs)
                }
                confidence = float(max(probs))
            else:
                prob_dict = {class_names[int(pred)]: 1.0}
                confidence = 1.0

            results.append(
                PredictionResponse(
                    prediction=int(pred),
                    class_name=class_names[int(pred)],
                    confidence=confidence,
                    probabilities=prob_dict,
                    timestamp=timestamp,
                )
            )

        return BatchPredictionResponse(
            predictions=results,
            batch_size=len(batch_features.instances),
            timestamp=timestamp,
        )

    except Exception as e:
        logger.error(f"Batch prediction error: {e}")
        raise HTTPException(
            status_code=500, detail=f"Batch prediction failed: {str(e)}"
        )


@app.post("/model/reload")
async def reload_model():
    """Reload the model from MLflow"""
    try:
        load_model()
        return {
            "message": "Model reloaded successfully",
            "timestamp": datetime.now().isoformat(),
            "model_info": model_info,
        }
    except Exception as e:
        logger.error(f"Model reload error: {e}")
        raise HTTPException(status_code=500, detail=f"Model reload failed: {str(e)}")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
