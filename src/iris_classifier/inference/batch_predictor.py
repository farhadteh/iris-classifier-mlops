"""
Batch prediction utilities for processing multiple instances
"""

import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from typing import Any, Dict, Iterator, List, Optional, Union

import numpy as np
import pandas as pd

from ..data.preprocessor import DataPreprocessor
from ..data.validator import DataValidator
from ..models.base import BaseModel
from .predictor import ModelPredictor

logger = logging.getLogger(__name__)


class BatchPredictor:
    """Batch prediction service for high-throughput processing"""

    def __init__(
        self,
        model: BaseModel,
        preprocessor: Optional[DataPreprocessor] = None,
        validator: Optional[DataValidator] = None,
        max_workers: int = 4,
    ):
        """
        Initialize batch predictor

        Args:
            model: Trained model instance
            preprocessor: Data preprocessor (optional)
            validator: Data validator (optional)
            max_workers: Maximum number of worker threads
        """
        self.model = model
        self.preprocessor = preprocessor
        self.validator = validator or DataValidator()
        self.max_workers = max_workers
        self.class_names = ["setosa", "versicolor", "virginica"]

        # Initialize single predictor for reuse
        self.single_predictor = ModelPredictor(model, preprocessor, validator)

        if not model.is_fitted:
            logger.warning("Model is not fitted. Batch predictions may fail.")

    def predict_batch(
        self,
        batch_data: List[Dict[str, float]],
        return_probabilities: bool = True,
        chunk_size: int = 100,
    ) -> Dict[str, Any]:
        """
        Make batch predictions efficiently

        Args:
            batch_data: List of dictionaries with feature values
            return_probabilities: Whether to return class probabilities
            chunk_size: Size of chunks for processing

        Returns:
            Batch prediction results
        """
        start_time = time.time()
        logger.info(f"Starting batch prediction for {len(batch_data)} instances")

        # Validate batch input
        is_valid, errors, invalid_indices = self.validator.validate_batch_input(
            batch_data
        )
        if not is_valid:
            raise ValueError(f"Batch validation failed: {errors}")

        try:
            # Convert to DataFrame for efficient processing
            batch_df = pd.DataFrame(batch_data)

            # Apply preprocessing if available
            if self.preprocessor:
                X_processed, _ = self.preprocessor.transform(batch_df)
                X_input = X_processed
            else:
                # Ensure correct feature order
                feature_order = [
                    "sepal_length",
                    "sepal_width",
                    "petal_length",
                    "petal_width",
                ]
                X_input = batch_df[feature_order].values

            # Make batch predictions
            predictions = self.model.predict(X_input)

            # Get probabilities if requested and available
            probabilities = None
            if return_probabilities:
                try:
                    probabilities = self.model.predict_proba(X_input)
                except Exception as e:
                    logger.warning(f"Could not get batch probabilities: {e}")

            # Format results
            results = []
            timestamp = datetime.now()

            for i, prediction in enumerate(predictions):
                class_name = self.class_names[int(prediction)]

                result = {
                    "prediction": int(prediction),
                    "class_name": class_name,
                    "timestamp": timestamp,
                    "input_features": batch_data[i],
                }

                if probabilities is not None:
                    probs = probabilities[i]
                    result["probabilities"] = {
                        self.class_names[j]: float(prob) for j, prob in enumerate(probs)
                    }
                    result["confidence"] = float(max(probs))
                else:
                    result["probabilities"] = {class_name: 1.0}
                    result["confidence"] = 1.0

                results.append(result)

            processing_time = time.time() - start_time

            batch_result = {
                "predictions": results,
                "batch_size": len(batch_data),
                "processing_time_ms": processing_time * 1000,
                "timestamp": timestamp,
                "success_rate": 1.0,  # All successful if we reach here
                "throughput_per_second": (
                    len(batch_data) / processing_time if processing_time > 0 else 0
                ),
            }

            logger.info(
                f"Batch prediction completed in {processing_time:.3f}s for {len(batch_data)} instances"
            )
            return batch_result

        except Exception as e:
            logger.error(f"Batch prediction failed: {e}")
            raise RuntimeError(f"Batch prediction failed: {str(e)}")

    def predict_batch_parallel(
        self, batch_data: List[Dict[str, float]], return_probabilities: bool = True
    ) -> Dict[str, Any]:
        """
        Make batch predictions using parallel processing

        Args:
            batch_data: List of dictionaries with feature values
            return_probabilities: Whether to return class probabilities

        Returns:
            Batch prediction results
        """
        start_time = time.time()
        logger.info(
            f"Starting parallel batch prediction for {len(batch_data)} instances"
        )

        # Validate batch input
        is_valid, errors, invalid_indices = self.validator.validate_batch_input(
            batch_data
        )
        if not is_valid:
            raise ValueError(f"Batch validation failed: {errors}")

        results = []
        failed_indices = []

        try:
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                # Submit prediction tasks
                future_to_index = {
                    executor.submit(
                        self.single_predictor.predict, instance, return_probabilities
                    ): i
                    for i, instance in enumerate(batch_data)
                }

                # Collect results
                for future in as_completed(future_to_index):
                    index = future_to_index[future]
                    try:
                        result = future.result()
                        results.append((index, result))
                    except Exception as e:
                        logger.error(f"Prediction failed for instance {index}: {e}")
                        failed_indices.append(index)

            # Sort results by original index
            results.sort(key=lambda x: x[0])
            sorted_results = [result[1] for result in results]

            processing_time = time.time() - start_time
            success_rate = len(sorted_results) / len(batch_data)

            batch_result = {
                "predictions": sorted_results,
                "batch_size": len(batch_data),
                "successful_predictions": len(sorted_results),
                "failed_predictions": len(failed_indices),
                "failed_indices": failed_indices,
                "processing_time_ms": processing_time * 1000,
                "timestamp": datetime.now(),
                "success_rate": success_rate,
                "throughput_per_second": (
                    len(sorted_results) / processing_time if processing_time > 0 else 0
                ),
            }

            logger.info(
                f"Parallel batch prediction completed: {len(sorted_results)}/{len(batch_data)} successful"
            )
            return batch_result

        except Exception as e:
            logger.error(f"Parallel batch prediction failed: {e}")
            raise RuntimeError(f"Parallel batch prediction failed: {str(e)}")

    def predict_from_file(
        self, file_path: str, output_path: Optional[str] = None, chunk_size: int = 1000
    ) -> Dict[str, Any]:
        """
        Make predictions from a CSV file

        Args:
            file_path: Path to input CSV file
            output_path: Path to save results (optional)
            chunk_size: Size of processing chunks

        Returns:
            Prediction summary
        """
        logger.info(f"Processing predictions from file: {file_path}")

        try:
            # Read input file
            df = pd.read_csv(file_path)
            logger.info(f"Loaded {len(df)} rows from {file_path}")

            # Convert to list of dictionaries
            batch_data = df.to_dict("records")

            # Process in chunks if large file
            if len(batch_data) > chunk_size:
                all_results = []
                total_processing_time = 0

                for i in range(0, len(batch_data), chunk_size):
                    chunk = batch_data[i : i + chunk_size]
                    logger.info(
                        f"Processing chunk {i//chunk_size + 1}/{(len(batch_data)-1)//chunk_size + 1}"
                    )

                    chunk_result = self.predict_batch(chunk)
                    all_results.extend(chunk_result["predictions"])
                    total_processing_time += chunk_result["processing_time_ms"]

                final_result = {
                    "predictions": all_results,
                    "batch_size": len(batch_data),
                    "processing_time_ms": total_processing_time,
                    "timestamp": datetime.now(),
                    "chunks_processed": (len(batch_data) - 1) // chunk_size + 1,
                }
            else:
                final_result = self.predict_batch(batch_data)

            # Save results if output path provided
            if output_path:
                self.save_results_to_file(final_result, output_path)

            return final_result

        except Exception as e:
            logger.error(f"File prediction failed: {e}")
            raise RuntimeError(f"File prediction failed: {str(e)}")

    def save_results_to_file(self, results: Dict[str, Any], output_path: str) -> None:
        """
        Save prediction results to a CSV file

        Args:
            results: Prediction results dictionary
            output_path: Output file path
        """
        logger.info(f"Saving results to: {output_path}")

        try:
            # Extract predictions and convert to DataFrame
            predictions = results["predictions"]

            rows = []
            for pred in predictions:
                row = pred["input_features"].copy()
                row["predicted_class"] = pred["class_name"]
                row["predicted_index"] = pred["prediction"]
                row["confidence"] = pred["confidence"]

                # Add probability columns
                if "probabilities" in pred:
                    for class_name, prob in pred["probabilities"].items():
                        row[f"prob_{class_name}"] = prob

                rows.append(row)

            # Create DataFrame and save
            results_df = pd.DataFrame(rows)
            results_df.to_csv(output_path, index=False)

            logger.info(f"Results saved successfully to {output_path}")

        except Exception as e:
            logger.error(f"Failed to save results: {e}")
            raise RuntimeError(f"Failed to save results: {str(e)}")

    def get_batch_statistics(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Calculate statistics for batch prediction results

        Args:
            results: Batch prediction results

        Returns:
            Statistical summary
        """
        predictions = results["predictions"]

        if not predictions:
            return {"error": "No predictions to analyze"}

        # Class distribution
        class_counts = {}
        confidence_scores = []

        for pred in predictions:
            class_name = pred["class_name"]
            class_counts[class_name] = class_counts.get(class_name, 0) + 1
            confidence_scores.append(pred["confidence"])

        # Calculate statistics
        stats = {
            "total_predictions": len(predictions),
            "class_distribution": class_counts,
            "class_percentages": {
                cls: (count / len(predictions)) * 100
                for cls, count in class_counts.items()
            },
            "confidence_statistics": {
                "mean": np.mean(confidence_scores),
                "std": np.std(confidence_scores),
                "min": np.min(confidence_scores),
                "max": np.max(confidence_scores),
                "median": np.median(confidence_scores),
            },
            "processing_info": {
                "batch_size": results.get("batch_size", 0),
                "processing_time_ms": results.get("processing_time_ms", 0),
                "throughput_per_second": results.get("throughput_per_second", 0),
            },
        }

        return stats

    def health_check(self) -> Dict[str, Any]:
        """Perform health check on the batch predictor"""
        try:
            # Test with small batch
            test_batch = [
                {
                    "sepal_length": 5.0,
                    "sepal_width": 3.0,
                    "petal_length": 1.5,
                    "petal_width": 0.2,
                }
            ]

            result = self.predict_batch(test_batch)

            return {
                "status": "healthy",
                "can_predict_batch": True,
                "max_workers": self.max_workers,
                "test_processing_time_ms": result.get("processing_time_ms", 0),
                "timestamp": datetime.now(),
            }
        except Exception as e:
            return {
                "status": "unhealthy",
                "can_predict_batch": False,
                "error": str(e),
                "timestamp": datetime.now(),
            }
