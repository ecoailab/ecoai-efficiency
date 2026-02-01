"""
Core measurement functionality for AI efficiency.
"""

import time
from dataclasses import dataclass
from typing import Any, Optional, Callable, Union
import warnings

from .constants import CARBON_INTENSITY, GRADES, HARDWARE_POWER, DEFAULT_REGION, DEFAULT_HARDWARE


@dataclass
class EfficiencyScore:
    """Result of an efficiency measurement."""

    # Core metrics
    efficiency: float          # Accuracy / kWh (higher is better)
    accuracy: float            # Model accuracy or quality metric (0-100)
    kwh_per_1k: float         # kWh per 1000 queries
    co2_per_1k: float         # grams CO2 per 1000 queries
    grade: str                 # A+, A, B, C, D

    # Details
    total_queries: int
    total_time_seconds: float
    total_kwh: float
    total_co2_grams: float

    # Configuration
    hardware: str
    region: str

    def __str__(self) -> str:
        return (
            f"AI Efficiency Report\n"
            f"{'='*40}\n"
            f"Efficiency Score: {self.efficiency:,.0f}\n"
            f"Grade: {self.grade}\n"
            f"\n"
            f"Performance:\n"
            f"  Accuracy: {self.accuracy:.1f}%\n"
            f"  Queries: {self.total_queries:,}\n"
            f"  Time: {self.total_time_seconds:.2f}s\n"
            f"\n"
            f"Energy (per 1,000 queries):\n"
            f"  Energy: {self.kwh_per_1k:.6f} kWh\n"
            f"  Carbon: {self.co2_per_1k:.2f}g CO2\n"
            f"\n"
            f"Total:\n"
            f"  Energy: {self.total_kwh:.6f} kWh\n"
            f"  Carbon: {self.total_co2_grams:.2f}g CO2\n"
            f"\n"
            f"Hardware: {self.hardware}\n"
            f"Region: {self.region} ({CARBON_INTENSITY.get(self.region, 450)}g CO2/kWh)\n"
        )

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON export."""
        return {
            "efficiency_score": self.efficiency,
            "grade": self.grade,
            "accuracy": self.accuracy,
            "kwh_per_1k_queries": self.kwh_per_1k,
            "co2_grams_per_1k_queries": self.co2_per_1k,
            "total_queries": self.total_queries,
            "total_time_seconds": self.total_time_seconds,
            "total_kwh": self.total_kwh,
            "total_co2_grams": self.total_co2_grams,
            "hardware": self.hardware,
            "region": self.region,
        }


def _get_grade(efficiency: float) -> str:
    """Determine grade based on efficiency score."""
    for grade, threshold in GRADES.items():
        if efficiency >= threshold:
            return grade
    return "D"


def _detect_hardware() -> str:
    """Attempt to detect available hardware."""
    try:
        import torch
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            # Map to known hardware
            if "A100" in gpu_name:
                return "A100"
            elif "H100" in gpu_name:
                return "H100"
            elif "V100" in gpu_name:
                return "V100"
            elif "4090" in gpu_name:
                return "RTX4090"
            elif "3090" in gpu_name:
                return "RTX3090"
            elif "T4" in gpu_name:
                return "T4"
    except ImportError:
        pass

    try:
        import platform
        if platform.processor().startswith("Apple"):
            return "M1"  # Simplified
    except:
        pass

    return DEFAULT_HARDWARE


def _measure_power(hardware: str) -> float:
    """Get power consumption in Watts for hardware."""
    return HARDWARE_POWER.get(hardware, HARDWARE_POWER["CPU"])


def _get_predictions(model: Any, data: Any, n_samples: int) -> tuple:
    """Get predictions from model, handling different frameworks."""

    # Determine how to call the model
    if hasattr(model, "predict"):
        predict_fn = model.predict
    elif hasattr(model, "__call__"):
        predict_fn = model.__call__
    elif hasattr(model, "forward"):
        predict_fn = model.forward
    else:
        raise ValueError("Model must have predict(), __call__(), or forward() method")

    # Handle different data types
    if hasattr(data, "__len__"):
        n_samples = min(n_samples, len(data))
        sample_data = data[:n_samples]
    else:
        sample_data = data

    # Measure time
    start_time = time.perf_counter()
    predictions = predict_fn(sample_data)
    end_time = time.perf_counter()

    return predictions, end_time - start_time, n_samples


def _calculate_accuracy(predictions: Any, labels: Optional[Any]) -> float:
    """Calculate accuracy or return default."""
    if labels is None:
        # No labels provided, assume quality metric of 100 (just measuring efficiency)
        return 100.0

    try:
        import numpy as np
        predictions = np.array(predictions)
        labels = np.array(labels)

        # Handle different prediction formats
        if predictions.ndim > 1:
            predictions = predictions.argmax(axis=-1)

        accuracy = (predictions == labels).mean() * 100
        return float(accuracy)
    except Exception as e:
        warnings.warn(f"Could not calculate accuracy: {e}. Using default 100%.")
        return 100.0


def measure(
    model: Any,
    data: Any,
    labels: Optional[Any] = None,
    n_samples: int = 1000,
    hardware: Optional[str] = None,
    region: str = DEFAULT_REGION,
    accuracy_metric: Optional[Callable] = None,
) -> EfficiencyScore:
    """
    Measure the energy efficiency of an AI model.

    Args:
        model: Any model with predict(), __call__(), or forward() method
        data: Input data for the model
        labels: Optional ground truth labels for accuracy calculation
        n_samples: Number of samples to use for measurement
        hardware: Hardware type (auto-detected if None)
        region: Region code for carbon intensity (e.g., "KR", "US", "EU")
        accuracy_metric: Optional custom accuracy function(predictions, labels) -> float

    Returns:
        EfficiencyScore with all metrics

    Example:
        >>> from ai_efficiency import measure
        >>> score = measure(model, test_data, test_labels)
        >>> print(score)
        >>> print(f"Grade: {score.grade}")
    """

    # Detect hardware if not specified
    if hardware is None:
        hardware = _detect_hardware()

    # Get power consumption
    power_watts = _measure_power(hardware)

    # Run predictions and measure time
    predictions, elapsed_time, actual_samples = _get_predictions(model, data, n_samples)

    # Calculate accuracy
    if accuracy_metric is not None:
        accuracy = accuracy_metric(predictions, labels)
    else:
        accuracy = _calculate_accuracy(predictions, labels)

    # Calculate energy consumption
    # Energy (kWh) = Power (W) × Time (h) / 1000
    total_kwh = (power_watts * elapsed_time / 3600) / 1000

    # Scale to per-1000 queries
    kwh_per_1k = (total_kwh / actual_samples) * 1000

    # Calculate carbon emissions
    carbon_intensity = CARBON_INTENSITY.get(region, CARBON_INTENSITY["WORLD"])
    total_co2 = total_kwh * carbon_intensity
    co2_per_1k = kwh_per_1k * carbon_intensity

    # Calculate efficiency score
    # Efficiency = Accuracy / kWh (per 1000 queries)
    if kwh_per_1k > 0:
        efficiency = accuracy / kwh_per_1k
    else:
        efficiency = float("inf")

    # Determine grade
    grade = _get_grade(efficiency)

    return EfficiencyScore(
        efficiency=efficiency,
        accuracy=accuracy,
        kwh_per_1k=kwh_per_1k,
        co2_per_1k=co2_per_1k,
        grade=grade,
        total_queries=actual_samples,
        total_time_seconds=elapsed_time,
        total_kwh=total_kwh,
        total_co2_grams=total_co2,
        hardware=hardware,
        region=region,
    )
