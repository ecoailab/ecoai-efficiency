"""
SCI for AI: Software Carbon Intensity for AI workloads.

Implements the Green Software Foundation's SCI specification adapted for AI.
https://sci-for-ai.greensoftware.foundation/

SCI = ((E × I) + M) per R

Where:
- E = Energy consumed (kWh)
- I = Carbon intensity of electricity (gCO2eq/kWh)
- M = Embodied emissions from hardware (gCO2eq)
- R = Functional unit (e.g., per 1000 inferences)

References:
- Green Software Foundation SCI: https://greensoftware.foundation/articles/software-carbon-intensity
- SCI for AI Specification: https://sci-for-ai.greensoftware.foundation/
- Embodied Carbon: https://github.com/Green-Software-Foundation/sci/blob/main/Software_Carbon_Intensity/Software_Carbon_Intensity_Specification.md
"""

import time
from dataclasses import dataclass, field
from typing import Any, Optional, Dict, List
from datetime import datetime, timezone

from .constants import CARBON_INTENSITY, HARDWARE_POWER
from .carbon import get_carbon_intensity, get_carbon_data, CarbonIntensityData


# Embodied carbon per hardware type (gCO2eq per hour of use)
# Based on hardware lifecycle (typically 3-5 years for servers)
# Source: Boavizta, Dell lifecycle studies, NVIDIA sustainability reports
EMBODIED_CARBON = {
    # NVIDIA GPUs (per hour, assuming 4-year lifecycle)
    "A100": 15.0,    # ~525 kg manufacturing / 35040 hours
    "H100": 20.0,    # Higher due to larger die, advanced packaging
    "V100": 12.0,
    "RTX4090": 8.0,
    "RTX3090": 7.0,
    "RTX3080": 6.0,
    "T4": 4.0,

    # AMD GPUs
    "MI300X": 22.0,
    "MI250X": 16.0,

    # Apple Silicon (lower embodied due to integrated design)
    "M1": 2.0,
    "M2": 2.2,
    "M3": 2.5,
    "M1-Pro": 3.0,
    "M1-Max": 4.0,
    "M2-Ultra": 5.0,

    # CPU
    "CPU": 3.0,
    "CPU-server": 8.0,

    # TPU
    "TPUv4": 12.0,
    "TPUv5": 14.0,
}


@dataclass
class SCIScore:
    """
    SCI (Software Carbon Intensity) score for AI workloads.

    Following Green Software Foundation specification.
    """

    # Core SCI components
    sci: float                      # Final SCI score (gCO2eq per functional unit)
    energy_kwh: float               # E: Energy consumed
    carbon_intensity: float         # I: Grid carbon intensity (gCO2eq/kWh)
    embodied_carbon: float          # M: Embodied emissions (gCO2eq)
    functional_unit: str            # R: What we're measuring per

    # Breakdown
    operational_carbon: float       # E × I (gCO2eq)
    total_carbon: float            # (E × I) + M

    # Metadata
    hardware: str
    region: str
    n_samples: int
    measurement_time_s: float
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    # Optional accuracy for GES comparison
    accuracy: Optional[float] = None
    ges: Optional[float] = None     # Green Efficiency Score for comparison

    def __str__(self) -> str:
        return f"""
SCI for AI Report
{'='*50}
SCI Score: {self.sci:.4f} gCO2eq/{self.functional_unit}
{'='*50}

Components:
  E (Energy):           {self.energy_kwh:.6f} kWh
  I (Carbon Intensity): {self.carbon_intensity:.0f} gCO2eq/kWh
  M (Embodied):         {self.embodied_carbon:.4f} gCO2eq
  R (Functional Unit):  {self.functional_unit}

Carbon Breakdown:
  Operational (E × I):  {self.operational_carbon:.4f} gCO2eq
  Embodied (M):         {self.embodied_carbon:.4f} gCO2eq
  Total:                {self.total_carbon:.4f} gCO2eq

Configuration:
  Hardware: {self.hardware}
  Region: {self.region}
  Samples: {self.n_samples:,}
  Time: {self.measurement_time_s:.2f}s
"""

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON export."""
        return {
            "sci_score": self.sci,
            "components": {
                "E_energy_kwh": self.energy_kwh,
                "I_carbon_intensity_gco2_kwh": self.carbon_intensity,
                "M_embodied_gco2": self.embodied_carbon,
                "R_functional_unit": self.functional_unit,
            },
            "carbon": {
                "operational_gco2": self.operational_carbon,
                "embodied_gco2": self.embodied_carbon,
                "total_gco2": self.total_carbon,
            },
            "metadata": {
                "hardware": self.hardware,
                "region": self.region,
                "n_samples": self.n_samples,
                "measurement_time_s": self.measurement_time_s,
                "timestamp": self.timestamp,
            },
            "comparison": {
                "accuracy": self.accuracy,
                "ges": self.ges,
            },
        }

    def to_sci_report(self, model_name: str = "model") -> str:
        """
        Generate SCI-compliant report text.

        Format follows Green Software Foundation reporting guidelines.
        """
        return f"""# SCI for AI Report

**Model**: {model_name}
**Generated**: {self.timestamp}
**Standard**: SCI for AI v1.0

## SCI Score

**{self.sci:.4f} gCO2eq/{self.functional_unit}**

## Components (SCI = ((E × I) + M) / R)

| Component | Description | Value |
|-----------|-------------|-------|
| **E** | Energy consumed | {self.energy_kwh:.6f} kWh |
| **I** | Carbon intensity | {self.carbon_intensity:.0f} gCO2eq/kWh |
| **M** | Embodied emissions | {self.embodied_carbon:.4f} gCO2eq |
| **R** | Functional unit | {self.functional_unit} |

## Carbon Breakdown

| Category | Emissions |
|----------|-----------|
| Operational (E × I) | {self.operational_carbon:.4f} gCO2eq |
| Embodied (M) | {self.embodied_carbon:.4f} gCO2eq |
| **Total** | **{self.total_carbon:.4f} gCO2eq** |

## Methodology

This report follows the [SCI for AI specification](https://sci-for-ai.greensoftware.foundation/)
from the Green Software Foundation.

- Energy measured via power estimation (TDP-based)
- Carbon intensity: {self.region} grid ({self.carbon_intensity:.0f} gCO2eq/kWh)
- Embodied carbon: Hardware lifecycle allocation ({self.hardware})

---
*Generated by ai-efficiency (https://github.com/ecoailab/ai-efficiency)*
"""


def sci_report(
    model: Any,
    data: Any,
    labels: Optional[Any] = None,
    n_samples: int = 1000,
    hardware: Optional[str] = None,
    region: str = "WORLD",
    functional_unit: str = "1000 inferences",
    include_embodied: bool = True,
    real_time_carbon: bool = False,
) -> SCIScore:
    """
    Calculate SCI score for an AI model.

    Args:
        model: Model with predict(), __call__(), or forward() method
        data: Input data for the model
        labels: Optional ground truth labels
        n_samples: Number of samples to measure
        hardware: Hardware type (auto-detected if None)
        region: Region code for carbon intensity
        functional_unit: What to measure per (default: "1000 inferences")
        include_embodied: Include embodied carbon (M component)
        real_time_carbon: Use real-time carbon intensity from API

    Returns:
        SCIScore with full SCI breakdown

    Example:
        >>> from ai_efficiency import sci_report
        >>> sci = sci_report(model, test_data)
        >>> print(sci)
        >>> print(f"SCI: {sci.sci:.4f} gCO2eq/1000 inferences")
    """
    from .measure import _detect_hardware, _measure_power, _get_predictions, _calculate_accuracy

    # Detect hardware
    if hardware is None:
        hardware = _detect_hardware()

    # Get power consumption
    power_watts = _measure_power(hardware)

    # Measure predictions
    predictions, elapsed_time, actual_samples = _get_predictions(model, data, n_samples)
    accuracy = _calculate_accuracy(predictions, labels)

    # E: Energy consumed (kWh)
    total_kwh = (power_watts * elapsed_time / 3600) / 1000

    # I: Carbon intensity (gCO2eq/kWh)
    if real_time_carbon:
        carbon_data = get_carbon_data(region, real_time=True)
        carbon_intensity = carbon_data.intensity
    else:
        carbon_intensity = CARBON_INTENSITY.get(region, CARBON_INTENSITY["WORLD"])

    # M: Embodied carbon (gCO2eq) - allocated per hour of use
    if include_embodied:
        embodied_per_hour = EMBODIED_CARBON.get(hardware, EMBODIED_CARBON["CPU"])
        # Allocate based on actual usage time (in hours)
        embodied_carbon = embodied_per_hour * (elapsed_time / 3600)
    else:
        embodied_carbon = 0.0

    # Calculate operational carbon: E × I
    operational_carbon = total_kwh * carbon_intensity

    # Total carbon: (E × I) + M
    total_carbon = operational_carbon + embodied_carbon

    # SCI = Total carbon per functional unit
    # For "1000 inferences", scale to per-1000
    scale_factor = 1000 / actual_samples
    sci = total_carbon * scale_factor

    # Calculate GES for comparison
    kwh_per_1k = (total_kwh / actual_samples) * 1000
    ges = accuracy / kwh_per_1k if kwh_per_1k > 0 else float("inf")

    return SCIScore(
        sci=sci,
        energy_kwh=total_kwh,
        carbon_intensity=carbon_intensity,
        embodied_carbon=embodied_carbon,
        functional_unit=functional_unit,
        operational_carbon=operational_carbon,
        total_carbon=total_carbon,
        hardware=hardware,
        region=region,
        n_samples=actual_samples,
        measurement_time_s=elapsed_time,
        accuracy=accuracy,
        ges=ges,
    )


def compare_sci(
    models: List[Any],
    model_names: List[str],
    data: Any,
    labels: Optional[Any] = None,
    n_samples: int = 1000,
    region: str = "WORLD",
) -> str:
    """
    Compare SCI scores across multiple models.

    Returns a markdown comparison table.
    """
    results = []
    for model, name in zip(models, model_names):
        sci = sci_report(model, data, labels, n_samples, region=region)
        results.append((name, sci))

    # Sort by SCI (lowest is best)
    results.sort(key=lambda x: x[1].sci)

    table = """# SCI Comparison

| Model | SCI (gCO2eq/1K) | Energy (kWh) | Operational | Embodied | Accuracy |
|-------|-----------------|--------------|-------------|----------|----------|
"""
    for name, s in results:
        acc_str = f"{s.accuracy:.1f}%" if s.accuracy else "N/A"
        table += f"| {name} | {s.sci:.4f} | {s.energy_kwh:.6f} | {s.operational_carbon:.4f} | {s.embodied_carbon:.4f} | {acc_str} |\n"

    table += f"\n**Most Efficient**: {results[0][0]} ({results[0][1].sci:.4f} gCO2eq/1K inferences)"

    return table
