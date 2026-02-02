"""
ai-efficiency: Measure the energy efficiency of any AI model.

The simplest way to measure AI efficiency. One number. No excuses.

Example:
    >>> from ai_efficiency import measure
    >>> score = measure(model, test_data)
    >>> print(score.grade)  # A+, A, B, C, or D

For academic papers:
    >>> from ai_efficiency import measure_academic
    >>> metrics = measure_academic(model, test_data, test_labels)
    >>> print(metrics.to_latex_table())

For SCI compliance (Green Software Foundation):
    >>> from ai_efficiency import sci_report
    >>> sci = sci_report(model, test_data)
    >>> print(sci.sci)  # gCO2eq per 1000 inferences

For real-time carbon intensity:
    >>> from ai_efficiency import get_carbon_intensity
    >>> intensity = get_carbon_intensity("KR", real_time=True)
"""

from .measure import measure, EfficiencyScore
from .compare import compare
from .report import report
from .academic import measure_academic, compare_academic, AcademicMetrics
from .sci import sci_report, compare_sci, SCIScore
from .carbon import (
    get_carbon_intensity,
    get_carbon_data,
    get_cloud_carbon_factor,
    CarbonIntensityData,
    CarbonIntensityProvider,
)
from .constants import CARBON_INTENSITY, GRADES, HARDWARE_POWER

__version__ = "1.0.0"  # NeurIPS 2026 D&B Paper Release
__all__ = [
    # Core
    "measure",
    "compare",
    "report",
    "EfficiencyScore",
    # Academic (paper-ready metrics)
    "measure_academic",
    "compare_academic",
    "AcademicMetrics",
    # SCI for AI (Green Software Foundation compliance)
    "sci_report",
    "compare_sci",
    "SCIScore",
    # Carbon intensity
    "get_carbon_intensity",
    "get_carbon_data",
    "get_cloud_carbon_factor",
    "CarbonIntensityData",
    "CarbonIntensityProvider",
    # Constants
    "CARBON_INTENSITY",
    "GRADES",
    "HARDWARE_POWER",
]
