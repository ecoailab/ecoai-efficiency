"""
Academic metrics for research papers.

Based on literature review of Green AI papers from NeurIPS, ICML, ICLR, ACL.
References:
- Strubell et al. (2019) "Energy and Policy Considerations for Deep Learning in NLP"
- Schwartz et al. (2020) "Green AI"
- Patterson et al. (2021) "Carbon Emissions and Large Neural Network Training"
- Luccioni et al. (2023) "Estimating the Carbon Footprint of BLOOM"
"""

import time
from dataclasses import dataclass, field
from typing import Any, Optional, List, Dict
import warnings

from .constants import CARBON_INTENSITY, HARDWARE_POWER


@dataclass
class AcademicMetrics:
    """
    Comprehensive metrics for academic papers on AI efficiency.

    These metrics follow conventions from Green AI literature and can be
    directly reported in research papers.
    """

    # === Energy Metrics ===
    energy_kwh: float                    # Total energy in kWh
    energy_per_sample_wh: float          # Wh per inference sample
    energy_per_1k_samples_kwh: float     # kWh per 1000 samples

    # === Carbon Metrics ===
    carbon_kg: float                     # Total CO2 in kg
    carbon_per_sample_g: float           # grams CO2 per sample
    carbon_per_1k_samples_kg: float      # kg CO2 per 1000 samples

    # === Performance Metrics ===
    accuracy: float                      # Model accuracy (0-100)
    throughput_samples_per_sec: float    # Inference throughput
    latency_ms: float                    # Average latency per sample

    # === Efficiency Ratios (Novel Metrics) ===
    ges: float                           # Green Efficiency Score (Accuracy/kWh)
    perf_per_watt: float                 # Performance per Watt
    accuracy_per_carbon: float           # Accuracy per kg CO2

    # === FLOPs Metrics ===
    flops_per_sample: Optional[float] = None
    energy_per_flop: Optional[float] = None

    # === Comparison Baselines ===
    relative_efficiency: Optional[float] = None  # vs baseline model
    carbon_saved_vs_baseline: Optional[float] = None

    # === Metadata ===
    hardware: str = ""
    region: str = ""
    n_samples: int = 0
    measurement_time_s: float = 0.0

    def to_latex_table(self, caption: str = "AI Efficiency Metrics") -> str:
        """Generate LaTeX table for paper."""
        return f"""
\\begin{{table}}[h]
\\centering
\\caption{{{caption}}}
\\label{{tab:efficiency}}
\\begin{{tabular}}{{lrr}}
\\toprule
\\textbf{{Metric}} & \\textbf{{Value}} & \\textbf{{Unit}} \\\\
\\midrule
\\multicolumn{{3}}{{l}}{{\\textit{{Energy Consumption}}}} \\\\
Total Energy & {self.energy_kwh:.4f} & kWh \\\\
Energy per Sample & {self.energy_per_sample_wh:.4f} & Wh \\\\
Energy per 1K Samples & {self.energy_per_1k_samples_kwh:.6f} & kWh \\\\
\\midrule
\\multicolumn{{3}}{{l}}{{\\textit{{Carbon Emissions}}}} \\\\
Total Carbon & {self.carbon_kg:.4f} & kg CO$_2$ \\\\
Carbon per Sample & {self.carbon_per_sample_g:.4f} & g CO$_2$ \\\\
Carbon per 1K Samples & {self.carbon_per_1k_samples_kg:.6f} & kg CO$_2$ \\\\
\\midrule
\\multicolumn{{3}}{{l}}{{\\textit{{Performance}}}} \\\\
Accuracy & {self.accuracy:.2f} & \\% \\\\
Throughput & {self.throughput_samples_per_sec:.1f} & samples/s \\\\
Latency & {self.latency_ms:.2f} & ms \\\\
\\midrule
\\multicolumn{{3}}{{l}}{{\\textit{{Efficiency Ratios}}}} \\\\
GES (Green Efficiency Score) & {self.ges:,.0f} & Acc/kWh \\\\
Performance per Watt & {self.perf_per_watt:.2f} & Acc/W \\\\
Accuracy per Carbon & {self.accuracy_per_carbon:.2f} & Acc/kgCO$_2$ \\\\
\\bottomrule
\\end{{tabular}}
\\end{{table}}
"""

    def to_markdown_table(self) -> str:
        """Generate Markdown table for README or reports."""
        return f"""
| Metric | Value | Unit |
|--------|------:|------|
| **Energy** | | |
| Total Energy | {self.energy_kwh:.4f} | kWh |
| Energy per Sample | {self.energy_per_sample_wh:.4f} | Wh |
| Energy per 1K Samples | {self.energy_per_1k_samples_kwh:.6f} | kWh |
| **Carbon** | | |
| Total Carbon | {self.carbon_kg:.4f} | kg CO2 |
| Carbon per Sample | {self.carbon_per_sample_g:.4f} | g CO2 |
| **Performance** | | |
| Accuracy | {self.accuracy:.2f} | % |
| Throughput | {self.throughput_samples_per_sec:.1f} | samples/s |
| Latency | {self.latency_ms:.2f} | ms |
| **Efficiency** | | |
| GES (Green Efficiency Score) | {self.ges:,.0f} | Acc/kWh |
| Performance per Watt | {self.perf_per_watt:.2f} | Acc/W |
"""

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON/CSV export."""
        return {
            "energy_kwh": self.energy_kwh,
            "energy_per_sample_wh": self.energy_per_sample_wh,
            "energy_per_1k_samples_kwh": self.energy_per_1k_samples_kwh,
            "carbon_kg": self.carbon_kg,
            "carbon_per_sample_g": self.carbon_per_sample_g,
            "carbon_per_1k_samples_kg": self.carbon_per_1k_samples_kg,
            "accuracy": self.accuracy,
            "throughput_samples_per_sec": self.throughput_samples_per_sec,
            "latency_ms": self.latency_ms,
            "ges": self.ges,
            "perf_per_watt": self.perf_per_watt,
            "accuracy_per_carbon": self.accuracy_per_carbon,
            "flops_per_sample": self.flops_per_sample,
            "energy_per_flop": self.energy_per_flop,
            "hardware": self.hardware,
            "region": self.region,
            "n_samples": self.n_samples,
        }


def measure_academic(
    model: Any,
    data: Any,
    labels: Optional[Any] = None,
    n_samples: int = 1000,
    hardware: Optional[str] = None,
    region: str = "WORLD",
    flops_per_sample: Optional[float] = None,
    baseline_energy_kwh: Optional[float] = None,
) -> AcademicMetrics:
    """
    Measure comprehensive academic metrics for a model.

    This function provides all metrics needed for a research paper on
    AI efficiency, following conventions from Green AI literature.

    Args:
        model: Model with predict() or __call__() method
        data: Input data
        labels: Ground truth labels (optional)
        n_samples: Number of samples to measure
        hardware: Hardware type (auto-detected if None)
        region: Region code for carbon intensity
        flops_per_sample: FLOPs per inference (optional, for FLOP metrics)
        baseline_energy_kwh: Baseline model energy for comparison

    Returns:
        AcademicMetrics with all paper-ready metrics

    Example:
        >>> metrics = measure_academic(model, test_data, test_labels)
        >>> print(metrics.to_latex_table())
        >>> metrics_dict = metrics.to_dict()
    """
    from .measure import _detect_hardware, _measure_power, _get_predictions, _calculate_accuracy

    # Detect hardware
    if hardware is None:
        hardware = _detect_hardware()

    power_watts = _measure_power(hardware)

    # Measure
    predictions, elapsed_time, actual_samples = _get_predictions(model, data, n_samples)
    accuracy = _calculate_accuracy(predictions, labels)

    # Energy calculations
    total_kwh = (power_watts * elapsed_time / 3600) / 1000
    energy_per_sample_wh = (total_kwh * 1000) / actual_samples
    energy_per_1k_kwh = (total_kwh / actual_samples) * 1000

    # Carbon calculations
    carbon_intensity = CARBON_INTENSITY.get(region, 450)
    carbon_kg = total_kwh * carbon_intensity / 1000
    carbon_per_sample_g = (carbon_kg * 1000) / actual_samples
    carbon_per_1k_kg = (carbon_kg / actual_samples) * 1000

    # Performance metrics
    throughput = actual_samples / elapsed_time
    latency_ms = (elapsed_time / actual_samples) * 1000

    # Efficiency ratios
    ges = accuracy / energy_per_1k_kwh if energy_per_1k_kwh > 0 else float("inf")
    perf_per_watt = accuracy / power_watts
    acc_per_carbon = accuracy / carbon_kg if carbon_kg > 0 else float("inf")

    # FLOPs metrics
    energy_per_flop = None
    if flops_per_sample is not None:
        total_flops = flops_per_sample * actual_samples
        energy_per_flop = (total_kwh * 1000 * 3600) / total_flops  # Joules per FLOP

    # Comparison metrics
    relative_efficiency = None
    carbon_saved = None
    if baseline_energy_kwh is not None:
        relative_efficiency = baseline_energy_kwh / total_kwh
        carbon_saved = (baseline_energy_kwh - total_kwh) * carbon_intensity / 1000

    return AcademicMetrics(
        energy_kwh=total_kwh,
        energy_per_sample_wh=energy_per_sample_wh,
        energy_per_1k_samples_kwh=energy_per_1k_kwh,
        carbon_kg=carbon_kg,
        carbon_per_sample_g=carbon_per_sample_g,
        carbon_per_1k_samples_kg=carbon_per_1k_kg,
        accuracy=accuracy,
        throughput_samples_per_sec=throughput,
        latency_ms=latency_ms,
        ges=ges,
        perf_per_watt=perf_per_watt,
        accuracy_per_carbon=acc_per_carbon,
        flops_per_sample=flops_per_sample,
        energy_per_flop=energy_per_flop,
        relative_efficiency=relative_efficiency,
        carbon_saved_vs_baseline=carbon_saved,
        hardware=hardware,
        region=region,
        n_samples=actual_samples,
        measurement_time_s=elapsed_time,
    )


def compare_academic(
    models: List[Any],
    model_names: List[str],
    data: Any,
    labels: Optional[Any] = None,
    n_samples: int = 1000,
    region: str = "WORLD",
) -> str:
    """
    Generate a LaTeX comparison table for multiple models.

    Returns a complete LaTeX table ready for inclusion in a paper.
    """
    results = []
    for model, name in zip(models, model_names):
        metrics = measure_academic(model, data, labels, n_samples, region=region)
        results.append((name, metrics))

    # Sort by GES (most efficient first)
    results.sort(key=lambda x: x[1].ges, reverse=True)

    # Generate LaTeX
    latex = """
\\begin{table*}[t]
\\centering
\\caption{Comparison of Model Efficiency}
\\label{tab:comparison}
\\begin{tabular}{lrrrrrrr}
\\toprule
\\textbf{Model} & \\textbf{Acc (\\%)} & \\textbf{kWh/1K} & \\textbf{CO$_2$ (g/1K)} & \\textbf{GES} & \\textbf{Throughput} & \\textbf{Latency (ms)} \\\\
\\midrule
"""
    for name, m in results:
        latex += f"{name} & {m.accuracy:.1f} & {m.energy_per_1k_samples_kwh:.6f} & {m.carbon_per_sample_g*1000:.2f} & {m.ges:,.0f} & {m.throughput_samples_per_sec:.1f} & {m.latency_ms:.2f} \\\\\n"

    latex += """\\bottomrule
\\end{tabular}
\\end{table*}
"""
    return latex
