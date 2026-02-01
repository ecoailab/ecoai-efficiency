"""Tests for academic metrics module."""

import pytest
import numpy as np


class MockModel:
    """Simple mock model for testing."""

    def __init__(self, accuracy: float = 0.9, latency: float = 0.001):
        self.accuracy = accuracy
        self.latency = latency

    def predict(self, data):
        import time
        time.sleep(self.latency * len(data))
        n = len(data)
        correct = int(n * self.accuracy)
        predictions = np.zeros(n, dtype=int)
        predictions[:correct] = np.arange(correct) % 10
        return predictions


def test_measure_academic_basic():
    """Test basic academic metrics."""
    from ai_efficiency import measure_academic

    model = MockModel(accuracy=0.9)
    data = np.random.randn(100, 10)
    labels = np.arange(100) % 10

    metrics = measure_academic(model, data, labels, n_samples=100, hardware="CPU", region="KR")

    # Energy metrics
    assert metrics.energy_kwh > 0
    assert metrics.energy_per_sample_wh > 0
    assert metrics.energy_per_1k_samples_kwh > 0

    # Carbon metrics
    assert metrics.carbon_kg > 0
    assert metrics.carbon_per_sample_g > 0

    # Performance metrics
    assert metrics.accuracy == pytest.approx(90.0, abs=5)
    assert metrics.throughput_samples_per_sec > 0
    assert metrics.latency_ms > 0

    # Efficiency ratios
    assert metrics.ges > 0
    assert metrics.perf_per_watt > 0


def test_academic_latex_table():
    """Test LaTeX table generation."""
    from ai_efficiency import measure_academic

    model = MockModel()
    data = np.random.randn(50, 10)

    metrics = measure_academic(model, data, n_samples=50, hardware="CPU")
    latex = metrics.to_latex_table(caption="Test Results")

    assert "\\begin{table}" in latex
    assert "\\end{table}" in latex
    assert "Test Results" in latex
    assert "Energy" in latex
    assert "CO$_2$" in latex
    assert "GES" in latex


def test_academic_markdown_table():
    """Test Markdown table generation."""
    from ai_efficiency import measure_academic

    model = MockModel()
    data = np.random.randn(50, 10)

    metrics = measure_academic(model, data, n_samples=50, hardware="CPU")
    md = metrics.to_markdown_table()

    assert "| Metric | Value | Unit |" in md
    assert "**Energy**" in md
    assert "**Carbon**" in md
    assert "**Efficiency**" in md


def test_academic_to_dict():
    """Test dictionary export."""
    from ai_efficiency import measure_academic

    model = MockModel()
    data = np.random.randn(50, 10)

    metrics = measure_academic(model, data, n_samples=50, hardware="CPU")
    d = metrics.to_dict()

    assert "energy_kwh" in d
    assert "carbon_kg" in d
    assert "accuracy" in d
    assert "ges" in d
    assert "hardware" in d


def test_academic_flops_metrics():
    """Test FLOPS-based metrics."""
    from ai_efficiency import measure_academic

    model = MockModel()
    data = np.random.randn(50, 10)

    # With FLOPs specified
    metrics = measure_academic(
        model, data, n_samples=50, hardware="CPU",
        flops_per_sample=1e9  # 1 GFLOP per sample
    )

    assert metrics.flops_per_sample == 1e9
    assert metrics.energy_per_flop is not None
    assert metrics.energy_per_flop > 0


def test_academic_baseline_comparison():
    """Test baseline comparison metrics."""
    from ai_efficiency import measure_academic

    model = MockModel()
    data = np.random.randn(50, 10)

    # With baseline energy
    metrics = measure_academic(
        model, data, n_samples=50, hardware="CPU",
        baseline_energy_kwh=0.01  # 10 Wh baseline
    )

    assert metrics.relative_efficiency is not None
    assert metrics.carbon_saved_vs_baseline is not None


def test_compare_academic():
    """Test academic comparison table."""
    from ai_efficiency import compare_academic

    model_a = MockModel(accuracy=0.95, latency=0.002)
    model_b = MockModel(accuracy=0.85, latency=0.001)

    data = np.random.randn(50, 10)
    labels = np.arange(50) % 10

    latex = compare_academic(
        [model_a, model_b],
        ["High Accuracy", "Fast Model"],
        data,
        labels,
        n_samples=50,
    )

    assert "\\begin{table*}" in latex
    assert "Comparison" in latex
    assert "High Accuracy" in latex or "Fast Model" in latex
