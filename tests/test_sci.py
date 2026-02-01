"""Tests for SCI for AI module."""

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


def test_sci_report_basic():
    """Test basic SCI report generation."""
    from ai_efficiency import sci_report

    model = MockModel(accuracy=0.9)
    data = np.random.randn(100, 10)
    labels = np.arange(100) % 10

    sci = sci_report(model, data, labels, n_samples=100, hardware="CPU", region="KR")

    assert sci.sci > 0
    assert sci.energy_kwh > 0
    assert sci.carbon_intensity == 450  # KR static
    assert sci.embodied_carbon >= 0
    assert sci.hardware == "CPU"
    assert sci.functional_unit == "1000 inferences"


def test_sci_components():
    """Test SCI component calculations."""
    from ai_efficiency import sci_report

    model = MockModel()
    data = np.random.randn(50, 10)

    sci = sci_report(model, data, n_samples=50, hardware="CPU")

    # Check formula: SCI = ((E × I) + M) / R
    expected_operational = sci.energy_kwh * sci.carbon_intensity
    assert sci.operational_carbon == pytest.approx(expected_operational, rel=0.01)

    expected_total = expected_operational + sci.embodied_carbon
    assert sci.total_carbon == pytest.approx(expected_total, rel=0.01)


def test_sci_without_embodied():
    """Test SCI without embodied carbon."""
    from ai_efficiency import sci_report

    model = MockModel()
    data = np.random.randn(50, 10)

    sci = sci_report(model, data, n_samples=50, hardware="CPU", include_embodied=False)

    assert sci.embodied_carbon == 0.0
    assert sci.total_carbon == sci.operational_carbon


def test_sci_score_str():
    """Test SCIScore string representation."""
    from ai_efficiency import sci_report

    model = MockModel()
    data = np.random.randn(50, 10)

    sci = sci_report(model, data, n_samples=50, hardware="CPU")
    output = str(sci)

    assert "SCI for AI Report" in output
    assert "SCI Score:" in output
    assert "E (Energy):" in output
    assert "I (Carbon Intensity):" in output
    assert "M (Embodied):" in output


def test_sci_score_to_dict():
    """Test SCIScore to_dict export."""
    from ai_efficiency import sci_report

    model = MockModel()
    data = np.random.randn(50, 10)

    sci = sci_report(model, data, n_samples=50, hardware="CPU")
    d = sci.to_dict()

    assert "sci_score" in d
    assert "components" in d
    assert "E_energy_kwh" in d["components"]
    assert "I_carbon_intensity_gco2_kwh" in d["components"]
    assert "M_embodied_gco2" in d["components"]


def test_sci_report_markdown():
    """Test SCI markdown report generation."""
    from ai_efficiency import sci_report

    model = MockModel()
    data = np.random.randn(50, 10)

    sci = sci_report(model, data, n_samples=50, hardware="CPU")
    md = sci.to_sci_report(model_name="TestModel")

    assert "# SCI for AI Report" in md
    assert "TestModel" in md
    assert "SCI for AI v1.0" in md
    assert "Green Software Foundation" in md


def test_compare_sci():
    """Test SCI comparison across models."""
    from ai_efficiency import compare_sci

    model_a = MockModel(accuracy=0.95, latency=0.002)
    model_b = MockModel(accuracy=0.85, latency=0.001)

    data = np.random.randn(50, 10)
    labels = np.arange(50) % 10

    result = compare_sci(
        [model_a, model_b],
        ["High Accuracy", "Fast Model"],
        data,
        labels,
        n_samples=50,
    )

    assert "# SCI Comparison" in result
    assert "High Accuracy" in result
    assert "Fast Model" in result
    assert "**Most Efficient**:" in result
