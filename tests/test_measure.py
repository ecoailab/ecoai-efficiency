"""Tests for core measure functionality."""

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
        # Return predictions with given accuracy
        n = len(data)
        correct = int(n * self.accuracy)
        predictions = np.zeros(n, dtype=int)
        predictions[:correct] = np.arange(correct) % 10  # Match labels
        return predictions


def test_measure_basic():
    """Test basic measure functionality."""
    from ai_efficiency import measure

    model = MockModel(accuracy=0.9)
    data = np.random.randn(100, 10)
    labels = np.arange(100) % 10

    score = measure(model, data, labels, n_samples=100, hardware="CPU", region="KR")

    assert score.accuracy == pytest.approx(90.0, abs=5)
    assert score.efficiency > 0
    assert score.grade in ["A+", "A", "B", "C", "D"]
    assert score.hardware == "CPU"
    assert score.region == "KR"


def test_measure_no_labels():
    """Test measure without labels (accuracy defaults to 100)."""
    from ai_efficiency import measure

    model = MockModel()
    data = np.random.randn(50, 10)

    score = measure(model, data, n_samples=50, hardware="CPU")

    assert score.accuracy == 100.0
    assert score.efficiency > 0


def test_efficiency_score_str():
    """Test EfficiencyScore string representation."""
    from ai_efficiency import measure

    model = MockModel()
    data = np.random.randn(50, 10)

    score = measure(model, data, n_samples=50, hardware="CPU")
    output = str(score)

    assert "AI Efficiency Report" in output
    assert "Grade:" in output
    assert "Accuracy:" in output


def test_efficiency_score_to_dict():
    """Test EfficiencyScore to_dict export."""
    from ai_efficiency import measure

    model = MockModel()
    data = np.random.randn(50, 10)

    score = measure(model, data, n_samples=50, hardware="CPU")
    d = score.to_dict()

    assert "efficiency_score" in d
    assert "grade" in d
    assert "accuracy" in d
    assert "hardware" in d


def test_compare_models():
    """Test model comparison."""
    from ai_efficiency import compare

    model_a = MockModel(accuracy=0.95, latency=0.002)
    model_b = MockModel(accuracy=0.85, latency=0.001)

    data = np.random.randn(50, 10)
    labels = np.arange(50) % 10

    result = compare(
        [model_a, model_b],
        data,
        labels,
        model_names=["High Accuracy", "Fast Model"],
        n_samples=50,
        hardware="CPU",
    )

    assert len(result.ranking) == 2
    assert len(result.scores) == 2
    assert "High Accuracy" in result.scores
    assert "Fast Model" in result.scores


def test_grades():
    """Test grade assignment based on efficiency."""
    from ai_efficiency.measure import _get_grade

    assert _get_grade(150000) == "A+"
    assert _get_grade(100000) == "A+"
    assert _get_grade(75000) == "A"
    assert _get_grade(50000) == "A"
    assert _get_grade(25000) == "B"
    assert _get_grade(5000) == "C"
    assert _get_grade(500) == "D"
    assert _get_grade(0) == "D"
