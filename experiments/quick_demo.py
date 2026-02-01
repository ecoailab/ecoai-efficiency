"""
Quick demo of GES measurement.
No heavy dependencies required - uses mock models.
"""

import numpy as np
import sys
sys.path.insert(0, '../src')

from ai_efficiency import measure, compare, measure_academic, sci_report


class MockModel:
    """Mock model for demonstration."""

    def __init__(self, name: str, accuracy: float = 0.9, latency: float = 0.001):
        self.name = name
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


def demo_measure():
    """Demonstrate basic measurement."""
    print("=" * 60)
    print("1. Basic GES Measurement")
    print("=" * 60)

    model = MockModel("DemoModel", accuracy=0.92)
    data = np.random.randn(100, 10)
    labels = np.arange(100) % 10

    score = measure(model, data, labels, n_samples=100, hardware="CPU", region="KR")
    print(score)


def demo_compare():
    """Demonstrate model comparison."""
    print("\n" + "=" * 60)
    print("2. Model Comparison")
    print("=" * 60)

    models = [
        MockModel("HighAccuracy", accuracy=0.95, latency=0.002),
        MockModel("FastModel", accuracy=0.85, latency=0.0005),
        MockModel("Balanced", accuracy=0.90, latency=0.001),
    ]

    data = np.random.randn(100, 10)
    labels = np.arange(100) % 10

    result = compare(
        models, data, labels,
        model_names=["High Accuracy", "Fast Model", "Balanced"],
        n_samples=100,
        hardware="CPU",
        region="KR"
    )
    print(result)


def demo_academic():
    """Demonstrate academic metrics."""
    print("\n" + "=" * 60)
    print("3. Academic Metrics (Paper-ready)")
    print("=" * 60)

    model = MockModel("ResNet-50", accuracy=0.92)
    data = np.random.randn(100, 10)
    labels = np.arange(100) % 10

    metrics = measure_academic(
        model, data, labels,
        n_samples=100,
        hardware="CPU",
        region="KR",
        flops_per_sample=1e9
    )

    print("\n[LaTeX Table]")
    print(metrics.to_latex_table())

    print("\n[Markdown Table]")
    print(metrics.to_markdown_table())


def demo_sci():
    """Demonstrate SCI for AI compliance."""
    print("\n" + "=" * 60)
    print("4. SCI for AI (Green Software Foundation)")
    print("=" * 60)

    model = MockModel("BERT-base", accuracy=0.88)
    data = np.random.randn(100, 10)

    sci = sci_report(
        model, data,
        n_samples=100,
        hardware="CPU",
        region="KR"
    )
    print(sci)


def main():
    print("=" * 60)
    print("ecoai-efficiency Quick Demo")
    print("GES: Green Efficiency Score for AI")
    print("=" * 60)

    demo_measure()
    demo_compare()
    demo_academic()
    demo_sci()

    print("\n" + "=" * 60)
    print("Demo complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
