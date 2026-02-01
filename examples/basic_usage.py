"""
Basic usage example for ai-efficiency.
"""

import numpy as np

# Simulate a simple model for demonstration
class SimpleModel:
    """A mock model for demonstration."""

    def predict(self, X):
        # Simulate some computation
        return (X.sum(axis=1) > 0).astype(int)


def main():
    # Import ai-efficiency
    from ai_efficiency import measure, compare, report

    # Create a simple model
    model = SimpleModel()

    # Generate some test data
    np.random.seed(42)
    X_test = np.random.randn(1000, 10)
    y_test = (X_test.sum(axis=1) > 0).astype(int)

    # ===================
    # 1. Measure efficiency
    # ===================
    print("=" * 50)
    print("1. Measuring Model Efficiency")
    print("=" * 50)

    score = measure(
        model=model,
        data=X_test,
        labels=y_test,
        n_samples=1000,
        region="KR",  # South Korea
    )

    print(score)
    print()

    # ===================
    # 2. Compare models
    # ===================
    print("=" * 50)
    print("2. Comparing Multiple Models")
    print("=" * 50)

    # Create another model (simulating a less efficient one)
    class HeavyModel:
        def predict(self, X):
            # Simulate heavier computation
            for _ in range(100):
                _ = X @ X.T
            return (X.sum(axis=1) > 0).astype(int)

    model_light = SimpleModel()
    model_heavy = HeavyModel()

    result = compare(
        models=[model_light, model_heavy],
        data=X_test,
        labels=y_test,
        model_names=["LightModel", "HeavyModel"],
        region="KR",
    )

    print(result)
    print()

    # ===================
    # 3. Generate report
    # ===================
    print("=" * 50)
    print("3. Generating Report")
    print("=" * 50)

    r = report(
        model=model,
        data=X_test,
        labels=y_test,
        model_name="SimpleModel-v1",
        region="KR",
    )

    # Save in different formats
    r.save("efficiency_report.md")
    r.save("efficiency_report.json")

    print("Reports saved:")
    print("  - efficiency_report.md")
    print("  - efficiency_report.json")
    print()

    # ===================
    # 4. Access metrics
    # ===================
    print("=" * 50)
    print("4. Accessing Individual Metrics")
    print("=" * 50)

    print(f"Efficiency Score: {score.efficiency:,.0f}")
    print(f"Grade: {score.grade}")
    print(f"Accuracy: {score.accuracy:.1f}%")
    print(f"Energy per 1K queries: {score.kwh_per_1k:.6f} kWh")
    print(f"Carbon per 1K queries: {score.co2_per_1k:.2f}g CO2")
    print()

    # Export as dict (for logging, databases, etc.)
    metrics_dict = score.to_dict()
    print("As dictionary:")
    for key, value in metrics_dict.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
