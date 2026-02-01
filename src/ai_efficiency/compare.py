"""
Compare efficiency across multiple models.
"""

from typing import Any, List, Optional, Dict
from dataclasses import dataclass

from .measure import measure, EfficiencyScore


@dataclass
class ComparisonResult:
    """Result of comparing multiple models."""

    scores: Dict[str, EfficiencyScore]
    ranking: List[str]  # Model names sorted by efficiency (best first)

    def __str__(self) -> str:
        lines = [
            "AI Efficiency Comparison",
            "=" * 70,
            f"{'Model':<20} | {'Accuracy':>8} | {'kWh/1K':>10} | {'Efficiency':>12} | {'Grade':>5}",
            "-" * 70,
        ]

        for name in self.ranking:
            score = self.scores[name]
            lines.append(
                f"{name:<20} | {score.accuracy:>7.1f}% | {score.kwh_per_1k:>10.6f} | "
                f"{score.efficiency:>12,.0f} | {score.grade:>5}"
            )

        lines.append("-" * 70)
        lines.append(f"Winner: {self.ranking[0]} (most efficient)")

        return "\n".join(lines)

    def to_dict(self) -> dict:
        return {
            "scores": {name: score.to_dict() for name, score in self.scores.items()},
            "ranking": self.ranking,
            "winner": self.ranking[0] if self.ranking else None,
        }


def compare(
    models: List[Any],
    data: Any,
    labels: Optional[Any] = None,
    model_names: Optional[List[str]] = None,
    n_samples: int = 1000,
    hardware: Optional[str] = None,
    region: str = "WORLD",
) -> ComparisonResult:
    """
    Compare efficiency across multiple models.

    Args:
        models: List of models to compare
        data: Input data for all models
        labels: Optional ground truth labels
        model_names: Optional names for each model
        n_samples: Number of samples to use
        hardware: Hardware type (auto-detected if None)
        region: Region code for carbon intensity

    Returns:
        ComparisonResult with scores and ranking

    Example:
        >>> from ai_efficiency import compare
        >>> result = compare([model_a, model_b], test_data, test_labels)
        >>> print(result)
        >>> print(f"Winner: {result.ranking[0]}")
    """

    # Generate names if not provided
    if model_names is None:
        model_names = [f"model_{i}" for i in range(len(models))]

    if len(model_names) != len(models):
        raise ValueError("Number of model_names must match number of models")

    # Measure each model
    scores = {}
    for name, model in zip(model_names, models):
        scores[name] = measure(
            model=model,
            data=data,
            labels=labels,
            n_samples=n_samples,
            hardware=hardware,
            region=region,
        )

    # Rank by efficiency (highest first)
    ranking = sorted(scores.keys(), key=lambda x: scores[x].efficiency, reverse=True)

    return ComparisonResult(scores=scores, ranking=ranking)
