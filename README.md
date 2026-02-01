# ai-efficiency

> Measure the energy efficiency of any AI model. One number. No excuses.

[![PyPI version](https://img.shields.io/pypi/v/ai-efficiency.svg)](https://pypi.org/project/ai-efficiency/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**ai-efficiency** gives you one metric: **how much intelligence per watt**.

```python
from ai_efficiency import measure

score = measure(your_model, test_data)
print(score)
# → Efficiency: 47,000 (accuracy per kWh)
# → Grade: A
# → Carbon: 0.8g CO2 per 1000 queries
```

## Why?

- **Regulators are coming.** EU AI Act will require energy disclosure.
- **Investors are asking.** ESG funds want AI carbon footprints.
- **Costs are exploding.** GPU bills matter. Efficiency = money saved.

There's no standard way to measure AI efficiency. Now there is.

## Install

```bash
pip install ai-efficiency
```

## Quick Start

### Measure Any Model

```python
from ai_efficiency import measure

# Works with any model that has a predict() or __call__() method
score = measure(
    model=your_model,
    test_data=X_test,
    n_samples=1000
)

print(f"Efficiency Score: {score.efficiency:,.0f}")
print(f"Energy per 1K queries: {score.kwh_per_1k:.4f} kWh")
print(f"Carbon per 1K queries: {score.co2_per_1k:.2f}g CO2")
print(f"Grade: {score.grade}")
```

### Compare Models

```python
from ai_efficiency import compare

results = compare([model_a, model_b, model_c], test_data)

# Output:
# Model     | Accuracy | kWh/1K  | Efficiency | Grade
# ----------|----------|---------|------------|------
# model_a   | 94.2%    | 0.0100  | 9,420      | B
# model_b   | 91.8%    | 0.0008  | 114,750    | A+
# model_c   | 96.1%    | 0.0450  | 2,136      | C
```

### Generate Report

```python
from ai_efficiency import report

r = report(model, test_data)
r.save("efficiency_report.pdf")  # For regulators, investors
r.save("efficiency_report.json") # For CI/CD pipelines
```

## The Efficiency Score

```
                    Accuracy (or Quality Metric)
Efficiency Score = ──────────────────────────────
                    Energy Consumption (kWh)
```

Higher is better. A model with 95% accuracy using 0.001 kWh scores 95,000.

### Grades

| Grade | Efficiency Score | Meaning |
|-------|------------------|---------|
| A+    | > 100,000        | Exceptional efficiency |
| A     | 50,000 - 100,000 | Very efficient |
| B     | 10,000 - 50,000  | Good |
| C     | 1,000 - 10,000   | Average |
| D     | < 1,000          | Needs optimization |

## Carbon Calculation

We use regional grid carbon intensity:

```python
from ai_efficiency import measure

# Specify your region for accurate carbon calculation
score = measure(model, data, region="KR")  # South Korea: 450g CO2/kWh
score = measure(model, data, region="EU")  # Europe avg: 250g CO2/kWh
score = measure(model, data, region="US-CA")  # California: 200g CO2/kWh
```

### Real-Time Carbon Intensity

Get live grid carbon data (requires [Electricity Maps](https://electricitymaps.com) API key):

```python
from ai_efficiency import get_carbon_intensity

# Set API key (or use ELECTRICITY_MAPS_API_KEY env var)
intensity = get_carbon_intensity("KR", real_time=True, api_key="your-key")
print(f"Current grid intensity: {intensity} gCO2/kWh")
```

### Cloud Provider Carbon Factors

```python
from ai_efficiency import get_cloud_carbon_factor

# AWS, GCP, Azure carbon factors
aws_carbon = get_cloud_carbon_factor("aws", "us-east-1")  # 379 gCO2/kWh
gcp_carbon = get_cloud_carbon_factor("gcp", "us-west1")   # 92 gCO2/kWh (low!)
```

## SCI for AI (Green Software Foundation)

Generate reports compliant with [SCI for AI](https://sci-for-ai.greensoftware.foundation/) standard:

```python
from ai_efficiency import sci_report

sci = sci_report(model, test_data, region="KR")
print(sci)
# → SCI Score: 0.0234 gCO2eq/1000 inferences
# → E (Energy): 0.000052 kWh
# → I (Carbon Intensity): 450 gCO2eq/kWh
# → M (Embodied): 0.0001 gCO2eq

# Export SCI-compliant report
print(sci.to_sci_report(model_name="MyModel"))
```

## Academic Paper Metrics

Generate paper-ready metrics with LaTeX export:

```python
from ai_efficiency import measure_academic

metrics = measure_academic(
    model, test_data, test_labels,
    region="KR",
    flops_per_sample=1.2e9  # Optional: FLOPs if known
)

# LaTeX table for papers
print(metrics.to_latex_table())

# Markdown for README
print(metrics.to_markdown_table())

# Full metrics
print(f"GES: {metrics.ges:,.0f}")           # Green Efficiency Score
print(f"Energy/FLOP: {metrics.energy_per_flop:.2e}")  # If FLOPs provided
```

### Compare Models (Academic)

```python
from ai_efficiency import compare_academic

latex_table = compare_academic(
    [model_a, model_b],
    ["BERT-base", "DistilBERT"],
    test_data, test_labels
)
# → Generates LaTeX table sorted by efficiency
```

## Supported Models

- **PyTorch**: Any `nn.Module`
- **TensorFlow/Keras**: Any model with `predict()`
- **Scikit-learn**: Any estimator
- **Hugging Face**: Transformers, Diffusers
- **OpenAI/Anthropic**: API-based models (estimated)
- **Custom**: Any callable with `.predict()` or `__call__()`

## CLI Tool

```bash
# Measure a saved model
ai-efficiency measure model.pt --data test.csv

# Compare multiple models
ai-efficiency compare model_a.pt model_b.pt --data test.csv

# Generate compliance report
ai-efficiency report model.pt --data test.csv --output report.pdf
```

## CI/CD Integration

```yaml
# .github/workflows/efficiency.yml
- name: Check AI Efficiency
  run: |
    pip install ai-efficiency
    ai-efficiency check model.pt --min-grade B --fail-below C
```

## Research

This project is based on research from EcoAI Lab, Hanbat National University:

- Energy-aware machine learning
- Carbon footprint of AI systems
- Efficient inference optimization

## Contributing

We need help with:
- GPU power measurement accuracy
- More hardware profiles (TPU, Apple Silicon)
- Cloud provider integrations (AWS, GCP, Azure)
- Regional carbon intensity data

See [CONTRIBUTING.md](CONTRIBUTING.md).

## Roadmap

- [x] Basic measurement (CPU)
- [x] GPU power measurement (NVIDIA)
- [x] Real-time carbon intensity (Electricity Maps API)
- [x] Cloud provider carbon factors (AWS, GCP, Azure)
- [x] SCI for AI compliance (Green Software Foundation)
- [x] Academic metrics with LaTeX export
- [x] Embodied carbon calculation
- [ ] Process-level power measurement
- [ ] Optimization suggestions
- [ ] Certification API

## License

MIT

## Team

**EcoAI Lab, Hanbat National University**
- Director: Prof. Sangkeum Lee
- Website: [ecoai.hanbat.ac.kr](https://sites.google.com/view/ecoai)

---

*"The greenest AI is the one that does more with less."*
