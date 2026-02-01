"""
Benchmark script for GES paper experiments.

Measures efficiency of 12 models across 3 tasks:
- Image Classification: ResNet-50, EfficientNet-B0, MobileNetV3
- Text Classification: BERT-base, DistilBERT, TinyBERT, ALBERT
- Language Generation: GPT-2 (small, medium)

Usage:
    python benchmark_models.py --task image --output results/
    python benchmark_models.py --task text --output results/
    python benchmark_models.py --task all --output results/
"""

import argparse
import json
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")


def check_dependencies():
    """Check if required packages are installed."""
    missing = []

    try:
        import torch
    except ImportError:
        missing.append("torch")

    try:
        import transformers
    except ImportError:
        missing.append("transformers")

    try:
        import torchvision
    except ImportError:
        missing.append("torchvision")

    try:
        import numpy
    except ImportError:
        missing.append("numpy")

    if missing:
        print(f"Missing packages: {', '.join(missing)}")
        print(f"Install with: pip install {' '.join(missing)}")
        return False
    return True


def benchmark_image_models(n_samples: int = 1000, device: str = "cuda") -> List[Dict]:
    """Benchmark image classification models."""
    import torch
    import torchvision.models as models
    from ai_efficiency import measure

    results = []

    # Model configurations
    image_models = [
        ("ResNet-50", models.resnet50, {"weights": models.ResNet50_Weights.DEFAULT}),
        ("EfficientNet-B0", models.efficientnet_b0, {"weights": models.EfficientNet_B0_Weights.DEFAULT}),
        ("MobileNetV3-Large", models.mobilenet_v3_large, {"weights": models.MobileNet_V3_Large_Weights.DEFAULT}),
        ("MobileNetV3-Small", models.mobilenet_v3_small, {"weights": models.MobileNet_V3_Small_Weights.DEFAULT}),
    ]

    # Create dummy data (ImageNet-like)
    dummy_data = torch.randn(n_samples, 3, 224, 224)
    if device == "cuda" and torch.cuda.is_available():
        dummy_data = dummy_data.cuda()

    for name, model_fn, kwargs in image_models:
        print(f"Benchmarking {name}...")

        try:
            model = model_fn(**kwargs)
            model.eval()
            if device == "cuda" and torch.cuda.is_available():
                model = model.cuda()

            # Wrap for ai_efficiency
            class ModelWrapper:
                def __init__(self, m):
                    self.model = m

                def predict(self, x):
                    with torch.no_grad():
                        return self.model(x)

            wrapper = ModelWrapper(model)

            # Measure efficiency
            score = measure(
                wrapper,
                dummy_data,
                n_samples=n_samples,
                region="KR"
            )

            results.append({
                "model": name,
                "task": "Image Classification",
                "accuracy": 76.0,  # Placeholder - use actual ImageNet accuracy
                "efficiency": score.efficiency,
                "grade": score.grade,
                "kwh_per_1k": score.kwh_per_1k,
                "co2_per_1k": score.co2_per_1k,
                "hardware": score.hardware,
                "n_samples": n_samples,
            })

            print(f"  GES: {score.efficiency:,.0f}, Grade: {score.grade}")

        except Exception as e:
            print(f"  Error: {e}")

    return results


def benchmark_text_models(n_samples: int = 1000, device: str = "cuda") -> List[Dict]:
    """Benchmark text classification models."""
    import torch
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    from ai_efficiency import measure

    results = []

    # Model configurations (model_name, accuracy on GLUE)
    text_models = [
        ("bert-base-uncased", "BERT-base", 88.5),
        ("distilbert-base-uncased", "DistilBERT", 87.2),
        ("huawei-noah/TinyBERT_General_4L_312D", "TinyBERT", 84.5),
        ("albert-base-v2", "ALBERT-base", 86.3),
    ]

    for model_id, name, accuracy in text_models:
        print(f"Benchmarking {name}...")

        try:
            tokenizer = AutoTokenizer.from_pretrained(model_id)
            model = AutoModelForSequenceClassification.from_pretrained(model_id, num_labels=2)
            model.eval()
            if device == "cuda" and torch.cuda.is_available():
                model = model.cuda()

            # Create dummy text data
            texts = ["This is a sample text for classification."] * n_samples
            inputs = tokenizer(texts, padding=True, truncation=True, max_length=128, return_tensors="pt")
            if device == "cuda" and torch.cuda.is_available():
                inputs = {k: v.cuda() for k, v in inputs.items()}

            # Wrap for ai_efficiency
            class ModelWrapper:
                def __init__(self, m, inp):
                    self.model = m
                    self.inputs = inp

                def predict(self, x):
                    with torch.no_grad():
                        outputs = self.model(**self.inputs)
                        return outputs.logits.argmax(dim=-1)

            wrapper = ModelWrapper(model, inputs)

            # Measure efficiency
            score = measure(
                wrapper,
                texts,
                n_samples=n_samples,
                region="KR"
            )

            results.append({
                "model": name,
                "task": "Text Classification",
                "accuracy": accuracy,
                "efficiency": score.efficiency,
                "grade": score.grade,
                "kwh_per_1k": score.kwh_per_1k,
                "co2_per_1k": score.co2_per_1k,
                "hardware": score.hardware,
                "n_samples": n_samples,
            })

            print(f"  GES: {score.efficiency:,.0f}, Grade: {score.grade}")

        except Exception as e:
            print(f"  Error: {e}")

    return results


def benchmark_generation_models(n_samples: int = 100, device: str = "cuda") -> List[Dict]:
    """Benchmark language generation models."""
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from ai_efficiency import measure

    results = []

    # Model configurations
    gen_models = [
        ("gpt2", "GPT-2 Small"),
        ("gpt2-medium", "GPT-2 Medium"),
    ]

    for model_id, name in gen_models:
        print(f"Benchmarking {name}...")

        try:
            tokenizer = AutoTokenizer.from_pretrained(model_id)
            tokenizer.pad_token = tokenizer.eos_token
            model = AutoModelForCausalLM.from_pretrained(model_id)
            model.eval()
            if device == "cuda" and torch.cuda.is_available():
                model = model.cuda()

            # Create dummy prompts
            prompts = ["The future of AI is"] * n_samples
            inputs = tokenizer(prompts, padding=True, return_tensors="pt")
            if device == "cuda" and torch.cuda.is_available():
                inputs = {k: v.cuda() for k, v in inputs.items()}

            # Wrap for ai_efficiency
            class ModelWrapper:
                def __init__(self, m, tok, inp):
                    self.model = m
                    self.tokenizer = tok
                    self.inputs = inp

                def predict(self, x):
                    with torch.no_grad():
                        outputs = self.model.generate(
                            **self.inputs,
                            max_new_tokens=20,
                            do_sample=False,
                            pad_token_id=self.tokenizer.eos_token_id
                        )
                        return outputs

            wrapper = ModelWrapper(model, tokenizer, inputs)

            # Measure efficiency
            score = measure(
                wrapper,
                prompts,
                n_samples=n_samples,
                region="KR"
            )

            results.append({
                "model": name,
                "task": "Language Generation",
                "accuracy": 100.0,  # Generation uses perplexity, not accuracy
                "efficiency": score.efficiency,
                "grade": score.grade,
                "kwh_per_1k": score.kwh_per_1k,
                "co2_per_1k": score.co2_per_1k,
                "hardware": score.hardware,
                "n_samples": n_samples,
            })

            print(f"  GES: {score.efficiency:,.0f}, Grade: {score.grade}")

        except Exception as e:
            print(f"  Error: {e}")

    return results


def save_results(results: List[Dict], output_dir: str):
    """Save benchmark results to JSON and generate LaTeX table."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Save JSON
    json_path = output_path / f"benchmark_results_{timestamp}.json"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {json_path}")

    # Generate LaTeX table
    latex = generate_latex_table(results)
    latex_path = output_path / f"results_table_{timestamp}.tex"
    with open(latex_path, "w") as f:
        f.write(latex)
    print(f"LaTeX table saved to {latex_path}")

    # Generate Markdown table
    md = generate_markdown_table(results)
    md_path = output_path / f"results_table_{timestamp}.md"
    with open(md_path, "w") as f:
        f.write(md)
    print(f"Markdown table saved to {md_path}")


def generate_latex_table(results: List[Dict]) -> str:
    """Generate LaTeX table from results."""
    # Sort by GES (efficiency)
    sorted_results = sorted(results, key=lambda x: x["efficiency"], reverse=True)

    latex = r"""
\begin{table}[h]
\centering
\caption{GES Benchmark Results}
\label{tab:benchmark}
\begin{tabular}{llrrrr}
\toprule
\textbf{Model} & \textbf{Task} & \textbf{Acc (\%)} & \textbf{kWh/1K} & \textbf{GES} & \textbf{Grade} \\
\midrule
"""

    for r in sorted_results:
        latex += f"{r['model']} & {r['task'][:10]} & {r['accuracy']:.1f} & {r['kwh_per_1k']:.6f} & {r['efficiency']:,.0f} & {r['grade']} \\\\\n"

    latex += r"""\bottomrule
\end{tabular}
\end{table}
"""
    return latex


def generate_markdown_table(results: List[Dict]) -> str:
    """Generate Markdown table from results."""
    sorted_results = sorted(results, key=lambda x: x["efficiency"], reverse=True)

    md = "# GES Benchmark Results\n\n"
    md += "| Model | Task | Accuracy | kWh/1K | GES | Grade |\n"
    md += "|-------|------|----------|--------|-----|-------|\n"

    for r in sorted_results:
        md += f"| {r['model']} | {r['task']} | {r['accuracy']:.1f}% | {r['kwh_per_1k']:.6f} | {r['efficiency']:,.0f} | {r['grade']} |\n"

    return md


def main():
    parser = argparse.ArgumentParser(description="Benchmark models for GES paper")
    parser.add_argument("--task", choices=["image", "text", "generation", "all"], default="all")
    parser.add_argument("--output", default="results", help="Output directory")
    parser.add_argument("--n-samples", type=int, default=100, help="Number of samples")
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    args = parser.parse_args()

    if not check_dependencies():
        return

    print("=" * 60)
    print("GES Benchmark - ecoai-efficiency")
    print("=" * 60)
    print(f"Task: {args.task}")
    print(f"Samples: {args.n_samples}")
    print(f"Device: {args.device}")
    print("=" * 60)

    all_results = []

    if args.task in ["image", "all"]:
        print("\n[Image Classification Models]")
        results = benchmark_image_models(args.n_samples, args.device)
        all_results.extend(results)

    if args.task in ["text", "all"]:
        print("\n[Text Classification Models]")
        results = benchmark_text_models(args.n_samples, args.device)
        all_results.extend(results)

    if args.task in ["generation", "all"]:
        print("\n[Language Generation Models]")
        results = benchmark_generation_models(min(args.n_samples, 50), args.device)
        all_results.extend(results)

    if all_results:
        save_results(all_results, args.output)
        print("\n" + "=" * 60)
        print("Benchmark complete!")
        print("=" * 60)


if __name__ == "__main__":
    main()
