"""
Command-line interface for ai-efficiency.
"""

import argparse
import sys
import json


def main():
    parser = argparse.ArgumentParser(
        prog="ai-efficiency",
        description="Measure the energy efficiency of AI models",
    )

    subparsers = parser.add_subparsers(dest="command", help="Commands")

    # measure command
    measure_parser = subparsers.add_parser("measure", help="Measure a model's efficiency")
    measure_parser.add_argument("model", help="Path to model file")
    measure_parser.add_argument("--data", required=True, help="Path to test data")
    measure_parser.add_argument("--labels", help="Path to labels (optional)")
    measure_parser.add_argument("--hardware", help="Hardware type (auto-detected if not specified)")
    measure_parser.add_argument("--region", default="WORLD", help="Region code for carbon intensity")
    measure_parser.add_argument("--samples", type=int, default=1000, help="Number of samples")
    measure_parser.add_argument("--output", "-o", help="Output file (json)")

    # compare command
    compare_parser = subparsers.add_parser("compare", help="Compare multiple models")
    compare_parser.add_argument("models", nargs="+", help="Paths to model files")
    compare_parser.add_argument("--data", required=True, help="Path to test data")
    compare_parser.add_argument("--output", "-o", help="Output file (json)")

    # report command
    report_parser = subparsers.add_parser("report", help="Generate efficiency report")
    report_parser.add_argument("model", help="Path to model file")
    report_parser.add_argument("--data", required=True, help="Path to test data")
    report_parser.add_argument("--output", "-o", default="efficiency_report.md", help="Output file")
    report_parser.add_argument("--name", default="model", help="Model name for report")

    # check command (for CI/CD)
    check_parser = subparsers.add_parser("check", help="Check if model meets efficiency threshold")
    check_parser.add_argument("model", help="Path to model file")
    check_parser.add_argument("--data", required=True, help="Path to test data")
    check_parser.add_argument("--min-grade", default="C", help="Minimum acceptable grade")
    check_parser.add_argument("--fail-below", default="D", help="Fail if grade is this or worse")

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(0)

    if args.command == "measure":
        print(f"Measuring efficiency of {args.model}...")
        print("(Full implementation requires loading your specific model format)")
        print()
        print("Example usage in Python:")
        print("  from ai_efficiency import measure")
        print("  score = measure(model, test_data)")
        print("  print(score)")

    elif args.command == "compare":
        print(f"Comparing {len(args.models)} models...")
        print("(Full implementation requires loading your specific model format)")

    elif args.command == "report":
        print(f"Generating report for {args.model}...")
        print(f"Output: {args.output}")

    elif args.command == "check":
        print(f"Checking {args.model} against grade threshold {args.min_grade}...")
        print("(For CI/CD integration)")


if __name__ == "__main__":
    main()
