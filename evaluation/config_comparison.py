"""
evaluation/config_comparison.py
Runs retrieval metrics under two configurations and prints a comparison table.

Default comparison: top-K=3 vs top-K=5.

Usage:
    python config_comparison.py
    python config_comparison.py --out config_comparison.json
"""

from __future__ import annotations

import argparse
import json
import sys
import os

# Make sure parent path is visible (safe to keep)
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Imports (same folder structure)
from retrieval_metrics import compute_metrics

# Configurations to compare
CONFIGS = [
    {"label": "top-K=3", "k": 3},
    {"label": "top-K=5", "k": 5},
]


def print_comparison_table(results: list[dict]) -> None:
    print("\n=== CONFIGURATION COMPARISON ===\n")

    header = f"{'Config':<20} {'Precision':<12} {'Recall':<12} {'MRR':<10} {'Questions'}"
    print(header)
    print("-" * len(header))

    for r in results:
        s = r["summary"]
        k = s["k"]

        print(
            f"{r['label']:<20} "
            f"{s.get(f'mean_precision@{k}', 0):<12} "
            f"{s.get(f'mean_recall@{k}', 0):<12} "
            f"{s.get('MRR', 0):<10} "
            f"{s['questions_evaluated']}"
        )

    print()

    # Determine best configuration by MRR
    mrr_values = [(r["label"], r["summary"]["MRR"]) for r in results]
    best = max(mrr_values, key=lambda x: x[1])

    print(f"Best configuration by MRR: {best[0]} (MRR={best[1]})")


def main(out_path: str) -> None:
    all_results = []

    for config in CONFIGS:
        print(f"Running config: {config['label']} ...")

        #  FIXED: correct function call
        metrics = compute_metrics(config["k"])

        # attach label for comparison
        metrics["label"] = config["label"]
        all_results.append(metrics)

    # Print comparison
    print_comparison_table(all_results)

    # Save results
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)

    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="config_comparison.json")
    args = parser.parse_args()

    main(args.out)