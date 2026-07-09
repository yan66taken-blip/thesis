"""
Visualize Analysis Agent evaluation results.

Generates two charts:
  1. Accuracy bar chart across all three test categories
  2. Boundary-case confusion breakdown (expected vs predicted tool)

Usage:
    python test/plot_analysis_eval.py
"""

import json
from pathlib import Path
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np

RESULTS_PATH = Path(__file__).parent / "analysis_agent_results.json"
OUT_DIR = Path(__file__).parent.parent


def load_results():
    return json.loads(RESULTS_PATH.read_text())


def plot_accuracy_bar(results: dict):
    labels_map = {
        "tool_selection": "Tool Selection",
        "params": "Parameter Accuracy",
        "boundary": "Boundary Cases",
    }
    categories = list(labels_map.values())
    accuracies = [results[k]["accuracy"] * 100 for k in labels_map]
    counts = [results[k]["n"] for k in labels_map]

    fig, ax = plt.subplots(figsize=(8, 5))
    colors = ["#2ecc71", "#3498db", "#e67e22"]
    bars = ax.bar(categories, accuracies, color=colors, edgecolor="white", width=0.5)

    for bar, acc, n in zip(bars, accuracies, counts):
        correct = round(acc * n / 100)
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.5,
                f"{acc:.0f}%\n({correct}/{n})",
                ha="center", va="bottom", fontsize=11, fontweight="bold")

    ax.set_ylim(0, 115)
    ax.set_ylabel("Accuracy (%)", fontsize=12)
    ax.set_title("Analysis Agent Evaluation Results", fontsize=14, fontweight="bold")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.axhline(y=100, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)

    plt.tight_layout()
    out = OUT_DIR / "eval_analysis_accuracy.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Saved → {out}")


def plot_boundary_confusion(results: dict):
    details = results["boundary"]["details"]
    failures = [d for d in details if not d["match"]]

    if not failures:
        print("No boundary failures to plot.")
        return

    labels = []
    expected_tools = []
    predicted_tools = []
    for f in failures:
        labels.append(f["input"][:50] + ("..." if len(f["input"]) > 50 else ""))
        expected_tools.append(f["expected"].replace("plot_", ""))
        predicted_tools.append(f["predicted"].replace("plot_", "") if f["predicted"] != "none" else "no tool called")

    fig, ax = plt.subplots(figsize=(10, max(3, len(failures) * 0.8 + 1)))

    y = np.arange(len(failures))
    bar_height = 0.35

    ax.barh(y + bar_height / 2, [1] * len(failures), bar_height,
            color="#2ecc71", alpha=0.7, label="Expected")
    ax.barh(y - bar_height / 2, [1] * len(failures), bar_height,
            color="#e74c3c", alpha=0.7, label="Predicted")

    for i, (exp, pred) in enumerate(zip(expected_tools, predicted_tools)):
        ax.text(0.05, i + bar_height / 2, exp, va="center", fontsize=9, fontweight="bold", color="white")
        ax.text(0.05, i - bar_height / 2, pred, va="center", fontsize=9, fontweight="bold", color="white")

    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlim(0, 1.2)
    ax.set_xticks([])
    ax.set_title("Boundary Case Failures: Expected vs Predicted Tool", fontsize=12, fontweight="bold")
    ax.legend(loc="upper right", fontsize=10)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.invert_yaxis()

    plt.tight_layout()
    out = OUT_DIR / "eval_analysis_boundary.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Saved → {out}")


def plot_tool_coverage(results: dict):
    all_details = results["tool_selection"]["details"] + results["params"]["details"]
    expected_counts = Counter()
    correct_counts = Counter()

    for d in results["tool_selection"]["details"]:
        tool = d["expected"]
        expected_counts[tool] += 1
        if d["match"]:
            correct_counts[tool] += 1

    for d in results["params"]["details"]:
        tool = d["expected_tool"]
        expected_counts[tool] += 1
        if d["match"]:
            correct_counts[tool] += 1

    tools = sorted(expected_counts.keys())
    short_names = [t.replace("plot_", "") for t in tools]
    totals = [expected_counts[t] for t in tools]
    corrects = [correct_counts[t] for t in tools]

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(tools))
    width = 0.4

    ax.bar(x - width / 2, totals, width, label="Total Cases", color="#3498db", alpha=0.7)
    ax.bar(x + width / 2, corrects, width, label="Correct", color="#2ecc71", alpha=0.7)

    for i, (tot, cor) in enumerate(zip(totals, corrects)):
        ax.text(i, max(tot, cor) + 0.2, f"{cor}/{tot}", ha="center", fontsize=9, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(short_names, rotation=35, ha="right", fontsize=9)
    ax.set_ylabel("Number of Cases", fontsize=11)
    ax.set_title("Per-Tool Test Coverage and Accuracy", fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    out = OUT_DIR / "eval_analysis_coverage.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Saved → {out}")


def main():
    results = load_results()
    plot_accuracy_bar(results)
    plot_boundary_confusion(results)
    plot_tool_coverage(results)


if __name__ == "__main__":
    main()
