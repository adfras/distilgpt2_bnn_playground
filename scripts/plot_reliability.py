#!/usr/bin/env python
"""Create a reliability diagram from posterior predictive samples."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot reliability diagram from preds.npz")
    parser.add_argument(
        "preds_path",
        type=Path,
        nargs="?",
        default=Path("reports/preds.npz"),
        help="Path to npz file with mean_p, var_p, targets arrays",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("reports/reliability.png"),
        help="Output PNG path",
    )
    parser.add_argument(
        "--bins",
        type=int,
        default=10,
        help="Number of equal-width probability bins",
    )
    return parser.parse_args()


def load_predictions(path: Path) -> tuple[np.ndarray, np.ndarray]:
    if not path.exists():
        raise FileNotFoundError(f"Predictions file not found: {path}")
    data = np.load(path)
    if "mean_p" not in data or "targets" not in data:
        raise KeyError("npz file must contain 'mean_p' and 'targets'")
    return data["mean_p"], data["targets"]


def compute_reliability(mean_p: np.ndarray, targets: np.ndarray, bins: int):
    edges = np.linspace(0.0, 1.0, bins + 1)
    bin_ids = np.digitize(mean_p, edges, right=True)
    bin_ids = np.clip(bin_ids, 1, bins)

    bin_centers = []
    empirical_acc = []
    counts = []

    for b in range(1, bins + 1):
        mask = bin_ids == b
        count = mask.sum()
        if count == 0:
            continue
        probs_bin = mean_p[mask]
        targets_bin = targets[mask]
        bin_centers.append(probs_bin.mean())
        empirical_acc.append(targets_bin.mean())
        counts.append(count)

    return np.array(bin_centers), np.array(empirical_acc), np.array(counts)


def plot_reliability(bin_centers: np.ndarray, empirical_acc: np.ndarray, counts: np.ndarray, output: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Perfect calibration")
    ax.plot(bin_centers, empirical_acc, marker="o", label="Bayesian head")
    for x, y, c in zip(bin_centers, empirical_acc, counts):
        ax.text(x, y, str(int(c)), fontsize=8, va="bottom", ha="center")

    ax.set_xlabel("Predicted probability")
    ax.set_ylabel("Empirical accuracy")
    ax.set_title("Reliability Diagram")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend()
    ax.grid(alpha=0.3)

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    mean_p, targets = load_predictions(args.preds_path)
    bin_centers, empirical_acc, counts = compute_reliability(mean_p, targets, args.bins)
    if bin_centers.size == 0:
        raise ValueError("No bins with data; cannot plot reliability.")
    plot_reliability(bin_centers, empirical_acc, counts, args.output)
    print(f"Saved reliability diagram to {args.output.resolve()} (bins plotted: {len(bin_centers)})")


if __name__ == "__main__":
    main()
