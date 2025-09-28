#!/usr/bin/env python
"""Plot risk–coverage curves from prediction files."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np


def load_preds(path: Path) -> Dict[str, np.ndarray]:
    data = np.load(path, allow_pickle=True)
    if "mean_p" not in data or "targets" not in data:
        raise KeyError(f"File {path} missing required arrays 'mean_p' and 'targets'")
    payload = {
        "mean_p": data["mean_p"].astype(np.float64),
        "targets": data["targets"].astype(np.float64),
    }
    if "var_p" in data:
        payload["uncertainty"] = data["var_p"].astype(np.float64)
    elif "uncertainty" in data:
        payload["uncertainty"] = data["uncertainty"].astype(np.float64)
    else:
        payload["uncertainty"] = 1.0 - payload["mean_p"]
    payload["label"] = path.stem
    return payload


def compute_curve(mean_p: np.ndarray, targets: np.ndarray, uncertainty: np.ndarray, steps: int) -> Tuple[np.ndarray, np.ndarray]:
    order = np.argsort(uncertainty)
    mean_p = mean_p[order]
    targets = targets[order]
    coverage = []
    brier_scores = []
    total = mean_p.size

    for frac in np.linspace(1.0 / steps, 1.0, steps):
        k = max(1, int(round(frac * total)))
        idx = slice(0, k)
        brier = float(np.mean((mean_p[idx] - targets[idx]) ** 2))
        coverage.append(k / total)
        brier_scores.append(brier)

    return np.array(coverage), np.array(brier_scores)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot risk–coverage curves")
    parser.add_argument("inputs", type=Path, nargs="+", help="Prediction NPZ files")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("reports/risk_coverage.png"),
        help="Output PNG path",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=10,
        help="Number of coverage points",
    )
    args = parser.parse_args()

    curves = []
    for path in args.inputs:
        preds = load_preds(path)
        coverage, brier = compute_curve(preds["mean_p"], preds["targets"], preds["uncertainty"], args.steps)
        curves.append((preds["label"], coverage, brier))

    fig, ax = plt.subplots(figsize=(6, 5))
    for label, coverage, brier in curves:
        ax.plot(coverage, brier, marker="o", label=label)

    ax.set_xlabel("Coverage")
    ax.set_ylabel("Brier score")
    ax.set_title("Risk–Coverage Curve")
    ax.legend()
    ax.grid(alpha=0.3)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(args.output)
    plt.close(fig)
    print(f"Saved risk–coverage plot to {args.output.resolve()}")


if __name__ == "__main__":
    main()
