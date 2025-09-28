#!/usr/bin/env python
"""Evaluate the base LM probabilities as a calibration baseline."""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression

from logistic_head_baseline import (  # reuse helper utilities
    load_dataset,
    split_by_prompt,
    compute_ece,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate LM probability baseline on token features")
    parser.add_argument(
        "features_csv",
        type=Path,
        nargs="+",
        help="Token feature CSVs",
    )
    parser.add_argument(
        "--label-column",
        default="is_correct",
        help="Label column to evaluate (default: is_correct)",
    )
    parser.add_argument(
        "--prob-column",
        default="p_max",
        help="Probability column from the base LM (default: p_max)",
    )
    parser.add_argument(
        "--split-seed",
        type=int,
        default=123,
        help="Random seed for prompt splitting",
    )
    parser.add_argument(
        "--test-frac",
        type=float,
        default=0.2,
        help="Fraction of prompts to reserve for testing",
    )
    parser.add_argument(
        "--test-prompt",
        action="append",
        dest="test_prompts",
        default=None,
        help="Explicit prompt hashes to reserve for testing",
    )
    parser.add_argument(
        "--ece-bins",
        type=int,
        default=10,
        help="Number of bins for ECE",
    )
    parser.add_argument(
        "--calibration",
        choices=["none", "isotonic", "platt"],
        default="none",
        help="Optional calibration to fit on the training split",
    )
    parser.add_argument(
        "--calibration-input",
        choices=["prob", "log_prob"],
        default="prob",
        help="Input domain for calibration (probability or log probability)",
    )
    parser.add_argument(
        "--save-calibrator",
        type=Path,
        default=None,
        help="Optional path to save the fitted calibrator (CSV for isotonic, JSON for Platt)",
    )
    parser.add_argument(
        "--metrics-output",
        type=Path,
        default=None,
        help="Optional metrics JSON output path",
    )
    parser.add_argument(
        "--save-preds",
        type=Path,
        default=None,
        help="Optional NPZ output for test predictions",
    )
    parser.add_argument(
        "--print-config",
        action="store_true",
        help="Print configuration payload",
    )
    parser.add_argument(
        "--meta-path",
        type=Path,
        nargs="*",
        default=None,
        help="Optional metadata paths for each CSV",
    )
    return parser.parse_args()


def compute_metrics(probs: np.ndarray, targets: np.ndarray) -> Tuple[float, float]:
    try:
        auc = float(roc_auc_score(targets, probs))
    except ValueError:
        auc = float("nan")
    brier = float(np.mean((probs - targets) ** 2))
    return auc, brier


def apply_calibration(
    probs_train: np.ndarray,
    probs_test: np.ndarray,
    targets_train: np.ndarray,
    method: str,
    input_domain: str,
) -> Tuple[np.ndarray, np.ndarray, Optional[dict]]:
    if method == "none":
        return probs_train, probs_test, None

    eps = 1e-12
    if input_domain == "log_prob":
        x_train = np.log(np.clip(probs_train, eps, 1 - eps))
        x_test = np.log(np.clip(probs_test, eps, 1 - eps))
    else:
        x_train = probs_train
        x_test = probs_test

    calibration_info: Optional[dict] = None

    if method == "isotonic":
        iso = IsotonicRegression(out_of_bounds="clip")
        iso.fit(x_train, targets_train)
        probs_train = iso.transform(x_train)
        probs_test = iso.transform(x_test)
        calibration_info = {
            "type": "isotonic",
            "input_domain": input_domain,
            "x": iso.X_thresholds_.tolist(),
            "y": iso.y_thresholds_.tolist(),
        }
    elif method == "platt":
        clf = LogisticRegression()
        clf.fit(x_train.reshape(-1, 1), targets_train)
        probs_train = clf.predict_proba(x_train.reshape(-1, 1))[:, 1]
        probs_test = clf.predict_proba(x_test.reshape(-1, 1))[:, 1]
        calibration_info = {
            "type": "platt",
            "input_domain": input_domain,
            "coef": clf.coef_.ravel().tolist(),
            "intercept": float(clf.intercept_[0]),
        }
    else:
        raise ValueError(f"Unsupported calibration: {method}")

    return probs_train, probs_test, calibration_info


def main() -> None:
    args = parse_args()
    meta_paths = None
    if args.meta_path is not None:
        if len(args.meta_path) not in (0, len(args.features_csv)):
            raise ValueError("Number of --meta-path entries must match features")
        meta_paths = list(args.meta_path)

    if args.print_config:
        config = {
            "features_csv": [str(p) for p in args.features_csv],
            "label_column": args.label_column,
            "prob_column": args.prob_column,
            "split_seed": args.split_seed,
            "test_frac": args.test_frac,
            "test_prompts": args.test_prompts,
            "ece_bins": args.ece_bins,
            "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        }
        print(json.dumps(config, indent=2))

    df, prompt_meta = load_dataset(args.features_csv, meta_paths, args.label_column)

    if args.prob_column not in df.columns:
        raise ValueError(f"Probability column '{args.prob_column}' not found in dataset")

    mask = df[[args.label_column, args.prob_column]].notna().all(axis=1)
    df = df.loc[mask].reset_index(drop=True)

    prompts = df.get("prompt_hash", pd.Series(["unknown"] * len(df))).astype(str).to_numpy()
    train_idx, test_idx, chosen_prompts = split_by_prompt(
        prompts,
        args.test_prompts,
        args.test_frac,
        args.split_seed,
    )

    probs = df[args.prob_column].to_numpy(dtype=np.float32)
    targets = df[args.label_column].astype(np.float32).to_numpy()

    probs_train = probs[train_idx]
    probs_test = probs[test_idx]
    targets_train = targets[train_idx]
    targets_test = targets[test_idx]
    train_prompts = prompts[train_idx]
    test_prompts = prompts[test_idx]

    probs_train, probs_test, calibration_info = apply_calibration(
        probs_train,
        probs_test,
        targets_train,
        args.calibration,
        args.calibration_input,
    )

    if calibration_info is not None:
        print(
            f"Applied {calibration_info['type']} calibration on {calibration_info['input_domain']}"
        )

    auc_train, brier_train = compute_metrics(probs_train, targets_train)
    auc_test, brier_test = compute_metrics(probs_test, targets_test)
    ece, bin_centers, empirical_acc, counts = compute_ece(
        probs_test,
        targets_test,
        args.ece_bins,
    )

    print(
        f"Prompts: total={len(np.unique(prompts))}, train={len(np.unique(train_prompts))}, test={len(np.unique(test_prompts))}"
    )
    if chosen_prompts:
        print(f"Test prompt hashes: {chosen_prompts}")
    print(
        f"Train examples={len(train_idx)}, Test examples={len(test_idx)}, "
        f"train_pos_frac={targets_train.mean():.3f}, test_pos_frac={targets_test.mean():.3f}"
    )
    print(f"AUC (train): {auc_train:.4f}")
    print(f"Brier (train): {brier_train:.4f}")
    print(f"AUC (test): {auc_test:.4f}")
    print(f"Brier (test): {brier_test:.4f}")
    print(f"ECE (test): {ece:.4f}")

    metrics_payload = {
        "auc": auc_test,
        "brier": brier_test,
        "ece": ece,
        "method": "lm_baseline",
        "label_column": args.label_column,
        "prob_column": args.prob_column,
        "seed": None,
        "split_seed": args.split_seed,
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "num_examples": int(len(df)),
        "train_examples": int(len(train_idx)),
        "test_examples": int(len(test_idx)),
        "train_positive_frac": float(targets_train.mean()),
        "test_positive_frac": float(targets_test.mean()),
        "test_prompts": chosen_prompts if chosen_prompts else sorted(set(test_prompts)),
        "features_csv": [str(p) for p in args.features_csv],
        "prompt_metadata": prompt_meta,
        "calibration": calibration_info,
    }

    if calibration_info and args.save_calibrator is not None:
        args.save_calibrator.parent.mkdir(parents=True, exist_ok=True)
        if calibration_info["type"] == "isotonic":
            df_cal = pd.DataFrame(
                {
                    "x_input": calibration_info["x"],
                    "q_correct": calibration_info["y"],
                }
            )
            df_cal.to_csv(args.save_calibrator, index=False)
        else:
            args.save_calibrator.write_text(json.dumps(calibration_info, indent=2) + "\n", encoding="utf-8")
        print(f"Saved calibrator to {args.save_calibrator.resolve()}")

    if args.metrics_output is not None:
        args.metrics_output.parent.mkdir(parents=True, exist_ok=True)
        args.metrics_output.write_text(json.dumps(metrics_payload, indent=2) + "\n", encoding="utf-8")
        print(f"Saved metrics to {args.metrics_output.resolve()}")

    if args.save_preds is not None:
        args.save_preds.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            args.save_preds,
            mean_p=probs_test,
            targets=targets_test,
            uncertainty=1.0 - probs_test,
            prompt_hash=np.array(test_prompts, dtype=object),
            bin_centers=bin_centers,
            empirical_acc=empirical_acc,
            counts=counts,
        )
        print(f"Saved predictions to {args.save_preds.resolve()}")

    print("Metrics JSON:")
    print(json.dumps(metrics_payload, indent=2))


if __name__ == "__main__":
    main()
