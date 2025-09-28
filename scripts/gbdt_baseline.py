#!/usr/bin/env python
"""Gradient Boosted Decision Tree baseline with prompt-aware splitting."""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score

from logistic_head_baseline import (
    DEFAULT_SCALAR_FEATURES,
    FORBIDDEN_FEATURES,
    compute_ece,
    load_dataset,
    prepare_arrays,
    split_by_prompt,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="HistGradientBoosting baseline on token features")
    parser.add_argument("features_csv", type=Path, nargs="+", help="One or more token feature CSVs")
    parser.add_argument("--label-column", default="is_correct", help="Label column to predict")
    parser.add_argument(
        "--feature-columns",
        nargs="*",
        default=None,
        help=(
            "Explicit feature columns to use. If omitted and --include-hidden is false, "
            "the curated scalar defaults are used."
        ),
    )
    parser.add_argument(
        "--include-hidden",
        action="store_true",
        help="Append hidden-state features in addition to scalar columns",
    )
    parser.add_argument("--max-depth", type=int, default=3, help="Tree depth for each boosting stage")
    parser.add_argument("--learning-rate", type=float, default=0.05, help="Boosting learning rate")
    parser.add_argument("--max-iter", type=int, default=400, help="Number of boosting iterations")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--split-seed", type=int, default=123, help="Seed for prompt-level split")
    parser.add_argument("--test-frac", type=float, default=0.25, help="Fraction of prompts to hold out")
    parser.add_argument(
        "--test-prompt",
        action="append",
        dest="test_prompts",
        default=None,
        help="Explicit prompt hashes to reserve for testing (may repeat)",
    )
    parser.add_argument("--ece-bins", type=int, default=20, help="Bins for ECE computation")
    parser.add_argument("--metrics-output", type=Path, default=None, help="Optional metrics JSON path")
    parser.add_argument("--save-preds", type=Path, default=None, help="Optional NPZ with predictions")
    parser.add_argument("--print-config", action="store_true", help="Print configuration payload")
    parser.add_argument(
        "--meta-path",
        type=Path,
        nargs="*",
        default=None,
        help="Optional metadata JSON paths matching the input CSVs",
    )
    return parser.parse_args()


def resolve_default_features(df_columns: Sequence[str]) -> List[str]:
    resolved = [c for c in DEFAULT_SCALAR_FEATURES if c in df_columns]
    forbidden = FORBIDDEN_FEATURES.intersection(resolved)
    if forbidden:
        raise ValueError(
            "Default feature set unexpectedly includes forbidden columns: "
            f"{sorted(forbidden)}"
        )
    return resolved


def choose_features(
    df_columns: Sequence[str],
    feature_columns: Optional[Sequence[str]],
    include_hidden: bool,
) -> List[str]:
    selected: List[str] = []
    if feature_columns:
        missing = [c for c in feature_columns if c not in df_columns]
        if missing:
            raise ValueError(f"Requested feature columns missing from dataset: {missing}")
        selected.extend(feature_columns)
    else:
        selected.extend(resolve_default_features(df_columns))

    if include_hidden:
        hidden_cols = [c for c in df_columns if c.startswith("h")]
        if not hidden_cols:
            raise ValueError("No hidden-state columns (h*) found in dataset")
        selected.extend(hidden_cols)

    if not selected:
        raise ValueError("No feature columns selected. Provide --feature-columns or use --include-hidden")

    forbidden = FORBIDDEN_FEATURES.intersection(selected)
    if forbidden:
        raise ValueError(
            "Feature set includes disallowed columns that leak labels: "
            f"{sorted(forbidden)}"
        )

    # Preserve order while dropping duplicates
    seen = set()
    deduped: List[str] = []
    for col in selected:
        if col not in seen:
            deduped.append(col)
            seen.add(col)
    return deduped


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
            "max_depth": args.max_depth,
            "learning_rate": args.learning_rate,
            "max_iter": args.max_iter,
            "seed": args.seed,
            "split_seed": args.split_seed,
            "test_frac": args.test_frac,
            "test_prompts": args.test_prompts,
            "ece_bins": args.ece_bins,
            "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        }
        print(json.dumps(config, indent=2))

    df, prompt_meta = load_dataset(args.features_csv, meta_paths, args.label_column)
    selected_cols = choose_features(df.columns, args.feature_columns, args.include_hidden)

    prompts = df.get("prompt_hash", pd.Series(["unknown"] * len(df))).astype(str).to_numpy()
    train_idx, test_idx, chosen_prompts = split_by_prompt(
        prompts,
        args.test_prompts,
        args.test_frac,
        args.split_seed,
    )

    (
        X_train,
        y_train,
        X_test,
        y_test,
        train_prompts,
        test_prompts,
        _,
    ) = prepare_arrays(
        df,
        args.label_column,
        False,
        train_idx,
        test_idx,
        selected_cols,
        False,
    )

    preview_cols = selected_cols if len(selected_cols) <= 12 else [*selected_cols[:10], "..."]
    print(f"Feature columns ({len(selected_cols)}): {preview_cols}")

    model = HistGradientBoostingClassifier(
        loss="log_loss",
        max_depth=args.max_depth,
        learning_rate=args.learning_rate,
        max_iter=args.max_iter,
        random_state=args.seed,
        l2_regularization=0.1,
    )
    model.fit(X_train, y_train)

    probs_train = model.predict_proba(X_train)[:, 1]
    probs_test = model.predict_proba(X_test)[:, 1]

    train_auc = float(roc_auc_score(y_train, probs_train)) if len(np.unique(y_train)) > 1 else float("nan")
    test_auc = float(roc_auc_score(y_test, probs_test)) if len(np.unique(y_test)) > 1 else float("nan")
    train_brier = float(np.mean((probs_train - y_train) ** 2))
    test_brier = float(np.mean((probs_test - y_test) ** 2))
    ece, bin_centers, empirical_acc, counts = compute_ece(probs_test, y_test, args.ece_bins)

    print(
        f"Prompts: total={len(np.unique(prompts))}, train={len(np.unique(train_prompts))}, test={len(np.unique(test_prompts))}"
    )
    if chosen_prompts:
        print(f"Test prompt hashes: {chosen_prompts}")
    print(
        f"Train examples={len(y_train)}, Test examples={len(y_test)}, "
        f"train_pos_frac={y_train.mean():.3f}, test_pos_frac={y_test.mean():.3f}"
    )
    print(f"AUC (train): {train_auc:.4f}")
    print(f"Brier (train): {train_brier:.4f}")
    print(f"AUC (test): {test_auc:.4f}")
    print(f"Brier (test): {test_brier:.4f}")
    print(f"ECE (test): {ece:.4f}")

    metrics_payload = {
        "auc": test_auc,
        "brier": test_brier,
        "ece": ece,
        "method": "hgb_baseline",
        "label_column": args.label_column,
        "nll_percentile": None,
        "vi_steps": None,
        "lr": None,
        "n_samples": None,
        "max_depth": args.max_depth,
        "learning_rate": args.learning_rate,
        "max_iter": args.max_iter,
        "seed": args.seed,
        "split_seed": args.split_seed,
        "prompt_hash": None,
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "num_examples": int(len(df)),
        "hidden_dim": len([c for c in df.columns if c.startswith("h")]),
        "train_examples": int(len(y_train)),
        "test_examples": int(len(y_test)),
        "train_positive_frac": float(y_train.mean()),
        "test_positive_frac": float(y_test.mean()),
        "test_prompts": chosen_prompts if chosen_prompts else sorted(set(test_prompts)),
        "features_csv": [str(p) for p in args.features_csv],
        "prompt_metadata": prompt_meta,
        "feature_columns": selected_cols,
    }

    if args.metrics_output is not None:
        args.metrics_output.parent.mkdir(parents=True, exist_ok=True)
        args.metrics_output.write_text(json.dumps(metrics_payload, indent=2) + "\n", encoding="utf-8")
        print(f"Saved metrics to {args.metrics_output.resolve()}")

    print("Metrics JSON:")
    print(json.dumps(metrics_payload, indent=2))

    if args.save_preds is not None:
        args.save_preds.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            args.save_preds,
            mean_p=probs_test,
            targets=y_test,
            uncertainty=1.0 - probs_test,
            prompt_hash=np.array(test_prompts, dtype=object),
            bin_centers=bin_centers,
            empirical_acc=empirical_acc,
            counts=counts,
        )
        print(f"Saved predictions to {args.save_preds.resolve()}")


if __name__ == "__main__":
    main()
