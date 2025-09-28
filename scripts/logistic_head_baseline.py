#!/usr/bin/env python
"""Logistic regression baseline on token features with prompt-aware splits."""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

# High-signal scalar features derived from the analysis step. Hidden states are opt-in.
CORE_SCALAR_FEATURES: Sequence[str] = (
    "p_max",
    "log_p_max",
    "logit_gap",
    "entropy",
    "p_margin",
    "p_var",
)

RICH_SCALAR_FEATURES: Sequence[str] = (
    *CORE_SCALAR_FEATURES,
    "p_top2",
    "tail_mass_topk",
    "logit_gap_top3",
    "entropy_prev5_mean",
    "p_max_prev5_mean",
    "p_max_next5_mean",
    "gap_over_entropy_sm",
    "p_max_ma3",
    "p_max_ma5",
    "p_var_ma5",
    "p_var_prev5_mean",
    "p_var_next5_mean",
)

FEATURE_PRESETS = {
    "core": CORE_SCALAR_FEATURES,
    "rich": RICH_SCALAR_FEATURES,
}
FORBIDDEN_FEATURES = {"target_id", "pred_id"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train logistic regression baseline on token features")
    parser.add_argument(
        "features_csv",
        type=Path,
        nargs="+",
        help="One or more token feature CSVs (they can be merged beforehand)",
    )
    parser.add_argument(
        "--label-column",
        default="is_correct",
        help="Name of the label column to predict (default: is_correct)",
    )
    parser.add_argument(
        "--standardise",
        action="store_true",
        help="Z-score features using training split statistics",
    )
    parser.add_argument(
        "--feature-columns",
        nargs="*",
        default=None,
        help=(
            "Explicit feature columns to use. If omitted and --include-hidden is false, "
            "all hidden-state columns (h*) are used."
        ),
    )
    parser.add_argument(
        "--feature-preset",
        choices=tuple(FEATURE_PRESETS.keys()),
        default="rich",
        help=(
            "Named preset of scalar features to use when --feature-columns is not provided. "
            "'core' sticks to six universally strong logit-derived signals, while 'rich' "
            "adds wider context features."
        ),
    )
    parser.add_argument(
        "--include-hidden",
        action="store_true",
        help="Append hidden-state features in addition to any --feature-columns provided",
    )
    parser.add_argument(
        "--c",
        type=float,
        default=1.0,
        help="Inverse regularisation strength (C) for LogisticRegression",
    )
    parser.add_argument(
        "--max-iter",
        type=int,
        default=1000,
        help="Maximum iterations for solver",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--split-seed",
        type=int,
        default=123,
        help="Random seed for prompt-based splitting",
    )
    parser.add_argument(
        "--test-frac",
        type=float,
        default=0.2,
        help="Fraction of prompts to hold out for testing (rounded to at least one)",
    )
    parser.add_argument(
        "--test-prompt",
        action="append",
        dest="test_prompts",
        default=None,
        help="Explicit prompt_hash values to reserve for testing (can be given multiple times)",
    )
    parser.add_argument(
        "--ece-bins",
        type=int,
        default=10,
        help="Number of bins for Expected Calibration Error",
    )
    parser.add_argument(
        "--metrics-output",
        type=Path,
        default=None,
        help="Optional output path for metrics JSON",
    )
    parser.add_argument(
        "--save-preds",
        type=Path,
        default=None,
        help="Optional path to save test predictions (npz)",
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
        help="Optional metadata JSON paths for each CSV (defaults to <csv>.meta.json)",
    )
    parser.add_argument(
        "--coral-align",
        action="store_true",
        help="Apply CORAL feature alignment from test to training covariance",
    )
    parser.add_argument(
        "--coral-eps",
        type=float,
        default=1e-4,
        help="Regularisation term for CORAL covariance matrices",
    )
    parser.add_argument(
        "--temp-calibrate-frac",
        type=float,
        default=0.0,
        help="Fraction of target examples used for temperature scaling calibration",
    )
    parser.add_argument(
        "--temp-calibrate-max-iter",
        type=int,
        default=100,
        help="Maximum iterations for temperature scaling optimisation",
    )
    parser.add_argument(
        "--temp-calibrate-lr",
        type=float,
        default=0.05,
        help="Learning rate in log-temperature space during calibration",
    )
    parser.add_argument(
        "--conformal-alpha",
        type=float,
        default=0.0,
        help="Risk level α for conformal selective prediction (0 disables)",
    )
    parser.add_argument(
        "--conformal-calib-frac",
        type=float,
        default=0.2,
        help="Fraction of target examples reserved for conformal calibration",
    )
    return parser.parse_args()


def load_meta(csv_path: Path, meta_paths: List[Path | None], idx: int) -> dict:
    meta_path = None
    if meta_paths:
        meta_path = meta_paths[idx]
    if meta_path is None:
        meta_path = csv_path.with_suffix(".meta.json")
    if not meta_path.exists():
        return {}
    with meta_path.open(encoding="utf-8") as fh:
        return json.load(fh)


def load_dataset(
    csv_paths: List[Path],
    meta_paths: List[Path | None] | None,
    label_column: str,
) -> Tuple[pd.DataFrame, List[dict]]:
    frames = []
    prompt_meta: List[dict] = []
    for idx, csv_path in enumerate(csv_paths):
        if not csv_path.exists():
            raise FileNotFoundError(f"Features CSV not found: {csv_path}")
        df = pd.read_csv(csv_path)
        meta = load_meta(csv_path, meta_paths or [], idx)
        prompt_hash = meta.get("prompt_hash")
        if "prompt_hash" not in df.columns:
            df["prompt_hash"] = prompt_hash
        if "prompt" not in df.columns:
            df["prompt"] = meta.get("prompt")
        frames.append(df)
        prompt_meta.append({"source_csv": str(csv_path), **meta})
    merged = pd.concat(frames, ignore_index=True)
    if label_column not in merged.columns:
        raise ValueError(f"Label column '{label_column}' not found in dataset")
    return merged, prompt_meta


def resolve_feature_columns(
    df: pd.DataFrame,
    feature_columns: Optional[Sequence[str]],
    include_hidden: bool,
    feature_preset: str,
) -> List[str]:
    hidden_cols = [c for c in df.columns if c.startswith("h")]
    selected: List[str] = []

    if feature_columns:
        missing = [c for c in feature_columns if c not in df.columns]
        if missing:
            raise ValueError(f"Requested feature columns missing from dataset: {missing}")
        selected.extend(feature_columns)
    else:
        preset_columns = FEATURE_PRESETS.get(feature_preset)
        if preset_columns is None:
            raise ValueError(f"Unknown feature preset: {feature_preset}")
        default_available = [c for c in preset_columns if c in df.columns]
        if default_available:
            selected.extend(default_available)

    if include_hidden:
        if not hidden_cols:
            raise ValueError("No hidden-state columns (h*) found in dataset")
        selected.extend(hidden_cols)

    if not selected:
        if hidden_cols:
            selected.extend(hidden_cols)
        else:
            raise ValueError("No usable feature columns found; supply --feature-columns explicitly")

    forbidden = FORBIDDEN_FEATURES.intersection(selected)
    if forbidden:
        raise ValueError(
            "Feature set includes disallowed columns that leak labels: "
            f"{sorted(forbidden)}"
        )

    seen = set()
    deduped: List[str] = []
    for col in selected:
        if col not in seen:
            deduped.append(col)
            seen.add(col)
    return deduped


def prepare_arrays(
    df: pd.DataFrame,
    label_column: str,
    standardise: bool,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    feature_columns: Optional[Sequence[str]],
    include_hidden: bool,
    feature_preset: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[str], List[str], List[str]]:
    selected_cols = resolve_feature_columns(df, feature_columns, include_hidden, feature_preset)

    mask = df[label_column].notna()
    if selected_cols:
        mask &= df[selected_cols].notna().all(axis=1)

    mask_array = mask.to_numpy()
    valid_indices = np.flatnonzero(mask_array)
    if valid_indices.size == 0:
        raise ValueError("No rows remain after filtering NaNs in labels/features")

    index_map = np.full(len(df), -1, dtype=int)
    index_map[valid_indices] = np.arange(valid_indices.size)

    train_idx_mapped = index_map[train_idx]
    test_idx_mapped = index_map[test_idx]
    train_idx_mapped = train_idx_mapped[train_idx_mapped >= 0]
    test_idx_mapped = test_idx_mapped[test_idx_mapped >= 0]

    df = df.loc[mask].reset_index(drop=True)
    if train_idx_mapped.size == 0 or test_idx_mapped.size == 0:
        raise ValueError(
            "Train/test split became empty after filtering rows; "
            "check selected feature columns and prompt split."
        )

    features = df[selected_cols].to_numpy(dtype=np.float32)
    labels = df[label_column].astype(np.float32).to_numpy()
    prompts = df.get("prompt_hash", pd.Series(["unknown"] * len(df))).astype(str).to_numpy()

    if standardise:
        scaler = StandardScaler()
        scaler.fit(features[train_idx_mapped])
        features = scaler.transform(features)

    X_train = features[train_idx_mapped]
    y_train = labels[train_idx_mapped]
    X_test = features[test_idx_mapped]
    y_test = labels[test_idx_mapped]
    train_prompts = prompts[train_idx_mapped]
    test_prompts = prompts[test_idx_mapped]
    return X_train, y_train, X_test, y_test, train_prompts.tolist(), test_prompts.tolist(), selected_cols


def split_by_prompt(
    prompts: np.ndarray,
    test_prompts: List[str] | None,
    test_frac: float,
    split_seed: int,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    rng = np.random.default_rng(split_seed)
    unique_prompts = np.unique(prompts)

    if test_prompts:
        test_prompts_set = {p for p in test_prompts}
    elif unique_prompts.size > 1:
        shuffled = unique_prompts.copy()
        rng.shuffle(shuffled)
        holdout_count = max(1, int(round(test_frac * unique_prompts.size)))
        test_prompts_set = set(shuffled[:holdout_count])
    else:
        # Fallback to token-level split when only one prompt exists
        indices = np.arange(prompts.size)
        rng.shuffle(indices)
        split = max(1, int(0.2 * len(indices)))
        test_idx = indices[:split]
        train_idx = indices[split:]
        return train_idx, test_idx, []

    test_mask = np.isin(prompts, list(test_prompts_set))
    if not test_mask.any():
        raise ValueError("Specified test prompts produced no test examples")
    if test_mask.all():
        raise ValueError("Test prompts consumed all examples; adjust selection")

    test_idx = np.nonzero(test_mask)[0]
    train_idx = np.nonzero(~test_mask)[0]
    return train_idx, test_idx, sorted(test_prompts_set)


def compute_metrics(probs: np.ndarray, targets: np.ndarray) -> Dict[str, float]:
    try:
        auc = float(roc_auc_score(targets, probs))
    except ValueError:
        auc = float("nan")
    brier = float(np.mean((probs - targets) ** 2))
    return {"auc": auc, "brier": brier}


def compute_ece(probs: np.ndarray, targets: np.ndarray, bins: int) -> Tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    edges = np.linspace(0.0, 1.0, bins + 1)
    bin_ids = np.digitize(probs, edges, right=True)
    bin_ids = np.clip(bin_ids, 1, bins)

    bin_centers: List[float] = []
    empirical_acc: List[float] = []
    counts: List[int] = []
    ece = 0.0
    total = probs.size

    for b in range(1, bins + 1):
        mask = bin_ids == b
        count = int(mask.sum())
        if count == 0:
            continue
        p_bin = float(probs[mask].mean())
        acc_bin = float(targets[mask].mean())
        bin_centers.append(p_bin)
        empirical_acc.append(acc_bin)
        counts.append(count)
        ece += abs(acc_bin - p_bin) * (count / total)

    return ece, np.array(bin_centers), np.array(empirical_acc), np.array(counts)


def _safe_sigmoid(z: np.ndarray) -> np.ndarray:
    clipped = np.clip(z, -80.0, 80.0)
    return 1.0 / (1.0 + np.exp(-clipped))


def _matrix_power(mat: np.ndarray, power: float, eps: float) -> np.ndarray:
    eigvals, eigvecs = np.linalg.eigh(mat)
    eigvals = np.clip(eigvals, eps, None)
    powered = eigvecs @ np.diag(eigvals ** power) @ eigvecs.T
    return powered.astype(mat.dtype, copy=False)


def apply_coral_alignment(
    source: np.ndarray,
    target: np.ndarray,
    eps: float,
) -> np.ndarray:
    if target.shape[0] < 2:
        return target
    source_mean = source.mean(axis=0, keepdims=True)
    target_mean = target.mean(axis=0, keepdims=True)
    source_centered = source - source_mean
    target_centered = target - target_mean

    cov_source = np.cov(source_centered, rowvar=False, bias=False)
    cov_target = np.cov(target_centered, rowvar=False, bias=False)
    dim = source.shape[1]
    cov_source += eps * np.eye(dim, dtype=source.dtype)
    cov_target += eps * np.eye(dim, dtype=source.dtype)

    whitening = _matrix_power(cov_target, -0.5, eps)
    coloring = _matrix_power(cov_source, 0.5, eps)
    aligned = target_centered @ whitening @ coloring + source_mean
    return aligned


def temperature_scale_logits(
    logits: np.ndarray,
    targets: np.ndarray,
    frac: float,
    max_iter: int,
    lr: float,
    seed: int,
) -> float:
    if frac <= 0.0 or logits.size == 0:
        return 1.0
    n_examples = logits.shape[0]
    calib_size = max(1, int(round(frac * n_examples)))
    rng = np.random.default_rng(seed)
    idx = rng.choice(n_examples, size=calib_size, replace=False)
    logits_calib = logits[idx]
    targets_calib = targets[idx]

    tau = 0.0
    for _ in range(max_iter):
        temperature = np.exp(tau)
        probs = _safe_sigmoid(logits_calib / temperature)
        grad_tau = -(np.sum((probs - targets_calib) * logits_calib) / temperature)
        tau -= lr * grad_tau
    temperature = float(np.exp(tau))
    return float(np.clip(temperature, 0.1, 10.0))


def conformal_threshold(scores: np.ndarray, alpha: float) -> float:
    if scores.size == 0:
        return 1.0
    sorted_scores = np.sort(scores)
    k = int(np.ceil((scores.size + 1) * (1.0 - alpha))) - 1
    k = int(np.clip(k, 0, scores.size - 1))
    return float(sorted_scores[k])


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
            "standardise": args.standardise,
            "c": args.c,
            "max_iter": args.max_iter,
            "seed": args.seed,
            "split_seed": args.split_seed,
            "test_frac": args.test_frac,
            "test_prompts": args.test_prompts,
            "ece_bins": args.ece_bins,
            "feature_preset": args.feature_preset,
            "coral_align": args.coral_align,
            "coral_eps": args.coral_eps,
            "temp_calibrate_frac": args.temp_calibrate_frac,
            "temp_calibrate_max_iter": args.temp_calibrate_max_iter,
            "temp_calibrate_lr": args.temp_calibrate_lr,
            "conformal_alpha": args.conformal_alpha,
            "conformal_calib_frac": args.conformal_calib_frac,
            "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        }
        print(json.dumps(config, indent=2))

    df, prompt_meta = load_dataset(args.features_csv, meta_paths, args.label_column)
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
        selected_cols,
    ) = prepare_arrays(
        df,
        args.label_column,
        args.standardise,
        train_idx,
        test_idx,
        args.feature_columns,
        args.include_hidden,
        args.feature_preset,
    )

    if args.coral_align:
        X_test = apply_coral_alignment(X_train, X_test, args.coral_eps)

    preview_cols = selected_cols if len(selected_cols) <= 12 else [*selected_cols[:10], "..."]
    print(f"Feature columns ({len(selected_cols)}): {preview_cols}")

    model = LogisticRegression(
        C=args.c,
        max_iter=args.max_iter,
        random_state=args.seed,
        solver="lbfgs",
        n_jobs=None,
    )
    model.fit(X_train, y_train)

    logits_train = model.decision_function(X_train)
    logits_test = model.decision_function(X_test)

    temperature = temperature_scale_logits(
        logits_test,
        y_test,
        args.temp_calibrate_frac,
        args.temp_calibrate_max_iter,
        args.temp_calibrate_lr,
        args.seed,
    ) if args.temp_calibrate_frac > 0.0 else 1.0

    probs_train = _safe_sigmoid(logits_train / temperature)
    probs_test = _safe_sigmoid(logits_test / temperature)

    train_metrics = compute_metrics(probs_train, y_train)
    test_metrics = compute_metrics(probs_test, y_test)
    ece, bin_centers, empirical_acc, counts = compute_ece(probs_test, y_test, args.ece_bins)

    conformal_threshold_value = None
    conformal_coverage = None
    conformal_brier = None
    if args.conformal_alpha > 0.0 and y_test.size > 0:
        rng = np.random.default_rng(args.seed)
        n_calib = max(1, int(round(args.conformal_calib_frac * y_test.size)))
        n_calib = min(n_calib, y_test.size)
        calib_indices = rng.choice(y_test.size, size=n_calib, replace=False)
        confidence_calib = np.maximum(probs_test[calib_indices], 1.0 - probs_test[calib_indices])
        scores_calib = 1.0 - confidence_calib
        conformal_threshold_value = conformal_threshold(scores_calib, args.conformal_alpha)

        confidence_all = np.maximum(probs_test, 1.0 - probs_test)
        scores_all = 1.0 - confidence_all
        accept_mask = scores_all <= conformal_threshold_value
        conformal_coverage = float(accept_mask.mean())
        if accept_mask.any():
            conformal_brier = float(np.mean((probs_test[accept_mask] - y_test[accept_mask]) ** 2))


    print(
        f"Prompts: total={len(np.unique(prompts))}, train={len(np.unique(train_prompts))}, test={len(np.unique(test_prompts))}"
    )
    if chosen_prompts:
        print(f"Test prompt hashes: {chosen_prompts}")
    print(
        f"Train examples={len(y_train)}, Test examples={len(y_test)}, "
        f"train_pos_frac={y_train.mean():.3f}, test_pos_frac={y_test.mean():.3f}"
    )
    print(f"AUC (train): {train_metrics['auc']:.4f}")
    print(f"Brier (train): {train_metrics['brier']:.4f}")
    print(f"AUC (test): {test_metrics['auc']:.4f}")
    print(f"Brier (test): {test_metrics['brier']:.4f}")
    print(f"ECE (test): {ece:.4f}")
    if conformal_threshold_value is not None:
        brier_msg = f"{conformal_brier:.4f}" if conformal_brier is not None else "nan"
        print(
            f"Conformal selective: alpha={args.conformal_alpha:.3f}, threshold={conformal_threshold_value:.4f}, "
            f"coverage={conformal_coverage:.3f}, Brier@accept={brier_msg}"
        )

    metrics_payload = {
        "auc": test_metrics["auc"],
        "brier": test_metrics["brier"],
        "ece": ece,
        "method": "logistic_baseline",
        "label_column": args.label_column,
        "nll_percentile": None,
        "vi_steps": None,
        "lr": None,
        "n_samples": None,
        "c": args.c,
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
        "feature_preset": args.feature_preset,
        "coral_align": args.coral_align,
        "coral_eps": args.coral_eps,
        "temperature": temperature,
        "temp_calibrate_frac": args.temp_calibrate_frac,
        "conformal_alpha": args.conformal_alpha,
        "conformal_threshold": conformal_threshold_value,
        "conformal_coverage": conformal_coverage,
        "conformal_brier": conformal_brier,
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
