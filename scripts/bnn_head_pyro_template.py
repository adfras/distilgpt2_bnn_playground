#!/usr/bin/env python
"""Bayesian logistic regression head (Pyro VI) with prompt-aware splits."""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import pyro
import pyro.distributions as dist
import torch
import torch.nn.functional as F
from pyro.infer import SVI, Trace_ELBO
from pyro.optim import Adam
from sklearn.preprocessing import StandardScaler
from torch.distributions import constraints
from sklearn.metrics import roc_auc_score

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
    parser = argparse.ArgumentParser(description="Train Bayesian logistic head on token features")
    parser.add_argument(
        "features_csv",
        type=Path,
        nargs="+",
        help="One or more token feature CSVs",
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
            "Named preset of scalar features when --feature-columns is not provided. "
            "Use 'core' for a conservative six-feature set, 'rich' to include wider context signals."
        ),
    )
    parser.add_argument(
        "--include-hidden",
        action="store_true",
        help="Append hidden-state features in addition to any --feature-columns provided",
    )
    parser.add_argument(
        "--vi-steps",
        type=int,
        default=2000,
        help="Number of SVI optimisation steps",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=5e-3,
        help="Learning rate for the Adam optimiser",
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=None,
        help="DEPRECATED alias for --eval-samples; kept for backwards compatibility",
    )
    parser.add_argument(
        "--eval-samples",
        type=int,
        default=800,
        help="Number of posterior samples when estimating metrics",
    )
    parser.add_argument(
        "--train-particles",
        type=int,
        default=2,
        help="Number of MC particles for the ELBO during optimisation",
    )
    parser.add_argument(
        "--use-ard",
        action="store_true",
        help="Enable automatic relevance determination (feature-wise variance priors)",
    )
    parser.add_argument(
        "--ard-shape",
        type=float,
        default=2.0,
        help="Shape parameter for Gamma prior on precisions in ARD",
    )
    parser.add_argument(
        "--ard-rate",
        type=float,
        default=1.0,
        help="Rate parameter for Gamma prior on precisions in ARD",
    )
    parser.add_argument(
        "--guide-type",
        choices=("meanfield", "lowrank"),
        default="meanfield",
        help="Variational family for weights (meanfield diag or low-rank + diagonal)",
    )
    parser.add_argument(
        "--guide-rank",
        type=int,
        default=5,
        help="Rank of the low-rank factor when --guide-type=lowrank",
    )
    parser.add_argument(
        "--weight-scale",
        type=float,
        default=0.3,
        help="Prior standard deviation for weights",
    )
    parser.add_argument(
        "--bias-scale",
        type=float,
        default=1.0,
        help="Prior standard deviation for bias",
    )
    parser.add_argument(
        "--coral-align",
        action="store_true",
        help="Apply CORAL alignment of test features toward the training covariance",
    )
    parser.add_argument(
        "--coral-eps",
        type=float,
        default=1e-4,
        help="Regularisation term for CORAL covariance matrices",
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
        help="Explicit prompt_hash values to reserve for testing",
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
        help="Optional path to write metrics JSON",
    )
    parser.add_argument(
        "--save-preds",
        type=Path,
        default=None,
        help="Optional path to save test posterior predictions (npz)",
    )
    parser.add_argument(
        "--save-train-preds",
        type=Path,
        default=None,
        help="Optional path to save training posterior predictions (npz)",
    )
    parser.add_argument(
        "--print-config",
        action="store_true",
        help="Print parsed arguments as JSON",
    )
    parser.add_argument(
        "--meta-path",
        type=Path,
        nargs="*",
        default=None,
        help="Optional metadata JSON paths for each CSV",
    )
    parser.add_argument(
        "--temp-calibrate-frac",
        type=float,
        default=0.0,
        help="Fraction of target (test) examples to use for temperature scaling calibration (0 disables)",
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
        help="Learning rate in log-temperature space for calibration",
    )
    parser.add_argument(
        "--conformal-alpha",
        type=float,
        default=0.0,
        help="Risk level α for conformal selective prediction",
    )
    parser.add_argument(
        "--conformal-calib-frac",
        type=float,
        default=0.2,
        help="Fraction of target examples used to fit the conformal threshold",
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
        indices = np.arange(prompts.size)
        rng.shuffle(indices)
        split = max(1, int(0.2 * len(indices)))
        return indices[split:], indices[:split], []

    test_mask = np.isin(prompts, list(test_prompts_set))
    if not test_mask.any():
        raise ValueError("Specified test prompts produced no test examples")
    if test_mask.all():
        raise ValueError("Test prompts consumed all examples; adjust selection")

    test_idx = np.nonzero(test_mask)[0]
    train_idx = np.nonzero(~test_mask)[0]
    return train_idx, test_idx, sorted(test_prompts_set)


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


def prepare_tensors(
    df: pd.DataFrame,
    label_column: str,
    standardise: bool,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    feature_columns: Optional[Sequence[str]],
    include_hidden: bool,
    feature_preset: str,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, List[str], List[str], List[str]]:
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

    X_train = torch.from_numpy(features[train_idx_mapped])
    y_train = torch.from_numpy(labels[train_idx_mapped])
    X_test = torch.from_numpy(features[test_idx_mapped])
    y_test = torch.from_numpy(labels[test_idx_mapped])
    train_prompts = prompts[train_idx_mapped]
    test_prompts = prompts[test_idx_mapped]
    return X_train, y_train, X_test, y_test, train_prompts.tolist(), test_prompts.tolist(), selected_cols


def logistic_model(
    X: torch.Tensor,
    y: torch.Tensor | None,
    weight_scale: float,
    bias_scale: float,
    use_ard: bool,
    ard_shape: float,
    ard_rate: float,
) -> None:
    num_features = X.shape[1]
    if use_ard:
        with pyro.plate("feature_plate", num_features):
            weight_precision = pyro.sample(
                "weight_precision",
                dist.Gamma(
                    X.new_full((num_features,), ard_shape),
                    X.new_full((num_features,), ard_rate),
                ),
            )
        weight_prior_scale = torch.rsqrt(weight_precision)
    else:
        global_scale = pyro.sample(
            "global_scale",
            dist.HalfNormal(X.new_tensor(weight_scale))
        )
        with pyro.plate("feature_plate", num_features):
            local_scale = pyro.sample(
                "weight_scale_local",
                dist.HalfNormal(X.new_tensor(1.0))
            )
        weight_prior_scale = global_scale * local_scale

    weight_prior_loc = X.new_zeros(num_features)
    bias_prior_loc = X.new_tensor(0.0)
    bias_prior_scale = X.new_tensor(bias_scale)

    weight = pyro.sample(
        "weight",
        dist.Normal(weight_prior_loc, weight_prior_scale).to_event(1),
    )
    bias = pyro.sample("bias", dist.Normal(bias_prior_loc, bias_prior_scale))

    logits = X.matmul(weight) + bias
    with pyro.plate("data", X.shape[0]):
        pyro.sample("obs", dist.Bernoulli(logits=logits), obs=y)


def logistic_guide(
    X: torch.Tensor,
    y: torch.Tensor | None,
    weight_scale: float,
    bias_scale: float,
    use_ard: bool,
    ard_shape: float,
    ard_rate: float,
    guide_type: str,
    guide_rank: int,
) -> None:
    num_features = X.shape[1]
    if use_ard:
        conc_param = pyro.param(
            "q_weight_precision_concentration",
            X.new_full((num_features,), ard_shape),
            constraint=constraints.positive,
        )
        rate_param = pyro.param(
            "q_weight_precision_rate",
            X.new_full((num_features,), ard_rate),
            constraint=constraints.positive,
        )
        with pyro.plate("feature_plate", num_features):
            pyro.sample(
                "weight_precision",
                dist.Gamma(conc_param, rate_param),
            )
    else:
        global_loc = pyro.param(
            "q_global_scale_loc",
            torch.log(X.new_tensor(weight_scale))
        )
        global_scale_param = pyro.param(
            "q_global_scale_scale",
            X.new_tensor(0.25),
            constraint=constraints.positive,
        )
        pyro.sample("global_scale", dist.LogNormal(global_loc, global_scale_param))

        local_loc = pyro.param(
            "q_weight_scale_local_loc",
            X.new_zeros(num_features)
        )
        local_scale_param = pyro.param(
            "q_weight_scale_local_scale",
            X.new_full((num_features,), 0.25),
            constraint=constraints.positive,
        )
        with pyro.plate("feature_plate", num_features):
            pyro.sample(
                "weight_scale_local",
                dist.LogNormal(local_loc, local_scale_param),
            )

    weight_loc = pyro.param("q_weight_loc", X.new_zeros(num_features))
    if guide_type == "meanfield":
        weight_scale_param = pyro.param(
            "q_weight_scale",
            X.new_full((num_features,), weight_scale),
            constraint=constraints.positive,
        )
        weight_dist = dist.Normal(weight_loc, weight_scale_param).to_event(1)
    elif guide_type == "lowrank":
        rank = max(1, min(guide_rank, num_features))
        cov_factor = pyro.param(
            "q_weight_cov_factor",
            X.new_zeros(num_features, rank),
        )
        cov_diag = pyro.param(
            "q_weight_cov_diag",
            X.new_full((num_features,), 0.1),
            constraint=constraints.positive,
        )
        weight_dist = dist.LowRankMultivariateNormal(weight_loc, cov_factor, cov_diag)
    else:
        raise ValueError(f"Unsupported guide_type: {guide_type}")
    bias_loc = pyro.param("q_bias_loc", X.new_tensor(0.0))
    bias_scale = pyro.param(
        "q_bias_scale",
        X.new_tensor(bias_scale),
        constraint=constraints.positive,
    )

    pyro.sample("weight", weight_dist)
    pyro.sample("bias", dist.Normal(bias_loc, bias_scale))


def fit_vi(
    X: torch.Tensor,
    y: torch.Tensor,
    lr: float,
    steps: int,
    weight_scale: float,
    bias_scale: float,
    use_ard: bool,
    ard_shape: float,
    ard_rate: float,
    guide_type: str,
    guide_rank: int,
    train_particles: int,
) -> List[float]:
    pyro.clear_param_store()
    optimiser = Adam({"lr": lr})
    svi = SVI(
        lambda X_, y_: logistic_model(X_, y_, weight_scale, bias_scale, use_ard, ard_shape, ard_rate),
        lambda X_, y_: logistic_guide(
            X_,
            y_,
            weight_scale,
            bias_scale,
            use_ard,
            ard_shape,
            ard_rate,
            guide_type,
            guide_rank,
        ),
        optimiser,
        loss=Trace_ELBO(num_particles=train_particles),
    )

    losses: List[float] = []
    for step in range(1, steps + 1):
        loss = svi.step(X, y)
        losses.append(loss / X.shape[0])
        if step % max(steps // 10, 1) == 0 or step == 1:
            print(f"VI step {step:04d}/{steps}: loss = {losses[-1]:.4f}")
    return losses


def sample_posterior_predictions(
    X: torch.Tensor,
    n_samples: int,
    guide_type: str,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    weight_loc = pyro.param("q_weight_loc").detach()
    bias_loc = pyro.param("q_bias_loc").detach()
    bias_scale = pyro.param("q_bias_scale").detach()

    if guide_type == "meanfield":
        weight_scale = pyro.param("q_weight_scale").detach()
        weight_dist = dist.Normal(weight_loc, weight_scale)
    elif guide_type == "lowrank":
        cov_factor = pyro.param("q_weight_cov_factor").detach()
        cov_diag = pyro.param("q_weight_cov_diag").detach()
        weight_dist = dist.LowRankMultivariateNormal(weight_loc, cov_factor, cov_diag)
    else:
        raise ValueError(f"Unsupported guide_type for posterior sampling: {guide_type}")

    bias_dist = dist.Normal(bias_loc, bias_scale)

    weight_samples = weight_dist.rsample((n_samples,))
    if guide_type == "lowrank":
        weight_samples = weight_samples
    bias_samples = bias_dist.rsample((n_samples,))

    logits = weight_samples.matmul(X.T) + bias_samples.unsqueeze(1)
    probs = torch.sigmoid(logits)
    mean_p = probs.mean(dim=0)
    var_p = probs.var(dim=0, unbiased=False)
    return mean_p, var_p, logits


def calibrate_temperature(
    logits_mean: torch.Tensor,
    targets: torch.Tensor,
    frac: float,
    max_iter: int,
    lr: float,
    seed: int,
) -> float:
    if frac <= 0.0 or logits_mean.numel() == 0:
        return 1.0

    num_examples = logits_mean.numel()
    calib_size = max(1, int(round(frac * num_examples)))
    rng = np.random.default_rng(seed)
    idx = rng.choice(num_examples, size=calib_size, replace=False)
    logits_calib = logits_mean[idx]
    targets_calib = targets[idx]

    tau = torch.zeros(1, dtype=logits_mean.dtype, requires_grad=True, device=logits_mean.device)
    optimiser = torch.optim.SGD([tau], lr=lr)

    for _ in range(max_iter):
        optimiser.zero_grad()
        temperature = torch.exp(tau)
        scaled_probs = torch.sigmoid(logits_calib / temperature)
        loss = F.binary_cross_entropy(scaled_probs, targets_calib)
        loss.backward()
        optimiser.step()

    return float(torch.exp(tau.detach()).cpu().item())


def apply_coral_alignment_torch(
    source: torch.Tensor,
    target: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    if target.shape[0] < 2:
        return target
    source_mean = source.mean(dim=0, keepdim=True)
    target_mean = target.mean(dim=0, keepdim=True)
    source_centered = source - source_mean
    target_centered = target - target_mean

    cov_source = source_centered.T @ source_centered / max(source_centered.shape[0] - 1, 1)
    cov_target = target_centered.T @ target_centered / max(target_centered.shape[0] - 1, 1)
    eye = torch.eye(source.shape[1], device=source.device, dtype=source.dtype)
    cov_source = cov_source + eps * eye
    cov_target = cov_target + eps * eye

    evals_t, evecs_t = torch.linalg.eigh(cov_target)
    evals_s, evecs_s = torch.linalg.eigh(cov_source)
    evals_t = torch.clamp(evals_t, min=eps)
    evals_s = torch.clamp(evals_s, min=eps)
    cov_target_inv_sqrt = evecs_t @ torch.diag(evals_t.pow(-0.5)) @ evecs_t.T
    cov_source_sqrt = evecs_s @ torch.diag(evals_s.pow(0.5)) @ evecs_s.T

    aligned = target_centered @ cov_target_inv_sqrt @ cov_source_sqrt + source_mean
    return aligned


def conformal_threshold(scores: np.ndarray, alpha: float) -> float:
    if scores.size == 0:
        return 1.0
    sorted_scores = np.sort(scores)
    k = int(np.ceil((scores.size + 1) * (1.0 - alpha))) - 1
    k = int(np.clip(k, 0, scores.size - 1))
    return float(sorted_scores[k])


def compute_metrics(probs: torch.Tensor, targets: torch.Tensor) -> Dict[str, float]:
    probs_np = probs.detach().cpu().numpy()
    targets_np = targets.detach().cpu().numpy()
    try:
        auc = float(roc_auc_score(targets_np, probs_np))
    except ValueError:
        auc = float("nan")
    brier = float(np.mean((probs_np - targets_np) ** 2))
    return {"auc": auc, "brier": brier}


def compute_ece(probs: torch.Tensor, targets: torch.Tensor, bins: int) -> Tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    probs_np = probs.detach().cpu().numpy()
    targets_np = targets.detach().cpu().numpy()
    edges = np.linspace(0.0, 1.0, bins + 1)
    bin_ids = np.digitize(probs_np, edges, right=True)
    bin_ids = np.clip(bin_ids, 1, bins)

    bin_centers: List[float] = []
    empirical_acc: List[float] = []
    counts: List[int] = []
    total = probs_np.size
    ece = 0.0

    for b in range(1, bins + 1):
        mask = bin_ids == b
        count = mask.sum()
        if count == 0:
            continue
        p_bin = float(probs_np[mask].mean())
        acc_bin = float(targets_np[mask].mean())
        bin_centers.append(p_bin)
        empirical_acc.append(acc_bin)
        counts.append(int(count))
        ece += abs(acc_bin - p_bin) * (count / total)

    return ece, np.array(bin_centers), np.array(empirical_acc), np.array(counts)


def main() -> None:
    args = parse_args()
    if args.n_samples is not None:
        args.eval_samples = args.n_samples
    pyro.set_rng_seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

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
            "vi_steps": args.vi_steps,
            "lr": args.lr,
            "eval_samples": args.eval_samples,
            "train_particles": args.train_particles,
            "use_ard": args.use_ard,
            "ard_shape": args.ard_shape,
            "ard_rate": args.ard_rate,
            "guide_type": args.guide_type,
            "guide_rank": args.guide_rank,
            "coral_align": args.coral_align,
            "coral_eps": args.coral_eps,
            "temp_calibrate_frac": args.temp_calibrate_frac,
            "temp_calibrate_max_iter": args.temp_calibrate_max_iter,
            "temp_calibrate_lr": args.temp_calibrate_lr,
            "conformal_alpha": args.conformal_alpha,
            "conformal_calib_frac": args.conformal_calib_frac,
            "seed": args.seed,
            "split_seed": args.split_seed,
            "test_frac": args.test_frac,
            "test_prompts": args.test_prompts,
            "ece_bins": args.ece_bins,
            "feature_preset": args.feature_preset,
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
    ) = prepare_tensors(
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
        X_test = apply_coral_alignment_torch(X_train, X_test, args.coral_eps)

    preview_cols = selected_cols if len(selected_cols) <= 12 else [*selected_cols[:10], "..."]
    print(f"Feature columns ({len(selected_cols)}): {preview_cols}")

    print(
        f"Prompts: total={len(np.unique(prompts))}, train={len(np.unique(train_prompts))}, test={len(np.unique(test_prompts))}"
    )
    if chosen_prompts:
        print(f"Test prompt hashes: {chosen_prompts}")
    print(
        f"Train examples={len(y_train)}, Test examples={len(y_test)}, "
        f"train_pos_frac={float(y_train.mean()):.3f}, test_pos_frac={float(y_test.mean()):.3f}"
    )

    losses = fit_vi(
        X_train,
        y_train,
        lr=args.lr,
        steps=args.vi_steps,
        weight_scale=args.weight_scale,
        bias_scale=args.bias_scale,
        use_ard=args.use_ard,
        ard_shape=args.ard_shape,
        ard_rate=args.ard_rate,
        guide_type=args.guide_type,
        guide_rank=args.guide_rank,
        train_particles=args.train_particles,
    )

    eval_samples = args.eval_samples
    mean_train, var_train, logits_train = sample_posterior_predictions(
        X_train,
        eval_samples,
        args.guide_type,
    )
    mean_test, var_test, logits_test = sample_posterior_predictions(
        X_test,
        eval_samples,
        args.guide_type,
    )

    temperature = 1.0
    if args.temp_calibrate_frac > 0.0:
        logits_mean_test = logits_test.mean(dim=0).detach()
        temperature = calibrate_temperature(
            logits_mean_test,
            y_test.detach(),
            args.temp_calibrate_frac,
            args.temp_calibrate_max_iter,
            args.temp_calibrate_lr,
            args.seed,
        )
        probs_train = torch.sigmoid(logits_train / temperature)
        probs_test = torch.sigmoid(logits_test / temperature)
        mean_train = probs_train.mean(dim=0)
        var_train = probs_train.var(dim=0, unbiased=False)
        mean_test = probs_test.mean(dim=0)
        var_test = probs_test.var(dim=0, unbiased=False)
    else:
        temperature = 1.0

    train_metrics = compute_metrics(mean_train, y_train)
    test_metrics = compute_metrics(mean_test, y_test)
    ece, bin_centers, empirical_acc, counts = compute_ece(mean_test, y_test, args.ece_bins)

    conformal_threshold_value = None
    conformal_coverage = None
    conformal_brier = None
    if args.conformal_alpha > 0.0 and y_test.numel() > 0:
        probs_test_np = mean_test.detach().cpu().numpy()
        targets_np = y_test.detach().cpu().numpy()
        rng = np.random.default_rng(args.seed)
        n_calib = max(1, int(round(args.conformal_calib_frac * targets_np.size)))
        n_calib = min(n_calib, targets_np.size)
        calib_idx = rng.choice(targets_np.size, size=n_calib, replace=False)
        confidence_calib = np.maximum(probs_test_np[calib_idx], 1.0 - probs_test_np[calib_idx])
        scores_calib = 1.0 - confidence_calib
        conformal_threshold_value = conformal_threshold(scores_calib, args.conformal_alpha)

        confidence_all = np.maximum(probs_test_np, 1.0 - probs_test_np)
        scores_all = 1.0 - confidence_all
        accept_mask = scores_all <= conformal_threshold_value
        conformal_coverage = float(accept_mask.mean())
        if accept_mask.any():
            conformal_brier = float(np.mean((probs_test_np[accept_mask] - targets_np[accept_mask]) ** 2))

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
        "method": "bayesian_vi",
        "label_column": args.label_column,
        "nll_percentile": None,
        "vi_steps": args.vi_steps,
        "lr": args.lr,
        "n_samples": eval_samples,
        "seed": args.seed,
        "split_seed": args.split_seed,
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "num_examples": int(len(df)),
        "hidden_dim": X_train.shape[1],
        "train_examples": int(len(y_train)),
        "test_examples": int(len(y_test)),
        "train_positive_frac": float(y_train.mean()),
        "test_positive_frac": float(y_test.mean()),
        "test_prompts": chosen_prompts if chosen_prompts else sorted(set(test_prompts)),
        "features_csv": [str(p) for p in args.features_csv],
        "prompt_metadata": prompt_meta,
        "feature_columns": selected_cols,
        "feature_preset": args.feature_preset,
        "use_ard": args.use_ard,
        "ard_shape": args.ard_shape,
        "ard_rate": args.ard_rate,
        "guide_type": args.guide_type,
        "guide_rank": args.guide_rank,
        "train_particles": args.train_particles,
        "temperature": temperature,
        "temp_calibrate_frac": args.temp_calibrate_frac,
        "coral_align": args.coral_align,
        "coral_eps": args.coral_eps,
        "conformal_alpha": args.conformal_alpha,
        "conformal_threshold": conformal_threshold_value,
        "conformal_coverage": conformal_coverage,
        "conformal_brier": conformal_brier,
        "loss_history": losses,
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
            mean_p=mean_test.detach().cpu().numpy(),
            var_p=var_test.detach().cpu().numpy(),
            targets=y_test.detach().cpu().numpy(),
            prompt_hash=np.array(test_prompts, dtype=object),
            bin_centers=bin_centers,
            empirical_acc=empirical_acc,
            counts=counts,
        )
        print(f"Saved posterior predictions to {args.save_preds.resolve()}")

    if args.save_train_preds is not None:
        args.save_train_preds.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            args.save_train_preds,
            mean_p=mean_train.detach().cpu().numpy(),
            var_p=var_train.detach().cpu().numpy(),
            targets=y_train.detach().cpu().numpy(),
            prompt_hash=np.array(train_prompts, dtype=object),
        )
        print(f"Saved training posterior predictions to {args.save_train_preds.resolve()}")


if __name__ == "__main__":
    main()
