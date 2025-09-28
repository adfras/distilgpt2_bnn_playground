#!/usr/bin/env python
"""Generate text with DistilGPT-2 and dump per-token features to CSV.

The script is intentionally lightweight so it can run on CPU-only machines.
It prints the generated text to stdout and writes a feature matrix containing
per-token negative log-likelihoods (next-token prediction) and final hidden
states to a CSV file.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="DistilGPT-2 token feature dump")
    parser.add_argument(
        "--model-name",
        default="distilgpt2",
        help="Hugging Face model identifier to load",
    )
    parser.add_argument(
        "--prompt",
        default=(
            "Explain in two short paragraphs why Bayesian neural networks can "
            "provide calibrated uncertainty estimates compared to frequentist heads."
        ),
        help="Text prompt used for generation",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=80,
        help="Number of new tokens to generate",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.9,
        help="Sampling temperature",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=0.95,
        help="Top-p nucleus sampling parameter",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for torch manual_seed",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/token_features.csv"),
        help="Output CSV path for token features",
    )
    parser.add_argument(
        "--save-generation",
        type=Path,
        default=None,
        help="Optional path to save the generated text (e.g. reports/generation.txt)",
    )
    parser.add_argument(
        "--meta-output",
        type=Path,
        default=None,
        help="Optional path for metadata JSON; defaults to <output>.meta.json",
    )
    parser.add_argument(
        "--print-config",
        action="store_true",
        help="Print configuration as JSON before running",
    )
    return parser.parse_args()


def ensure_tokenizer_padding(tokenizer: AutoTokenizer) -> None:
    if tokenizer.pad_token is None:
        # DistilGPT-2 lacks an explicit pad token; reuse eos to avoid errors.
        tokenizer.pad_token = tokenizer.eos_token


def load_model_and_tokenizer(model_name: str) -> tuple[AutoTokenizer, AutoModelForCausalLM]:
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    model.eval()
    ensure_tokenizer_padding(tokenizer)
    return tokenizer, model


def generate_sequence(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompt: str,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
) -> torch.LongTensor:
    inputs = tokenizer(prompt, return_tensors="pt")
    if max_new_tokens <= 0:
        # When no generation is requested we simply return the encoded prompt;
        # this allows evaluation on fixed, non-model text passages.
        return inputs["input_ids"]
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=True,
            return_dict_in_generate=True,
        )
    return outputs.sequences


TOP_K_PROBS = 10


def compute_token_features(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    sequences: torch.LongTensor,
) -> pd.DataFrame:
    with torch.no_grad():
        outputs = model(
            input_ids=sequences,
            attention_mask=torch.ones_like(sequences),
            output_hidden_states=True,
            return_dict=True,
        )

    logits = outputs.logits[0].detach().cpu()  # (seq_len, vocab)
    hidden = outputs.hidden_states[-1][0].detach().cpu()  # (seq_len, hidden_dim)

    seq_len, hidden_dim = hidden.shape
    token_ids = sequences[0].tolist()
    tokens = tokenizer.convert_ids_to_tokens(token_ids)

    # Compute negative log-likelihood for predicting the next token.
    log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
    target_ids = sequences[0][1:].to(log_probs.device)
    nll = torch.full(
        (seq_len,),
        torch.nan,
        dtype=log_probs.dtype,
        device=log_probs.device,
    )
    if seq_len > 1:
        next_token_log_probs = log_probs[:-1, :]
        nll[:-1] = -next_token_log_probs.gather(1, target_ids.unsqueeze(-1)).squeeze(-1)

    pred_ids = torch.full((seq_len - 1,), fill_value=-1, dtype=torch.long, device=log_probs.device)
    if seq_len > 1:
        pred_ids = next_token_log_probs.argmax(dim=-1)
    correct = torch.zeros(seq_len, dtype=torch.float32)
    p_true = torch.full((seq_len,), torch.nan, dtype=torch.float32)
    p_max = torch.full((seq_len,), torch.nan, dtype=torch.float32)
    log_p_max = torch.full((seq_len,), torch.nan, dtype=torch.float32)
    logit_gap = torch.full((seq_len,), torch.nan, dtype=torch.float32)
    entropy = torch.full((seq_len,), torch.nan, dtype=torch.float32)
    p_second = torch.full((seq_len,), torch.nan, dtype=torch.float32)
    log_p_second = torch.full((seq_len,), torch.nan, dtype=torch.float32)
    p_margin = torch.full((seq_len,), torch.nan, dtype=torch.float32)
    p_ratio = torch.full((seq_len,), torch.nan, dtype=torch.float32)
    p_var = torch.full((seq_len,), torch.nan, dtype=torch.float32)
    gap_over_entropy_sm = torch.full((seq_len,), torch.nan, dtype=torch.float32)
    tail_mass_topk = torch.full((seq_len,), torch.nan, dtype=torch.float32)
    true_in_topk = torch.full((seq_len,), torch.nan, dtype=torch.float32)
    true_rank_topk = torch.full((seq_len,), torch.nan, dtype=torch.float32)
    logit_true_gap = torch.full((seq_len,), torch.nan, dtype=torch.float32)

    max_k = min(TOP_K_PROBS, model.config.vocab_size)
    p_top_cols = torch.full((seq_len, max_k), torch.nan, dtype=torch.float32)
    logit_top_cols = torch.full((seq_len, max_k), torch.nan, dtype=torch.float32)
    logit_gap_cols = torch.full((seq_len, max_k), torch.nan, dtype=torch.float32)
    if seq_len > 1:
        correct[:-1] = (pred_ids == target_ids).float().cpu()
        p_true[:-1] = torch.exp(-nll[:-1]).cpu()

        probs_next = torch.exp(next_token_log_probs)
        max_probs, max_idx = probs_next.max(dim=-1)
        max_probs = max_probs.clamp_min(1e-12)
        topk_logits = torch.topk(logits[:-1], k=max_k, dim=-1).values
        topk_indices = torch.topk(logits[:-1], k=max_k, dim=-1).indices
        topk_probs = torch.topk(probs_next, k=max_k, dim=-1).values
        second_probs = topk_probs[:, 1].clamp_min(1e-12)
        p_max[:-1] = max_probs.cpu()
        log_p_max[:-1] = torch.log(max_probs).cpu()
        logit_gap_vals = topk_logits[:, 0] - topk_logits[:, 1]
        logit_gap[:-1] = logit_gap_vals.cpu()
        entropy_vals = (-(probs_next * next_token_log_probs).sum(dim=-1))
        entropy[:-1] = entropy_vals.cpu()

        p_second[:-1] = second_probs.cpu()
        log_p_second[:-1] = torch.log(second_probs).cpu()
        p_margin[:-1] = (max_probs - second_probs).cpu()
        p_ratio[:-1] = (max_probs / (second_probs + 1e-12)).cpu()
        p_var[:-1] = (max_probs * (1 - max_probs)).cpu()
        gap_over_entropy_sm[:-1] = (logit_gap_vals / (entropy_vals + 1e-3)).cpu()
        tail_mass_topk[:-1] = (1.0 - topk_probs.sum(dim=-1)).cpu()

        p_top_cols[:-1, : topk_probs.shape[1]] = topk_probs.cpu()
        logit_top_cols[:-1, : topk_logits.shape[1]] = topk_logits.cpu()
        logit_gap_cols[:-1, : topk_logits.shape[1]] = (
            (topk_logits[:, :1] - topk_logits).cpu()
        )

        target_ids_next = target_ids
        target_logits = logits[:-1].gather(1, target_ids_next.unsqueeze(-1)).squeeze(-1)
        logit_true_gap[:-1] = (topk_logits[:, 0] - target_logits).cpu()

        topk_matches = topk_indices == target_ids_next.unsqueeze(-1)
        in_topk = topk_matches.any(dim=-1)
        true_in_topk[:-1] = in_topk.float().cpu()
        if in_topk.any():
            ranks = torch.argmax(topk_matches.float(), dim=-1).to(torch.float32) + 1.0
            masked_ranks = torch.where(
                in_topk,
                ranks,
                torch.full_like(ranks, float("nan"))
            )
            true_rank_topk[:-1] = masked_ranks.cpu()

    df = pd.DataFrame(
        {
            "position": range(seq_len),
            "token": tokens,
            "nll": nll.cpu().numpy(),
        }
    )

    target_ids_np = torch.full((seq_len,), -1, dtype=torch.long)
    if seq_len > 1:
        target_ids_np[:-1] = target_ids.cpu()

    pred_ids_np = torch.full((seq_len,), -1, dtype=torch.long)
    if seq_len > 1:
        pred_ids_np[:-1] = pred_ids.cpu()

    df["target_id"] = target_ids_np.numpy()
    df["pred_id"] = pred_ids_np.numpy()
    df["is_correct"] = correct.numpy()
    df["p_true"] = p_true.numpy()
    df["p_max"] = p_max.numpy()
    df["log_p_max"] = log_p_max.numpy()
    df["logit_gap"] = logit_gap.numpy()
    df["entropy"] = entropy.numpy()
    df["p_second"] = p_second.numpy()
    df["log_p_second"] = log_p_second.numpy()
    df["p_margin"] = p_margin.numpy()
    df["p_ratio"] = p_ratio.numpy()
    df["p_var"] = p_var.numpy()
    df["gap_over_entropy_sm"] = gap_over_entropy_sm.numpy()
    df["tail_mass_topk"] = tail_mass_topk.numpy()
    df["true_in_topk"] = true_in_topk.numpy()
    df["true_rank_topk"] = true_rank_topk.numpy()
    df["logit_true_gap"] = logit_true_gap.numpy()

    # Top-k probability and logit columns (1-indexed for readability)
    for k in range(max_k):
        df[f"p_top{k+1}"] = p_top_cols[:, k].numpy()
        df[f"logit_top{k+1}"] = logit_top_cols[:, k].numpy()
        df[f"logit_gap_top{k+1}"] = logit_gap_cols[:, k].numpy()

    hidden_np = hidden.numpy()
    feature_columns = {f"h{i}": hidden_np[:, i] for i in range(hidden_dim)}
    feature_df = pd.DataFrame(feature_columns)
    df = pd.concat([df, feature_df], axis=1)

    # Deterministic feature engineering to keep downstream heads lightweight.
    df["p_max_sq"] = df["p_max"] ** 2
    df["logit_gap_sq"] = df["logit_gap"] ** 2
    df["gap_over_entropy"] = df["logit_gap"] / (df["entropy"] + 1e-3)

    # Rolling aggregates provide context-aware signals without touching the base model.
    df["p_max_ma3"] = df["p_max"].rolling(window=3, min_periods=1).mean()
    df["p_max_ma5"] = df["p_max"].rolling(window=5, min_periods=1).mean()
    df["p_var_ma5"] = df["p_var"].rolling(window=5, min_periods=1).mean()

    # Directional context windows (previous/next five tokens) to capture local streaks.
    df["p_max_prev5_mean"] = df["p_max"].rolling(window=5, min_periods=1).mean()
    df["p_max_next5_mean"] = (
        df["p_max"].iloc[::-1].rolling(window=5, min_periods=1).mean().iloc[::-1]
    )
    df["entropy_prev5_mean"] = df["entropy"].rolling(window=5, min_periods=1).mean()
    df["entropy_next5_mean"] = (
        df["entropy"].iloc[::-1].rolling(window=5, min_periods=1).mean().iloc[::-1]
    )
    df["p_var_prev5_mean"] = df["p_var"].rolling(window=5, min_periods=1).mean()
    df["p_var_next5_mean"] = (
        df["p_var"].iloc[::-1].rolling(window=5, min_periods=1).mean().iloc[::-1]
    )
    return df


def save_generation_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)

    if args.print_config:
        config = {
            "model_name": args.model_name,
            "prompt": args.prompt,
            "max_new_tokens": args.max_new_tokens,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "seed": args.seed,
            "output": str(args.output),
            "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        }
        print(json.dumps(config, indent=2))

    tokenizer, model = load_model_and_tokenizer(args.model_name)

    sequences = generate_sequence(
        model,
        tokenizer,
        args.prompt,
        args.max_new_tokens,
        args.temperature,
        args.top_p,
    )

    generated_text = tokenizer.decode(sequences[0], skip_special_tokens=True)
    print("\n=== Generated Text ===\n")
    print(generated_text)
    print("\n======================\n")

    df = compute_token_features(model, tokenizer, sequences)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output, index=False)
    print(f"Saved token features to {args.output.resolve()}")

    if args.save_generation is not None:
        save_generation_text(args.save_generation, generated_text)
        print(f"Saved generation text to {args.save_generation.resolve()}")

    meta_path = args.meta_output
    if meta_path is None:
        meta_path = args.output.with_suffix(".meta.json")

    prompt_hash = hashlib.sha1(args.prompt.encode("utf-8")).hexdigest()
    meta = {
        "prompt": args.prompt,
        "prompt_hash": prompt_hash,
        "model_name": args.model_name,
        "max_new_tokens": args.max_new_tokens,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "seed": args.seed,
        "features_csv": str(args.output),
        "generation_path": str(args.save_generation) if args.save_generation else None,
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    meta_path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    print(f"Saved metadata to {meta_path.resolve()}")


if __name__ == "__main__":
    main()
