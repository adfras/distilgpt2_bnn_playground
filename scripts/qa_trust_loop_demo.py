#!/usr/bin/env python3
"""Compare vanilla DistilGPT-2 decoding with the repo's trust-loop heuristics."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from trust_loop_utils import (  # noqa: E402
    BayesianTrustHead,
    LogisticTrustHead,
    build_corpus,
    combine_generation_kwargs,
    decode_schedule,
    default_generation_kwargs,
    load_generator,
    random_search_decoding,
    run_round,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Trust-loop QA demo")
    parser.add_argument(
        "--model-name",
        default="distilbert/distilgpt2",
        help="Hugging Face model identifier",
    )
    parser.add_argument(
        "--question",
        default="Question: What is the capital of France?\nAnswer:",
        help="Prompt shown to the model",
    )
    parser.add_argument(
        "--reference",
        default="Paris is the capital of France.",
        help="Reference continuation used to guide the trust loop",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=25,
        help="Tokens generated for both baseline and trust loop",
    )
    parser.add_argument(
        "--search-iters",
        type=int,
        default=12,
        help="Random search trials for decoding parameters",
    )
    parser.add_argument(
        "--num-candidates",
        type=int,
        default=3,
        help="Samples per round during trust evaluation",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.78,
        help="Confidence threshold for accepting tokens",
    )
    parser.add_argument(
        "--var-limit",
        type=float,
        default=0.02,
        help="Variance threshold for Bayesian head",
    )
    parser.add_argument(
        "--span-window",
        type=int,
        default=5,
        help="Span window used when scoring trust",
    )
    parser.add_argument(
        "--var-penalty",
        type=float,
        default=0.5,
        help="Penalty applied to predictive variance in trust score",
    )
    parser.add_argument(
        "--align-weight",
        type=float,
        default=0.1,
        help="Weight for reference alignment in trust score",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )
    return parser.parse_args()


def decode_baseline(
    model_name: str,
    prompt: str,
    max_new_tokens: int,
    seed: int,
) -> Dict[str, str]:
    torch.manual_seed(seed)
    tokenizer, model = load_generator(model_name)
    inputs = tokenizer(prompt, return_tensors="pt")
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.9,
            top_p=0.95,
            do_sample=True,
        )
    completion_ids = outputs[0][inputs["input_ids"].shape[1] :]
    completion_text = tokenizer.decode(completion_ids, skip_special_tokens=True).strip()
    full_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return {
        "prompt": prompt,
        "completion": completion_text,
        "full_text": full_text,
    }


def train_heads(
    model_name: str,
    prompts_dir: Path,
    seed: int,
) -> tuple:
    torch.manual_seed(seed)
    np.random.seed(seed)
    tokenizer, model = load_generator(model_name)
    corpus = build_corpus(tokenizer, model, prompts_dir)
    logistic_head = build_logistic_head(corpus, seed)
    bayes_head = build_bayesian_head(corpus, logistic_head)
    return tokenizer, model, logistic_head, bayes_head


def build_logistic_head(corpus, seed: int) -> LogisticTrustHead:
    from trust_loop_utils import train_logistic_head

    return train_logistic_head(corpus, seed)


def build_bayesian_head(corpus, logistic_head: LogisticTrustHead) -> BayesianTrustHead:
    from trust_loop_utils import train_bayesian_head

    return train_bayesian_head(
        corpus,
        steps=0,
        samples=0,
        logistic_backbone=logistic_head,
    )


def run_trust_loop_once(
    tokenizer,
    model,
    logistic_head,
    bayes_head,
    prompt: str,
    reference: str,
    base_kwargs: Dict[str, float],
    args: argparse.Namespace,
) -> Dict[str, object]:
    best_kwargs, best_result = random_search_decoding(
        tokenizer,
        model,
        logistic_head,
        bayes_head,
        prompt,
        reference,
        base_kwargs,
        threshold=args.threshold,
        var_limit=args.var_limit,
        iterations=args.search_iters,
        num_candidates=args.num_candidates,
        span_window=args.span_window,
        var_penalty=args.var_penalty,
        align_weight=args.align_weight,
    )

    if best_result.flagged_indices:
        schedule = decode_schedule(best_kwargs)
        for round_index, override in enumerate(schedule[1:], start=2):
            combined = combine_generation_kwargs(best_kwargs, override)
            next_result = run_round(
                round_index=round_index,
                model=model,
                tokenizer=tokenizer,
                logistic_head=logistic_head,
                bayes_head=bayes_head,
                prompt=prompt,
                reference_continuation=reference,
                threshold=args.threshold,
                var_limit=args.var_limit,
                generation_kwargs=combined,
                num_candidates=args.num_candidates,
                span_window=args.span_window,
                var_penalty=args.var_penalty,
                align_weight=args.align_weight,
            )
            if next_result.trust_score > best_result.trust_score:
                best_result = next_result
                best_kwargs = combined
            if not next_result.flagged_indices:
                break

    return {
        "completion": best_result.completion_text.strip(),
        "trust_score": float(best_result.trust_score),
        "flagged_tokens": best_result.flagged_indices,
        "generation_kwargs": best_kwargs,
    }


def main() -> None:
    args = parse_args()
    baseline = decode_baseline(
        args.model_name,
        args.question,
        args.max_new_tokens,
        args.seed,
    )

    tokenizer, model, logistic_head, bayes_head = train_heads(
        args.model_name,
        Path("prompts"),
        args.seed,
    )

    base_kwargs = default_generation_kwargs(
        {
            "max_new_tokens": args.max_new_tokens,
            "temperature": 0.9,
            "top_p": 0.95,
        }
    )

    trust_run = run_trust_loop_once(
        tokenizer,
        model,
        logistic_head,
        bayes_head,
        args.question,
        args.reference,
        base_kwargs,
        args,
    )

    report = {
        "baseline": baseline,
        "trust_loop": trust_run,
        "reference": args.reference,
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
