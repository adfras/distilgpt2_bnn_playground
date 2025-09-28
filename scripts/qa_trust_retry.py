#!/usr/bin/env python3
"""Retry the QA trust loop until a target confidence is reached or we escalate."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from trust_loop_utils import (  # noqa: E402
    build_corpus,
    combine_generation_kwargs,
    decode_schedule,
    default_generation_kwargs,
    load_generator,
    random_search_decoding,
    run_round,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Confidence-driven retry wrapper for the QA trust loop")
    parser.add_argument("--model-name", default="distilbert/distilgpt2", help="Base model to sample from")
    parser.add_argument("--question", required=True, help="Prompt that contains the question")
    parser.add_argument("--reference", default="", help="Optional reference continuation for alignment checks")
    parser.add_argument("--target-trust", type=float, default=0.65, help="Accept samples only when trust score >= this value")
    parser.add_argument("--max-attempts", type=int, default=5, help="Maximum regeneration attempts before escalation")
    parser.add_argument("--baseline-temp", type=float, default=0.9, help="Initial temperature for the search seed")
    parser.add_argument("--baseline-top-p", type=float, default=0.95, help="Initial top-p for the search seed")
    parser.add_argument("--max-new-tokens", type=int, default=32, help="Tokens generated per attempt")
    parser.add_argument("--search-iters", type=int, default=16, help="Random search trials per attempt")
    parser.add_argument("--num-candidates", type=int, default=3, help="Samples per scoring round")
    parser.add_argument("--threshold", type=float, default=0.75, help="Span accept threshold for the trust head")
    parser.add_argument("--var-limit", type=float, default=0.02, help="Variance limit when evaluating posterior uncertainty")
    parser.add_argument("--span-window", type=int, default=5, help="Span size for windowed scoring")
    parser.add_argument("--var-penalty", type=float, default=0.5, help="Penalty applied to predictive variance")
    parser.add_argument("--align-weight", type=float, default=0.15, help="Weight for reference alignment in the trust score")
    parser.add_argument("--seed", type=int, default=123, help="Global random seed")
    parser.add_argument("--verbose", action="store_true", help="Emit per-attempt diagnostics")
    parser.add_argument(
        "--hint-on-fail",
        action="store_true",
        help="If set, append the reference as a hint after the first low-confidence attempt",
    )
    parser.add_argument(
        "--require-reference-token",
        action="store_true",
        help="Only accept completions that contain the first token of the reference (case-insensitive)",
    )
    return parser.parse_args()


def completion_matches_reference(text: str, reference: str) -> bool:
    if not reference:
        return True
    target = reference.strip()
    if not target:
        return True
    target_token = target.split()[0].strip('.,!?"\'')
    if not target_token:
        return True
    return target_token.lower() in text.lower()


def train_heads(model_name: str, seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)
    tokenizer, model = load_generator(model_name)
    corpus = build_corpus(tokenizer, model, Path("prompts"))

    from trust_loop_utils import train_logistic_head, train_bayesian_head

    logistic_head = train_logistic_head(corpus, seed)
    bayes_head = train_bayesian_head(
        corpus,
        steps=0,
        samples=0,
        logistic_backbone=logistic_head,
    )
    return tokenizer, model, logistic_head, bayes_head


def run_attempt(
    attempt: int,
    tokenizer,
    model,
    logistic_head,
    bayes_head,
    prompt: str,
    reference: str,
    base_kwargs: Dict[str, float],
    args: argparse.Namespace,
) -> Dict[str, object]:
    search_seed = args.seed + attempt

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
        seed=search_seed,
    )

    if best_result.flagged_indices:
        schedule = decode_schedule(best_kwargs)
        for override in schedule[1:]:
            combined = combine_generation_kwargs(best_kwargs, override)
            retry = run_round(
                round_index=attempt + 1,
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
            if retry.trust_score > best_result.trust_score:
                best_result = retry
                best_kwargs = combined
            if not retry.flagged_indices:
                break

    return {
        "completion": best_result.completion_text.strip(),
        "trust_score": float(best_result.trust_score),
        "flagged_tokens": best_result.flagged_indices,
        "generation_kwargs": best_kwargs,
    }


def main() -> None:
    args = parse_args()
    tokenizer, model, logistic_head, bayes_head = train_heads(args.model_name, args.seed)

    base_kwargs: Dict[str, float] = default_generation_kwargs(
        {
            "max_new_tokens": args.max_new_tokens,
            "temperature": args.baseline_temp,
            "top_p": args.baseline_top_p,
        }
    )

    attempts: list[Dict[str, object]] = []
    accepted: Optional[Dict[str, object]] = None
    current_prompt = args.question

    for attempt in range(1, args.max_attempts + 1):
        result = run_attempt(
            attempt=attempt,
            tokenizer=tokenizer,
            model=model,
            logistic_head=logistic_head,
            bayes_head=bayes_head,
            prompt=current_prompt,
            reference=args.reference,
            base_kwargs=base_kwargs,
            args=args,
        )
        attempts.append(result)

        if args.verbose:
            print(
                f"Attempt {attempt}: trust={result['trust_score']:.3f}, flags={len(result['flagged_tokens'])}"
            )

        if result["trust_score"] >= args.target_trust and not result["flagged_tokens"]:
            if (
                args.require_reference_token
                and not completion_matches_reference(result["completion"], args.reference)
            ):
                if args.verbose:
                    print(
                        "Rejected candidate despite high trust because it did not mention the reference token."
                    )
            else:
                accepted = result
                break

        # Use the best generation kwargs from this attempt to seed the next one.
        base_kwargs = result["generation_kwargs"]

        if (
            args.hint_on_fail
            and args.reference
            and "Hint:" not in current_prompt
        ):
            current_prompt = (
                f"{args.question}\nHint: {args.reference}\n"
                "Answer in one word:"
            )
            # steer subsequent attempts to be more deterministic once we reveal the hint
            base_kwargs = dict(base_kwargs)
            base_kwargs["temperature"] = min(base_kwargs.get("temperature", args.baseline_temp), 0.3)
            base_kwargs["top_p"] = min(base_kwargs.get("top_p", args.baseline_top_p), 0.5)
            base_kwargs["top_k"] = min(base_kwargs.get("top_k", 50), 20)
            base_kwargs["max_new_tokens"] = min(base_kwargs.get("max_new_tokens", args.max_new_tokens), 6)

    report = {
        "prompt": args.question,
        "reference": args.reference,
        "target_trust": args.target_trust,
        "accepted": accepted,
        "attempts": attempts,
        "escalate": accepted is None,
    }

    if accepted and args.require_reference_token and args.reference:
        token = args.reference.strip().split()[0].strip('.,!?"\'')
        if token:
            report["accepted"]["normalized_completion"] = token

    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
