from __future__ import annotations

import streamlit as st
from pathlib import Path
from typing import Callable, List, Optional
import numpy as np

from trust_loop_utils import (
    BayesianTrustHead,
    LogisticTrustHead,
    PROFILE,
    PROFILE_NAME,
    build_corpus,
    combine_generation_kwargs,
    decode_schedule,
    load_generator,
    random_search_decoding,
    render_tokens_html,
    run_round,
    split_prompt_and_reference,
    tokens_to_text,
    train_bayesian_head,
    train_logistic_head,
)

SCENARIOS = {
    "Maxwell Treatise (technical)": {
        "path": Path("prompts/maxwell_treatise_excerpt.txt"),
        "prefix_tokens": 60,
        "generation": {"max_new_tokens": 90, "temperature": 0.9, "top_p": 0.95},
    },
    "Moby-Dick (narrative)": {
        "path": Path("prompts/moby_dick_excerpt.txt"),
        "prefix_tokens": 60,
        "generation": {"max_new_tokens": 90, "temperature": 0.9, "top_p": 0.95},
    },
    "NOAA Arctic Bulletin": {
        "path": Path("prompts/noaa_arctic_report.txt"),
        "prefix_tokens": 55,
        "generation": {"max_new_tokens": 80, "temperature": 0.9, "top_p": 0.95},
    },
}

MODEL_NAME = "distilgpt2"

DEFAULT_POLICY = {
    "threshold": 0.78,
    "var_limit": 0.02,
    "max_rounds": 3,
    "search_iters": 12,
    "search_candidates": 2,
    "best_of": 3,
    "span_window": 5,
    "var_penalty": 0.5,
    "align_weight": 0.1,
}


@st.cache_resource(show_spinner=False)
def load_resources(model_name: str) -> tuple:
    tokenizer, model = load_generator(model_name)
    corpus = build_corpus(tokenizer, model, Path("prompts"))
    logistic_head = train_logistic_head(corpus)
    bayes_head = train_bayesian_head(corpus, logistic_backbone=logistic_head)
    return tokenizer, model, logistic_head, bayes_head


def run_trust_loop(
    tokenizer,
    model,
    logistic_head: LogisticTrustHead,
    bayes_head: BayesianTrustHead,
    prompt_text: str,
    reference_text: str,
    threshold: float,
    var_limit: float,
    generation_cfg: dict,
    max_rounds: int,
    search_iters: int,
    search_candidates: int,
    best_of_n: int,
    span_window: int,
    var_penalty: float,
    align_weight: float,
    progress_callback: Optional[Callable[[float], None]] = None,
):
    def set_progress(fraction: float) -> None:
        if progress_callback is not None:
            progress_callback(min(1.0, max(0.0, fraction)))

    rounds: List = []
    prompt_ids = tokenizer(
        prompt_text, add_special_tokens=False, return_tensors="pt"
    )["input_ids"][0].tolist()
    reference_ids = tokenizer(
        reference_text, add_special_tokens=False, return_tensors="pt"
    )["input_ids"][0].tolist()

    search_weight = PROFILE["search_progress_weight"]
    remaining_weight = max(0.0, 1.0 - search_weight)

    def search_progress_cb(fraction: float) -> None:
        set_progress(search_weight * min(1.0, max(0.0, fraction)))

    set_progress(0.0)

    best_kwargs, best_result = random_search_decoding(
        tokenizer,
        model,
        logistic_head,
        bayes_head,
        prompt_text,
        reference_text,
        generation_cfg,
        threshold,
        var_limit,
        iterations=max(1, search_iters),
        num_candidates=max(1, search_candidates),
        span_window=span_window,
        var_penalty=var_penalty,
        align_weight=align_weight,
        progress_callback=search_progress_cb,
    )
    best_result.round_index = 1
    rounds.append(best_result)
    set_progress(search_weight)

    base_kwargs = dict(best_kwargs)
    schedule = decode_schedule(base_kwargs)

    accepted_ids: List[int] = []
    flagged = best_result.flagged_indices
    tokens_ids = best_result.dataframe["token_id"].tolist()
    if flagged:
        accepted_slice = tokens_ids[: flagged[0]]
    else:
        accepted_slice = tokens_ids
    accepted_ids.extend(accepted_slice)

    if not flagged:
        set_progress(1.0)
        return rounds, True, ""

    prev_flagged = len(flagged)
    stagnant_rounds = 0
    stop_reason = ""
    success = False

    total_extra_rounds = max(1, max_rounds - 1)

    for round_index in range(2, max_rounds + 1):
        schedule_idx = min(round_index - 1, len(schedule) - 1)
        override_cfg = schedule[schedule_idx]
        generation_kwargs = combine_generation_kwargs(base_kwargs, override_cfg)

        working_prompt_ids = prompt_ids + accepted_ids
        working_prompt_text = tokenizer.decode(working_prompt_ids, skip_special_tokens=True)
        remaining_reference_ids = reference_ids[len(accepted_ids) :]
        remaining_reference_text = tokens_to_text(tokenizer, remaining_reference_ids)

        result = run_round(
            round_index,
            model,
            tokenizer,
            logistic_head,
            bayes_head,
            working_prompt_text,
            remaining_reference_text,
            threshold,
            var_limit,
            generation_kwargs=generation_kwargs,
            num_candidates=max(1, best_of_n),
            span_window=span_window,
            var_penalty=var_penalty,
            align_weight=align_weight,
        )
        rounds.append(result)

        flagged = result.flagged_indices
        tokens_ids = result.dataframe["token_id"].tolist()
        if flagged:
            accepted_slice = tokens_ids[: flagged[0]]
        else:
            accepted_slice = tokens_ids
        accepted_ids.extend(accepted_slice)

        if not flagged:
            success = True
            progress_fraction = search_weight + remaining_weight * min(
                1.0, (round_index - 1) / total_extra_rounds
            )
            set_progress(progress_fraction)
            break

        if prev_flagged is not None and len(flagged) >= prev_flagged and not accepted_slice:
            stagnant_rounds += 1
        else:
            stagnant_rounds = 0
        prev_flagged = len(flagged)

        progress_fraction = search_weight + remaining_weight * min(
            1.0, (round_index - 1) / total_extra_rounds
        )
        set_progress(progress_fraction)

        if stagnant_rounds >= 2:
            stop_reason = "Flagged spans are not shrinking; stopping early."
            break

    if not success and not stop_reason:
        stop_reason = "Reached the configured round limit."

    baseline_accept = np.ones(len(best_result.logistic_probs), dtype=bool)
    baseline_accept[best_result.flagged_indices] = False
    baseline_coverage = float(baseline_accept.mean()) if baseline_accept.size else 0.0
    baseline_info = {
        "settings": {
            "threshold": threshold,
            "var_limit": var_limit,
            "max_rounds": max_rounds,
            "search_iters": search_iters,
            "search_candidates": search_candidates,
            "best_of": best_of_n,
            "span_window": span_window,
            "var_penalty": var_penalty,
            "align_weight": align_weight,
            "generation_kwargs": base_kwargs,
        },
        "trust": float(best_result.trust_score),
        "coverage": baseline_coverage,
    }

    set_progress(1.0)

    return rounds, success, stop_reason, baseline_info


def main() -> None:
    st.set_page_config(page_title="Trust Loop Demo", layout="wide")
    st.title("LM Trust Loop Demo")
    st.caption("Generate → evaluate → revise with logistic + Bayesian heads")

    with st.spinner("Loading model and training trust heads (one-time setup)..."):
        tokenizer, model, logistic_head, bayes_head = load_resources(MODEL_NAME)

    scenario = st.selectbox("Scenario", list(SCENARIOS.keys()))
    config = SCENARIOS[scenario]
    full_text = config["path"].read_text(encoding="utf-8").strip()
    prompt_text, reference_text = split_prompt_and_reference(
        tokenizer, full_text, config["prefix_tokens"]
    )

    if "scenario_profiles" not in st.session_state:
        st.session_state["scenario_profiles"] = {}

    scenario_profiles = st.session_state["scenario_profiles"]
    scenario_key = scenario

    st.sidebar.header("Trust policy")
    st.sidebar.markdown(
        "The tool auto-searches decoding settings, stores the tuned baseline per "
        "scenario, and replays the regenerate loop. Use the buttons below to force a "
        "fresh calibration or to replay with the stored profile."
    )
    st.sidebar.caption(f"Runtime profile: `{PROFILE_NAME}`")

    needs_run = False
    if "last_scenario" not in st.session_state or st.session_state["last_scenario"] != scenario_key:
        st.session_state["last_scenario"] = scenario_key
        needs_run = True

    if st.sidebar.button("Re-calibrate now"):
        needs_run = True

    if scenario_key not in scenario_profiles:
        needs_run = True

    policy = scenario_profiles.get(scenario_key, {}).get("settings", DEFAULT_POLICY)

    if scenario_key in scenario_profiles:
        baseline = scenario_profiles[scenario_key]
        st.sidebar.success(
            f"Baseline trust: {baseline['trust']:.3f}\n\n"
            f"Baseline coverage: {baseline['coverage']:.0%}"
        )
        base_gen = config["generation"]
        gen_kwargs = baseline["settings"].get("generation_kwargs", base_gen)
        temp_val = gen_kwargs.get("temperature", base_gen.get("temperature", 0.9))
        top_p_val = gen_kwargs.get("top_p", base_gen.get("top_p", 0.9))
        top_k_val = gen_kwargs.get("top_k", base_gen.get("top_k", 50))
        st.sidebar.markdown(
            "**Optimised decode (round 1)**\n"
            f"temp={temp_val:.2f}, top_p={top_p_val:.2f}, top_k={int(top_k_val)}"
        )

    def run_with_policy(policy_dict, progress_bar=None):
        generation_cfg = policy_dict.get("generation_kwargs", config["generation"])
        def progress_cb(fraction: float):
            if progress_bar is not None:
                progress_bar.progress(min(100, max(0, int(fraction * 100))))
        return run_trust_loop(
            tokenizer,
            model,
            logistic_head,
            bayes_head,
            prompt_text,
            reference_text,
            threshold=policy_dict["threshold"],
            var_limit=policy_dict["var_limit"],
            generation_cfg=generation_cfg,
            max_rounds=policy_dict["max_rounds"],
            search_iters=policy_dict["search_iters"],
            search_candidates=policy_dict["search_candidates"],
            best_of_n=policy_dict["best_of"],
            span_window=policy_dict["span_window"],
            var_penalty=policy_dict["var_penalty"],
            align_weight=policy_dict["align_weight"],
            progress_callback=progress_cb,
        )

    if needs_run:
        progress_bar = st.progress(0)
        with st.spinner("Calibrating decoding policy for this scenario..."):
            rounds, success, reason, baseline_info = run_with_policy(DEFAULT_POLICY, progress_bar)
        progress_bar.progress(100)
        progress_bar.empty()
        scenario_profiles[scenario_key] = baseline_info
        st.session_state["trust_outcome"] = {
            "rounds": rounds,
            "success": success,
            "reason": reason,
            "baseline": baseline_info,
        }
        st.session_state["policy_locked"] = baseline_info["settings"]
    elif st.sidebar.button("Re-run with stored policy"):
        locked_policy = st.session_state.get("policy_locked") or policy
        progress_bar = st.progress(0)
        with st.spinner("Running regenerate loop with stored settings..."):
            rounds, success, reason, baseline_info = run_with_policy(locked_policy, progress_bar)
        progress_bar.progress(100)
        progress_bar.empty()
        st.session_state["trust_outcome"] = {
            "rounds": rounds,
            "success": success,
            "reason": reason,
            "baseline": scenario_profiles.get(scenario_key, baseline_info),
        }
        st.session_state["policy_locked"] = locked_policy

    outcome = st.session_state.get("trust_outcome")
    st.subheader("Prompt")
    st.write(prompt_text)

    policy_current = scenario_profiles.get(scenario_key, {}).get("settings", DEFAULT_POLICY)

    if not outcome or not outcome.get("rounds"):
        st.info("Select a scenario (or press Re-calibrate) to trigger auto-tuning.")
        return

    rounds = outcome["rounds"]
    success = outcome["success"]
    reason = outcome["reason"]
    baseline = outcome.get("baseline")
    threshold = policy_current["threshold"]
    var_limit = policy_current["var_limit"]

    for result in rounds:
        flagged = result.flagged_indices
        accept_mask = np.ones(len(result.logistic_probs), dtype=bool)
        accept_mask[flagged] = False
        coverage = float(accept_mask.mean()) if accept_mask.size else 0.0
        flagged_count = len(flagged)
        html_text = render_tokens_html(
            tokenizer,
            result.tokens,
            result.logistic_probs,
            result.match_reference,
            result.bayes_var,
            threshold,
            var_limit,
            flagged,
        )

        with st.container():
            st.markdown(f"### Round {result.round_index}")
            st.markdown(
                f"**Coverage**: {coverage:.0%} &nbsp;&nbsp; **Flagged tokens**: {flagged_count}"
                f" &nbsp;&nbsp; **Trust score**: {result.trust_score:.3f}",
            )
            st.markdown(html_text, unsafe_allow_html=True)
            st.expander("Show raw completion").write(result.completion_text)

    final_flagged = rounds[-1].flagged_indices
    final_trust = rounds[-1].trust_score
    baseline_trust = baseline["trust"] if baseline else None

    if baseline_trust is not None:
        delta = final_trust - baseline_trust
        if delta >= 0:
            st.success(
                f"Trust score {final_trust:.3f} (baseline {baseline_trust:.3f}, +{delta:.3f})."
            )
        else:
            st.error(
                f"Trust score {final_trust:.3f} is below the tuned baseline {baseline_trust:.3f} ({delta:.3f})."
            )

    if success and not final_flagged:
        st.info("Final round meets the trust policy — no spans flagged.")
    else:
        st.warning(reason)


if __name__ == "__main__":
    main()
