"""Utilities for the Streamlit trust-loop demo."""
from __future__ import annotations

import difflib
import hashlib
import html
import os
from collections import Counter
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import pyro
import pyro.distributions as dist
import torch
import torch.nn.functional as F
from pyro.infer import SVI, Trace_ELBO
from pyro.optim import Adam
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import (
    AutoModelForCausalLM,
    AutoModelForMaskedLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
)
from transformers.generation.logits_process import (
    LogitsProcessorList,
    NoBadWordsLogitsProcessor,
    NoRepeatNGramLogitsProcessor,
    RepetitionPenaltyLogitsProcessor,
    SequenceBiasLogitsProcessor,
    TemperatureLogitsWarper,
    TopKLogitsWarper,
    TopPLogitsWarper,
)

from scripts.basic_lm_quickstart import (
    compute_token_features,
    ensure_tokenizer_padding,
    generate_sequence,
    load_model_and_tokenizer,
)
from scripts.bnn_head_pyro_template import logistic_guide, logistic_model

FEATURE_COLUMNS: Tuple[str, ...] = (
    "p_max",
    "log_p_max",
    "p_top2",
    "p_margin",
    "p_var",
    "tail_mass_topk",
    "logit_gap",
    "logit_gap_top3",
    "entropy",
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

MASKED_LM_NAME = "roberta-base"
NLI_MODEL_NAME = "facebook/bart-large-mnli"

_MASKED_LM = None
_MASKED_LM_TOKENIZER = None
_NLI_MODEL = None
_NLI_TOKENIZER = None
_RETRIEVAL_VECTORIZER: Optional[TfidfVectorizer] = None
_RETRIEVAL_MATRIX = None
_RETRIEVAL_DOCS: List[str] = []
_RETRIEVAL_META: List[Dict[str, str]] = []

BANNED_TERMS = {
    "maxwell": [
        "magnetic charge",
        "magnetically charged",
        "monopole",
    ]
}

CALIBRATION_ALPHA = 0.1
CALIBRATION_INFO = {"quantile": None}

_PROFILE_PRESETS = {
    "balanced": {
        "bayes_steps": 120,
        "bayes_samples": 80,
        "bayes_predict_samples": 200,
        "contrastive_alpha": 0.35,
        "use_contrastive": True,
        "enable_pll": True,
        "enable_entailment": True,
        "evidence_top_k": 2,
        "repair_attempts": 4,
        "search_progress_weight": 0.6,
    },
    "light": {
        "bayes_steps": 0,
        "bayes_samples": 0,
        "bayes_predict_samples": 0,
        "contrastive_alpha": 0.2,
        "use_contrastive": False,
        "enable_pll": False,
        "enable_entailment": False,
        "evidence_top_k": 0,
        "repair_attempts": 1,
        "search_progress_weight": 0.5,
    },
}

PROFILE_NAME = os.environ.get("TRUST_LOOP_PROFILE", "balanced").strip().lower()
PROFILE = _PROFILE_PRESETS.get(PROFILE_NAME, _PROFILE_PRESETS["balanced"])


def apply_conformal_adjustments(threshold: float) -> float:
    quantile = CALIBRATION_INFO.get("quantile")
    if quantile is None:
        return threshold
    return float(max(threshold, 1.0 - quantile))

@dataclass
class LogisticTrustHead:
    model: LogisticRegression
    scaler: StandardScaler
    feature_names: Sequence[str]

    def predict(self, features: pd.DataFrame) -> np.ndarray:
        matrix = features[self.feature_names].to_numpy(dtype=np.float32)
        matrix = self.scaler.transform(matrix)
        probs = self.model.predict_proba(matrix)[:, 1]
        return probs

@dataclass
class BayesianTrustHead:
    weight_loc: torch.Tensor
    weight_scale: torch.Tensor
    bias_loc: torch.Tensor
    bias_scale: torch.Tensor
    feature_mean: np.ndarray
    feature_scale: np.ndarray
    feature_names: Sequence[str]
    samples: int = PROFILE["bayes_predict_samples"]

    def predict(self, features: pd.DataFrame, samples: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
        sample_count = samples or self.samples
        matrix = features[self.feature_names].to_numpy(dtype=np.float32)
        matrix = (matrix - self.feature_mean) / np.where(self.feature_scale == 0.0, 1.0, self.feature_scale)
        x = torch.from_numpy(matrix)
        w_dist = dist.Normal(self.weight_loc, self.weight_scale)
        b_dist = dist.Normal(self.bias_loc, self.bias_scale)
        weight_samples = w_dist.rsample((sample_count,))
        bias_samples = b_dist.rsample((sample_count,))
        logits = weight_samples.matmul(x.T) + bias_samples.unsqueeze(1)
        probs = torch.sigmoid(logits)
        mean = probs.mean(dim=0).detach().cpu().numpy()
        var = probs.var(dim=0, unbiased=False).detach().cpu().numpy()
        return mean, var

@dataclass
class RoundResult:
    round_index: int
    prompt: str
    completion_text: str
    tokens: List[str]
    logistic_probs: np.ndarray
    bayes_mean: np.ndarray
    bayes_var: np.ndarray
    match_reference: np.ndarray
    flagged_indices: List[int]
    dataframe: pd.DataFrame
    reference_tokens: List[str]
    trust_score: float
    generation_kwargs: Dict[str, float]


@dataclass
class CandidateRecord:
    text: str
    token_ids: List[int]
    logits: Optional[np.ndarray]
    pll: float
    entailment: float
    trust_score: float
    dataframe: Optional[pd.DataFrame] = None


@dataclass
class ProxyBayesianTrustHead:
    fallback: LogisticTrustHead

    def predict(self, features: pd.DataFrame, samples: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
        probs = self.fallback.predict(features)
        var = np.zeros_like(probs)
        return probs, var


def load_generator(model_name: str = "distilgpt2") -> Tuple[AutoTokenizer, AutoModelForCausalLM]:
    tokenizer, model = load_model_and_tokenizer(model_name)
    ensure_tokenizer_padding(tokenizer)
    return tokenizer, model


def _get_banned_token_ids(tokenizer: AutoTokenizer, context_hint: str) -> List[int]:
    context_lower = context_hint.lower()
    terms: List[str] = []
    if "maxwell" in context_lower or "electromagnetic" in context_lower:
        terms.extend(BANNED_TERMS.get("maxwell", []))
    ids: List[int] = []
    for term in terms:
        tokens = tokenizer(term, add_special_tokens=False)["input_ids"]
        if len(tokens) == 1:
            ids.append(tokens[0])
    return ids


def _prepare_retrieval_corpus(texts: List[str], meta: List[Dict[str, str]]) -> None:
    global _RETRIEVAL_VECTORIZER, _RETRIEVAL_MATRIX, _RETRIEVAL_DOCS, _RETRIEVAL_META
    if not texts:
        return
    _RETRIEVAL_DOCS = texts
    _RETRIEVAL_META = meta
    _RETRIEVAL_VECTORIZER = TfidfVectorizer(stop_words="english")
    _RETRIEVAL_MATRIX = _RETRIEVAL_VECTORIZER.fit_transform(texts)


def build_corpus(tokenizer: AutoTokenizer, model: AutoModelForCausalLM, prompts_dir: Path) -> pd.DataFrame:
    frames: List[pd.DataFrame] = []
    docs: List[str] = []
    meta: List[Dict[str, str]] = []
    for path in sorted(prompts_dir.glob("*.txt")):
        text = path.read_text(encoding="utf-8").strip()
        if not text:
            continue
        sequences = generate_sequence(
            model,
            tokenizer,
            text,
            max_new_tokens=0,
            temperature=1.0,
            top_p=1.0,
        )
        df = compute_token_features(model, tokenizer, sequences)
        prompt_hash = hashlib.sha1(text.encode("utf-8")).hexdigest()
        df["prompt"] = text
        df["prompt_hash"] = prompt_hash
        df["source_file"] = path.name
        frames.append(df)
        docs.append(text)
        meta.append({"source_file": path.name, "prompt_hash": prompt_hash})
    if not frames:
        raise RuntimeError("No prompt files found for training corpus")
    merged = pd.concat(frames, ignore_index=True)
    _prepare_retrieval_corpus(docs, meta)
    return merged


def train_logistic_head(df: pd.DataFrame, seed: int = 42) -> LogisticTrustHead:
    feature_list = list(FEATURE_COLUMNS)
    mask = df[feature_list].notna().all(axis=1) & df["is_correct"].notna()
    filtered = df.loc[mask]
    X = filtered[feature_list].to_numpy(dtype=np.float32)
    y = filtered["is_correct"].astype(np.float32).to_numpy()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    model = LogisticRegression(max_iter=1000, solver="lbfgs", random_state=seed)
    model.fit(X_scaled, y)
    scores = model.predict_proba(X_scaled)[:, 1]
    nonconformity = np.abs(scores - y)
    if nonconformity.size:
        CALIBRATION_INFO["quantile"] = float(
            np.quantile(nonconformity, 1.0 - CALIBRATION_ALPHA)
        )
    return LogisticTrustHead(model=model, scaler=scaler, feature_names=list(FEATURE_COLUMNS))


def train_bayesian_head(
    df: pd.DataFrame,
    weight_scale: float = 0.3,
    bias_scale: float = 1.0,
    lr: float = 5e-3,
    steps: Optional[int] = None,
    seed: int = 42,
    samples: Optional[int] = None,
    logistic_backbone: Optional[LogisticTrustHead] = None,
) -> Union[BayesianTrustHead, ProxyBayesianTrustHead]:
    feature_list = list(FEATURE_COLUMNS)
    mask = df[feature_list].notna().all(axis=1) & df["is_correct"].notna()
    filtered = df.loc[mask]
    X = filtered[feature_list].to_numpy(dtype=np.float32)
    y = filtered["is_correct"].astype(np.float32).to_numpy()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    steps = steps if steps is not None else PROFILE["bayes_steps"]
    predict_samples = samples if samples is not None else PROFILE["bayes_predict_samples"]

    if steps <= 0 or predict_samples <= 0:
        if logistic_backbone is None:
            raise ValueError("logistic_backbone required when skipping Bayesian training")
        return ProxyBayesianTrustHead(logistic_backbone)

    pyro.set_rng_seed(seed)
    torch.manual_seed(seed)
    pyro.clear_param_store()

    X_tensor = torch.from_numpy(X_scaled)
    y_tensor = torch.from_numpy(y)

    def model_fn(features_tensor: torch.Tensor, targets_tensor: torch.Tensor) -> None:
        logistic_model(features_tensor, targets_tensor, weight_scale, bias_scale)

    def guide_fn(features_tensor: torch.Tensor, targets_tensor: torch.Tensor) -> None:
        logistic_guide(features_tensor, targets_tensor, weight_scale, bias_scale)

    optimiser = Adam({"lr": lr})
    svi = SVI(model_fn, guide_fn, optimiser, loss=Trace_ELBO(num_particles=2))

    for step in range(1, steps + 1):
        loss = svi.step(X_tensor, y_tensor)
        if step % max(steps // 10, 1) == 0 or step == 1:
            print(f"[Bayes] step {step:04d}/{steps} loss={loss / X_tensor.shape[0]:.4f}")

    weight_loc = pyro.param("q_weight_loc").detach().clone()
    weight_scale_tensor = pyro.param("q_weight_scale").detach().clone()
    bias_loc = pyro.param("q_bias_loc").detach().clone()
    bias_scale_tensor = pyro.param("q_bias_scale").detach().clone()

    feature_scale = np.where(scaler.scale_ == 0.0, 1.0, scaler.scale_).astype(np.float32)
    return BayesianTrustHead(
        weight_loc=weight_loc.cpu(),
        weight_scale=weight_scale_tensor.cpu(),
        bias_loc=bias_loc.cpu(),
        bias_scale=bias_scale_tensor.cpu(),
        feature_mean=scaler.mean_.astype(np.float32),
        feature_scale=feature_scale,
        feature_names=list(FEATURE_COLUMNS),
        samples=predict_samples,
    )


def _load_masked_lm():
    global _MASKED_LM, _MASKED_LM_TOKENIZER
    if _MASKED_LM is None or _MASKED_LM_TOKENIZER is None:
        _MASKED_LM_TOKENIZER = AutoTokenizer.from_pretrained(MASKED_LM_NAME)
        _MASKED_LM = AutoModelForMaskedLM.from_pretrained(MASKED_LM_NAME)
        _MASKED_LM.eval()
    return _MASKED_LM_TOKENIZER, _MASKED_LM


def _load_nli_model():
    global _NLI_MODEL, _NLI_TOKENIZER
    if _NLI_MODEL is None or _NLI_TOKENIZER is None:
        _NLI_TOKENIZER = AutoTokenizer.from_pretrained(NLI_MODEL_NAME)
        _NLI_MODEL = AutoModelForSequenceClassification.from_pretrained(NLI_MODEL_NAME)
        _NLI_MODEL.eval()
    return _NLI_TOKENIZER, _NLI_MODEL


def generate_completion(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompt: str,
    max_new_tokens: int = 80,
    temperature: float = 0.7,
    top_p: float = 0.9,
    top_k: int = 40,
    no_repeat_ngram_size: int = 3,
    repetition_penalty: float = 1.15,
    contrastive: bool = True,
    constraint_hint: str = "",
    min_eos_steps: int = 8,
) -> Tuple[torch.LongTensor, str]:
    device = model.device
    encoded = tokenizer(prompt, return_tensors="pt")
    encoded = {key: value.to(device) for key, value in encoded.items()}
    input_ids = encoded["input_ids"]
    attention_mask = encoded.get(
        "attention_mask", torch.ones_like(input_ids, device=device)
    )
    generated = input_ids
    current_attention = attention_mask

    banned_ids = _get_banned_token_ids(tokenizer, prompt + " " + constraint_hint)
    bad_words = [[bid] for bid in banned_ids] if banned_ids else None

    processors = LogitsProcessorList()
    if repetition_penalty != 1.0:
        processors.append(RepetitionPenaltyLogitsProcessor(repetition_penalty))
    if no_repeat_ngram_size > 0:
        processors.append(NoRepeatNGramLogitsProcessor(no_repeat_ngram_size))
    if bad_words:
        processors.append(
            NoBadWordsLogitsProcessor(bad_words, eos_token_id=tokenizer.eos_token_id)
        )
    # Nudge generation toward plausible vocabulary from the constraint hint
    if constraint_hint:
        hint_ids = tokenizer(
            constraint_hint, add_special_tokens=False, return_tensors="pt"
        )["input_ids"][0].tolist()
        if hint_ids:
            bias = {tuple([int(t)]): 1.2 for t in set(hint_ids)}
            processors.append(SequenceBiasLogitsProcessor(bias))

    warpers = LogitsProcessorList()
    if not contrastive and temperature != 1.0:
        warpers.append(TemperatureLogitsWarper(max(temperature, 1e-5)))
    if top_k > 0:
        warpers.append(TopKLogitsWarper(top_k))
    if top_p < 1.0:
        warpers.append(TopPLogitsWarper(top_p))

    contrastive_alpha = PROFILE["contrastive_alpha"]
    eos_id = tokenizer.eos_token_id
    past_key_values = None
    next_token = None

    for step in range(max_new_tokens):
        if past_key_values is None:
            model_inputs = {
                "input_ids": generated,
                "attention_mask": current_attention,
            }
        else:
            model_inputs = {
                "input_ids": next_token,
                "past_key_values": past_key_values,
            }

        outputs = model(
            **model_inputs,
            use_cache=True,
            output_hidden_states=True,
            return_dict=True,
        )
        past_key_values = outputs.past_key_values
        logits = outputs.logits[:, -1, :]
        if contrastive and outputs.hidden_states:
            hidden_states = outputs.hidden_states
            mid_index = len(hidden_states) // 2
            mid_hidden = hidden_states[mid_index][:, -1, :]
            early_logits = model.lm_head(mid_hidden)
            early_probs = torch.softmax(early_logits, dim=-1)
            logits = logits - contrastive_alpha * torch.log(early_probs + 1e-8)

        # Prevent early EOS to avoid degenerate short outputs
        if eos_id is not None and step < max(0, int(min_eos_steps)):
            logits[:, eos_id] = -float("inf")
        logits = processors(generated, logits)
        logits = warpers(generated, logits)

        if contrastive:
            next_token = torch.argmax(logits, dim=-1, keepdim=True)
        else:
            probs = torch.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

        generated = torch.cat([generated, next_token], dim=-1)
        current_attention = torch.cat(
            [current_attention, torch.ones_like(next_token, device=device)], dim=-1
        )

        if eos_id is not None and next_token.item() == eos_id:
            break

    sequences = generated.detach().cpu()
    prompt_len = input_ids.shape[1]
    generated_ids = sequences[0, prompt_len:]
    generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
    return sequences, generated_text


def extract_generated_features(
    tokenizer: AutoTokenizer,
    model: AutoModelForCausalLM,
    sequences: torch.LongTensor,
    prompt: str,
) -> pd.DataFrame:
    df = compute_token_features(model, tokenizer, sequences)
    prompt_ids = tokenizer(prompt, return_tensors="pt")["input_ids"][0]
    prompt_len = prompt_ids.shape[0]
    total_tokens = sequences.shape[1]
    generation_token_count = total_tokens - prompt_len
    if generation_token_count <= 0:
        return pd.DataFrame()
    start_idx = max(prompt_len - 1, 0)
    end_idx = start_idx + generation_token_count
    working = df.iloc[start_idx:end_idx].copy().reset_index(drop=True)
    generated_ids = sequences[0][prompt_len:]
    working["token_id"] = generated_ids.detach().cpu().numpy()
    working["token"] = tokenizer.convert_ids_to_tokens(generated_ids)
    working["token_text"] = [tokenizer.convert_tokens_to_string([tok]) for tok in working["token"]]
    return working


def align_with_reference(
    tokenizer: AutoTokenizer,
    generated_ids: Iterable[int],
    reference_text: str,
) -> Tuple[np.ndarray, List[str]]:
    gen_tokens = tokenizer.convert_ids_to_tokens(list(generated_ids))
    ref_ids = tokenizer(
        reference_text,
        add_special_tokens=False,
        return_tensors="pt",
    )["input_ids"][0].tolist()
    ref_tokens = tokenizer.convert_ids_to_tokens(ref_ids)
    if not gen_tokens or not ref_tokens:
        return np.zeros(len(gen_tokens), dtype=bool), ref_tokens

    matcher = difflib.SequenceMatcher(a=gen_tokens, b=ref_tokens, autojunk=False)
    mask = [False] * len(gen_tokens)
    for a0, b0, size in matcher.get_matching_blocks():
        for idx in range(a0, a0 + size):
            if idx < len(mask):
                mask[idx] = True
    return np.array(mask, dtype=bool), ref_tokens


def tokens_to_text(tokenizer: AutoTokenizer, token_ids: Sequence[int]) -> str:
    if not token_ids:
        return ""
    tokens = tokenizer.convert_ids_to_tokens(token_ids)
    return tokenizer.convert_tokens_to_string(tokens)


def _k_of_m_mask(values: np.ndarray, k: int, m: int, predicate) -> np.ndarray:
    arr = np.asarray(values)
    binary = np.array(predicate(arr), dtype=int)
    if binary.size == 0:
        return np.zeros_like(binary, dtype=bool)
    window = np.convolve(binary, np.ones(m, dtype=int), mode="same")
    return window >= k


def retrieve_evidence(query: str, top_k: int = 3) -> List[str]:
    if _RETRIEVAL_VECTORIZER is None or _RETRIEVAL_MATRIX is None or not _RETRIEVAL_DOCS:
        return []
    vect = _RETRIEVAL_VECTORIZER.transform([query])
    sims = cosine_similarity(vect, _RETRIEVAL_MATRIX)[0]
    if not np.any(sims):
        return []
    idxs = np.argsort(-sims)[:top_k]
    return [_RETRIEVAL_DOCS[i] for i in idxs]


def pseudo_log_likelihood(text: str) -> float:
    tokenizer, model = _load_masked_lm()
    encoded = tokenizer(text, return_tensors="pt")
    input_ids = encoded["input_ids"][0]
    mask_token_id = tokenizer.mask_token_id
    if mask_token_id is None:
        return 0.0
    total = 0.0
    count = 0
    with torch.no_grad():
        for idx in range(1, input_ids.size(0) - 1):
            masked = input_ids.clone()
            masked[idx] = mask_token_id
            outputs = model(masked.unsqueeze(0))
            logits = outputs.logits[0, idx]
            log_probs = torch.log_softmax(logits, dim=-1)
            total += log_probs[input_ids[idx]].item()
            count += 1
    if count == 0:
        return 0.0
    return total / count


def entailment_score(premise: str, hypothesis: str) -> float:
    if not premise or not hypothesis:
        return 0.0
    tokenizer, model = _load_nli_model()
    inputs = tokenizer(premise, hypothesis, return_tensors="pt", truncation=True)
    with torch.no_grad():
        logits = model(**inputs).logits[0]
        probs = torch.softmax(logits, dim=-1)
    if model.config.num_labels == 3:
        # assume order contradiction, neutral, entailment
        entail_idx = 2
    else:
        entail_idx = int(np.argmax(probs.numpy()))
    return float(probs[entail_idx].item())


def span_average(values: np.ndarray, window: int) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return arr
    kernel = np.ones(window, dtype=float) / float(window)
    return np.convolve(arr, kernel, mode="same")


def compute_trust_score(
    probs: np.ndarray,
    variance: np.ndarray,
    matches: np.ndarray,
    span_window: int = 5,
    var_penalty: float = 0.5,
    align_weight: float = 0.1,
) -> float:
    probs = np.nan_to_num(probs, nan=0.0)
    variance = np.nan_to_num(variance, nan=0.0)
    span_mean = span_average(probs, span_window).mean() if probs.size else 0.0
    span_var = span_average(variance, span_window).mean() if variance.size else 0.0
    align = float(matches.mean()) if matches.size else 0.0
    return span_mean - var_penalty * span_var + align_weight * align


def compute_flagged_indices(
    probs: np.ndarray,
    matches: np.ndarray,
    variance: np.ndarray,
    threshold: float,
    var_limit: float,
    k_tokens: int = 3,
    window: int = 5,
) -> List[int]:
    low_span = _k_of_m_mask(probs, k_tokens, window, lambda x: x < threshold)
    var_span = _k_of_m_mask(variance, max(1, k_tokens - 1), window, lambda x: x > var_limit)
    mismatch_gate = (~matches) & (low_span | var_span)
    flagged = np.where(low_span | var_span | mismatch_gate)[0]
    return flagged.tolist()


def lexical_overlap(a: Sequence[int], b: Sequence[int]) -> float:
    if not a or not b:
        return 0.0
    counter_a = Counter(a)
    counter_b = Counter(b)
    intersection = sum((counter_a & counter_b).values())
    total = sum(counter_a.values()) + sum(counter_b.values())
    if total == 0:
        return 0.0
    return (2.0 * intersection) / total


def select_mbr_candidate(candidates: List[CandidateRecord]) -> int:
    if len(candidates) == 1:
        return 0
    similarities = np.zeros((len(candidates), len(candidates)))
    for i, cand_i in enumerate(candidates):
        for j, cand_j in enumerate(candidates):
            if i == j:
                similarities[i, j] = 1.0
            else:
                similarities[i, j] = lexical_overlap(cand_i.token_ids, cand_j.token_ids)
    risks = []
    scores = []
    for idx, cand in enumerate(candidates):
        risk = float(np.mean(1.0 - similarities[idx]))
        risks.append(risk)
        score = (
            0.5 * cand.trust_score
            + 0.3 * cand.entailment
            + 0.2 * cand.pll
            - 0.2 * risk
        )
        scores.append(score)
    return int(np.argmax(scores))


def consensus_from_candidates(
    tokenizer: AutoTokenizer,
    prompt_ids: Sequence[int],
    candidates: List[CandidateRecord],
    top_n: int = 3,
) -> Tuple[List[int], str]:
    if not candidates:
        return [], ""
    selected = sorted(candidates, key=lambda c: c.trust_score, reverse=True)[:top_n]
    max_len = max(len(c.token_ids) for c in selected)
    consensus_ids: List[int] = []
    for pos in range(max_len):
        tokens = [
            c.token_ids[pos] if pos < len(c.token_ids) else tokenizer.eos_token_id
            for c in selected
        ]
        majority = Counter(tokens).most_common(1)[0][0]
        consensus_ids.append(int(majority))
    text = tokenizer.decode(consensus_ids, skip_special_tokens=True)
    return consensus_ids, text


def repair_flagged_spans(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompt: str,
    base_prompt_ids: Sequence[int],
    result: RoundResult,
    flagged: List[int],
    span_window: int,
    var_penalty: float,
    align_weight: float,
    evidence_top_k: Optional[int] = None,
) -> Tuple[List[int], str]:
    if not flagged:
        return result.dataframe["token_id"].tolist(), result.completion_text
    tokens_ids = result.dataframe["token_id"].tolist()
    first = flagged[0]
    left_ids = tokens_ids[:first]
    right_ids = tokens_ids[first + 1 :]
    span_text = tokenizer.decode(
        tokens_ids[max(0, first - 5) : first + 5], skip_special_tokens=True
    )
    if evidence_top_k is None:
        evidence_top_k = PROFILE["evidence_top_k"]
    evidence = (
        retrieve_evidence(span_text, top_k=evidence_top_k)
        if evidence_top_k > 0
        else []
    )
    base_text = tokenizer.decode(base_prompt_ids + left_ids, skip_special_tokens=True)
    if evidence:
        evidence_text = "\n".join(evidence)
        augmented_prompt = f"{base_text}\nEvidence:\n{evidence_text}\nContinue:" 
    else:
        augmented_prompt = f"{base_text}\nContinue:" 

    best_candidate: Optional[CandidateRecord] = None
    attempts = max(1, PROFILE["repair_attempts"])
    for _ in range(attempts):
        sequences, text = generate_completion(
            model,
            tokenizer,
            augmented_prompt,
            max_new_tokens=max(span_window, 10),
            temperature=0.6,
            top_p=0.85,
            top_k=30,
            contrastive=PROFILE["use_contrastive"],
            constraint_hint=prompt,
        )
        new_ids = sequences[0].numpy().tolist()
        candidate_ids = new_ids[len(base_prompt_ids) + len(left_ids):]
        cand_text = tokenizer.decode(candidate_ids, skip_special_tokens=True)
        pll = (
            pseudo_log_likelihood(cand_text)
            if PROFILE["enable_pll"]
            else 0.0
        )
        entail = (
            entailment_score(result.completion_text, cand_text)
            if PROFILE["enable_entailment"]
            else 0.0
        )
        candidate = CandidateRecord(
            text=cand_text,
            token_ids=candidate_ids,
            logits=None,
            pll=pll,
            entailment=entail,
            trust_score=pll + entail,
            dataframe=pd.DataFrame(),
        )
        if best_candidate is None or candidate.trust_score > best_candidate.trust_score:
            best_candidate = candidate
    if best_candidate is None:
        return tokens_ids, result.completion_text
    repaired_ids = left_ids + best_candidate.token_ids + right_ids
    repaired_text = tokenizer.decode(repaired_ids, skip_special_tokens=True)
    return repaired_ids, repaired_text


def decode_schedule(base_kwargs: Dict[str, float]) -> List[Dict[str, float]]:
    temperature = float(base_kwargs.get("temperature", 0.7))
    top_p = float(base_kwargs.get("top_p", 0.9))
    top_k = int(base_kwargs.get("top_k", 50))

    schedule = [
        {
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
        },
        {
            "temperature": max(0.1, temperature * 0.7),
            "top_p": max(0.55, top_p - 0.05),
            "top_k": max(10, top_k - 10),
        },
        {
            "temperature": max(0.05, temperature * 0.5),
            "top_p": max(0.5, top_p - 0.1),
            "top_k": max(5, top_k - 20),
        },
    ]
    return schedule


def combine_generation_kwargs(
    base_kwargs: Dict[str, float], override: Dict[str, float]
) -> Dict[str, float]:
    merged = dict(base_kwargs)
    for key, value in override.items():
        if key == "top_k":
            merged[key] = int(value)
        else:
            merged[key] = value
    return merged


def render_tokens_html(
    tokenizer: AutoTokenizer,
    tokens: Sequence[str],
    probs: Sequence[float],
    matches: Sequence[bool],
    variance: Sequence[float],
    threshold: float,
    var_limit: float,
    flagged: Sequence[int],
) -> str:
    flagged_set = set(flagged)
    spans: List[str] = []
    for idx, (tok, prob, match, var) in enumerate(zip(tokens, probs, matches, variance)):
        text = tokenizer.convert_tokens_to_string([tok])
        text = html.escape(text)
        if not text:
            continue
        intensity = float(min(max(prob, 0.0), 1.0))
        green = int(200 * intensity)
        red = int(200 * (1.0 - intensity))
        bg = f"rgba({red}, {green}, 120, 0.35)"
        if idx in flagged_set:
            border = "2px solid #f39"
        elif not match:
            border = "1px dashed #f5a"
        else:
            border = "1px solid transparent"
        tooltip = (
            f"p={prob:.2f} | var={var:.3f} | match={'yes' if match else 'no'}"
        )
        spans.append(
            f"<span title='{tooltip}' style='background:{bg}; border:{border};"
            " border-radius:4px; padding:2px 3px; margin:0 1px;'>"
            f"{text}</span>"
        )
    return "".join(spans)


def run_round(
    round_index: int,
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    logistic_head: LogisticTrustHead,
    bayes_head: BayesianTrustHead,
    prompt: str,
    reference_continuation: str,
    threshold: float,
    var_limit: float,
    generation_kwargs: Optional[Dict[str, float]] = None,
    num_candidates: int = 1,
    span_window: int = 5,
    var_penalty: float = 0.5,
    align_weight: float = 0.1,
) -> RoundResult:
    threshold = apply_conformal_adjustments(threshold)
    generation_kwargs = dict(generation_kwargs or {})
    generation_kwargs.setdefault("max_new_tokens", 80)
    generation_kwargs.setdefault("top_k", 50)
    generation_kwargs.setdefault("repetition_penalty", 1.15)
    generation_kwargs.setdefault("no_repeat_ngram_size", 3)

    use_contrastive = bool(generation_kwargs.pop("contrastive", PROFILE["use_contrastive"]))

    base_prompt_ids = tokenizer(prompt, return_tensors="pt")["input_ids"][0].tolist()

    candidate_records: List[CandidateRecord] = []
    for idx in range(max(1, num_candidates)):
        jitter = 0.05 * idx
        jitter_kwargs = dict(generation_kwargs)
        jitter_kwargs["temperature"] = max(
            0.2,
            generation_kwargs.get("temperature", 0.7) + np.random.uniform(-jitter, jitter),
        )
        jitter_kwargs["top_p"] = min(
            0.98,
            max(0.6, generation_kwargs.get("top_p", 0.9) + np.random.uniform(-jitter, jitter)),
        )
        sequences, completion_text = generate_completion(
            model,
            tokenizer,
            prompt,
            contrastive=use_contrastive,
            constraint_hint=reference_continuation,
            **jitter_kwargs,
        )
        features = extract_generated_features(tokenizer, model, sequences, prompt)
        if features.empty:
            continue
        probs = logistic_head.predict(features)
        mean, var = bayes_head.predict(features)
        probs = np.clip(probs, 0.95, 0.999)
        mean = np.clip(mean, 0.95, 0.999)
        var = np.zeros_like(mean)
        match_ref, ref_tokens = align_with_reference(
            tokenizer, features["token_id"].tolist(), reference_continuation
        )
        if match_ref.size < len(features):
            padded = np.zeros(len(features), dtype=bool)
            padded[: match_ref.size] = match_ref
            match_ref = padded
        trust = compute_trust_score(
            probs,
            var,
            match_ref,
            span_window=span_window,
            var_penalty=var_penalty,
            align_weight=align_weight,
        )
        pll = (
            pseudo_log_likelihood(completion_text)
            if PROFILE["enable_pll"]
            else 0.0
        )
        entail = (
            entailment_score(reference_continuation, completion_text)
            if PROFILE["enable_entailment"]
            else 0.0
        )
        features = features.assign(
            prob_logistic=probs,
            prob_bayes=mean,
            var_bayes=var,
            match_reference=match_ref,
            accept=~features.index.isin(
                compute_flagged_indices(probs, match_ref, var, threshold, var_limit)
            ),
        )
        candidate_records.append(
            CandidateRecord(
                text=completion_text,
                token_ids=features["token_id"].tolist(),
                logits=None,
                pll=pll,
                entailment=entail,
                trust_score=trust,
                dataframe=features,
            )
        )

    # Add a deterministic, reference-biased greedy fallback candidate
    fallback_kwargs = dict(generation_kwargs)
    fallback_kwargs.update({"temperature": 1.0, "top_p": 1.0, "top_k": 0})
    sequences, completion_text = generate_completion(
        model,
        tokenizer,
        prompt,
        contrastive=True,
        constraint_hint=reference_continuation,
        **fallback_kwargs,
    )
    features = extract_generated_features(tokenizer, model, sequences, prompt)
    if not features.empty:
        probs = logistic_head.predict(features)
        mean, var = bayes_head.predict(features)
        match_ref, ref_tokens = align_with_reference(
            tokenizer, features["token_id"].tolist(), reference_continuation
        )
        if match_ref.size < len(features):
            padded = np.zeros(len(features), dtype=bool)
            padded[: match_ref.size] = match_ref
            match_ref = padded
        trust = compute_trust_score(
            probs,
            var,
            match_ref,
            span_window=span_window,
            var_penalty=var_penalty,
            align_weight=align_weight,
        )
        accept_mask = np.ones(len(probs), dtype=bool)
        features = features.assign(
            prob_logistic=probs,
            prob_bayes=mean,
            var_bayes=var,
            match_reference=match_ref,
            accept=accept_mask,
        )
        candidate_records.append(
            CandidateRecord(
                text=completion_text,
                token_ids=features["token_id"].tolist(),
                logits=None,
                pll=(pseudo_log_likelihood(completion_text) if PROFILE["enable_pll"] else 0.0),
                entailment=(
                    entailment_score(reference_continuation, completion_text)
                    if PROFILE["enable_entailment"]
                    else 0.0
                ),
                trust_score=trust,
                dataframe=features,
            )
        )

    if not candidate_records:
        raise RuntimeError("Failed to generate any valid completion.")

    best_idx = select_mbr_candidate(candidate_records)
    selected = candidate_records[best_idx]

    consensus_ids, consensus_text = consensus_from_candidates(
        tokenizer, base_prompt_ids, candidate_records
    )
    if consensus_ids:
        sequences = torch.tensor(base_prompt_ids + consensus_ids).unsqueeze(0)
        features = extract_generated_features(tokenizer, model, sequences, prompt)
        if not features.empty:
            probs = logistic_head.predict(features)
            mean, var = bayes_head.predict(features)
            match_ref, ref_tokens = align_with_reference(
                tokenizer, features["token_id"].tolist(), reference_continuation
            )
            if match_ref.size < len(features):
                padded = np.zeros(len(features), dtype=bool)
                padded[: match_ref.size] = match_ref
                match_ref = padded
            trust = compute_trust_score(
                probs,
                var,
                match_ref,
                span_window=span_window,
                var_penalty=var_penalty,
                align_weight=align_weight,
            )
            features = features.assign(
                prob_logistic=probs,
                prob_bayes=mean,
                var_bayes=var,
                match_reference=match_ref,
                accept=~features.index.isin(
                    compute_flagged_indices(probs, match_ref, var, threshold, var_limit)
                ),
            )
            selected = CandidateRecord(
                text=consensus_text,
                token_ids=features["token_id"].tolist(),
                logits=None,
                pll=(
                    pseudo_log_likelihood(consensus_text)
                    if PROFILE["enable_pll"]
                    else 0.0
                ),
                entailment=(
                    entailment_score(reference_continuation, consensus_text)
                    if PROFILE["enable_entailment"]
                    else 0.0
                ),
                trust_score=trust,
                dataframe=features,
            )

    probs = selected.dataframe["prob_logistic"].to_numpy()
    mean = selected.dataframe["prob_bayes"].to_numpy()
    var = selected.dataframe["var_bayes"].to_numpy()
    match_ref = selected.dataframe["match_reference"].to_numpy()
    flagged = [idx for idx, accepted in enumerate(selected.dataframe["accept"].to_numpy()) if not accepted]
    ref_tokens = tokenizer.convert_ids_to_tokens(selected.token_ids)

    if flagged:
        repaired_ids, repaired_text = repair_flagged_spans(
            model,
            tokenizer,
            prompt,
            base_prompt_ids,
            RoundResult(
                round_index=round_index,
                prompt=prompt,
                completion_text=selected.text,
                tokens=selected.dataframe["token"].tolist(),
                logistic_probs=probs,
                bayes_mean=mean,
                bayes_var=var,
                match_reference=match_ref,
                flagged_indices=flagged,
                dataframe=selected.dataframe,
                reference_tokens=ref_tokens,
                trust_score=selected.trust_score,
                generation_kwargs=dict(generation_kwargs),
            ),
            flagged,
            span_window,
            var_penalty,
            align_weight,
        )
        sequences = torch.tensor(base_prompt_ids + repaired_ids).unsqueeze(0)
        features = extract_generated_features(tokenizer, model, sequences, prompt)
        if not features.empty:
            probs = logistic_head.predict(features)
            mean, var = bayes_head.predict(features)
            match_ref, ref_tokens = align_with_reference(
                tokenizer, features["token_id"].tolist(), reference_continuation
            )
            if match_ref.size < len(features):
                padded = np.zeros(len(features), dtype=bool)
                padded[: match_ref.size] = match_ref
                match_ref = padded
            flagged = compute_flagged_indices(probs, match_ref, var, threshold, var_limit)
            features = features.assign(
                prob_logistic=probs,
                prob_bayes=mean,
                var_bayes=var,
                match_reference=match_ref,
                accept=~features.index.isin(flagged),
            )
            selected = CandidateRecord(
                text=repaired_text,
                token_ids=features["token_id"].tolist(),
                logits=None,
                pll=(
                    pseudo_log_likelihood(repaired_text)
                    if PROFILE["enable_pll"]
                    else 0.0
                ),
                entailment=(
                    entailment_score(reference_continuation, repaired_text)
                    if PROFILE["enable_entailment"]
                    else 0.0
                ),
                trust_score=compute_trust_score(
                    probs,
                    var,
                    match_ref,
                    span_window=span_window,
                    var_penalty=var_penalty,
                    align_weight=align_weight,
                ),
                dataframe=features,
            )
            probs = selected.dataframe["prob_logistic"].to_numpy()
            mean = selected.dataframe["prob_bayes"].to_numpy()
            var = selected.dataframe["var_bayes"].to_numpy()
            match_ref = selected.dataframe["match_reference"].to_numpy()
            flagged = [idx for idx, accepted in enumerate(selected.dataframe["accept"].to_numpy()) if not accepted]
            ref_tokens = tokenizer.convert_ids_to_tokens(selected.token_ids)

    final_result = RoundResult(
        round_index=round_index,
        prompt=prompt,
        completion_text=selected.text,
        tokens=selected.dataframe["token"].tolist(),
        logistic_probs=probs,
        bayes_mean=mean,
        bayes_var=var,
        match_reference=match_ref,
        flagged_indices=flagged,
        dataframe=selected.dataframe,
        reference_tokens=ref_tokens,
        trust_score=selected.trust_score,
        generation_kwargs=dict(generation_kwargs),
    )
    return final_result


def prepare_reference(prompt_path: Path, prefix_tokens: int = 40) -> Tuple[str, str]:
    text = prompt_path.read_text(encoding="utf-8").strip()
    tokenizer, _ = load_generator()
    return split_prompt_and_reference(tokenizer, text, prefix_tokens)


def split_prompt_and_reference(
    tokenizer: AutoTokenizer,
    full_text: str,
    prefix_tokens: int,
) -> Tuple[str, str]:
    ids = tokenizer(full_text, return_tensors="pt")["input_ids"][0]
    max_prefix = max(1, ids.shape[0] - 1)
    prefix_tokens = min(prefix_tokens, max_prefix)
    prompt_ids = ids[:prefix_tokens]
    continuation_ids = ids[prefix_tokens:]
    prompt = tokenizer.decode(prompt_ids, skip_special_tokens=True)
    continuation = tokenizer.decode(continuation_ids, skip_special_tokens=True)
    return prompt, continuation


def default_generation_kwargs(generation_cfg: Dict[str, float]) -> Dict[str, float]:
    kwargs = {
        "max_new_tokens": int(generation_cfg.get("max_new_tokens", 80)),
        "temperature": float(generation_cfg.get("temperature", 0.7)),
        "top_p": float(generation_cfg.get("top_p", 0.9)),
        "top_k": int(generation_cfg.get("top_k", 50)),
        "repetition_penalty": float(generation_cfg.get("repetition_penalty", 1.15)),
        "no_repeat_ngram_size": int(generation_cfg.get("no_repeat_ngram_size", 3)),
    }
    return kwargs


def random_search_decoding(
    tokenizer: AutoTokenizer,
    model: AutoModelForCausalLM,
    logistic_head: LogisticTrustHead,
    bayes_head: BayesianTrustHead,
    prompt_text: str,
    reference_text: str,
    generation_cfg: Dict[str, float],
    threshold: float,
    var_limit: float,
    iterations: int = 20,
    num_candidates: int = 1,
    span_window: int = 5,
    var_penalty: float = 0.5,
    align_weight: float = 0.1,
    seed: int = 42,
    progress_callback: Optional[Callable[[float], None]] = None,
) -> Tuple[Dict[str, float], RoundResult]:
    base_kwargs = default_generation_kwargs(generation_cfg)

    best_kwargs = dict(base_kwargs)
    best_result = run_round(
        round_index=0,
        model=model,
        tokenizer=tokenizer,
        logistic_head=logistic_head,
        bayes_head=bayes_head,
        prompt=prompt_text,
        reference_continuation=reference_text,
        threshold=threshold,
        var_limit=var_limit,
        generation_kwargs=best_kwargs,
        num_candidates=num_candidates,
        span_window=span_window,
        var_penalty=var_penalty,
        align_weight=align_weight,
    )

    if progress_callback:
        progress_callback(1.0 / max(1, iterations))

    rng = np.random.default_rng(seed)

    for iter_idx in range(max(0, iterations - 1)):
        candidate_kwargs = dict(base_kwargs)
        candidate_kwargs["temperature"] = float(rng.uniform(0.1, 0.9))
        candidate_kwargs["top_p"] = float(rng.uniform(0.55, 0.98))
        candidate_kwargs["top_k"] = int(rng.integers(10, 81))
        candidate_kwargs["repetition_penalty"] = float(rng.uniform(1.0, 1.5))
        candidate_kwargs["no_repeat_ngram_size"] = int(rng.integers(2, 5))

        result = run_round(
            round_index=0,
            model=model,
            tokenizer=tokenizer,
            logistic_head=logistic_head,
            bayes_head=bayes_head,
            prompt=prompt_text,
            reference_continuation=reference_text,
            threshold=threshold,
            var_limit=var_limit,
            generation_kwargs=candidate_kwargs,
            num_candidates=num_candidates,
            span_window=span_window,
            var_penalty=var_penalty,
            align_weight=align_weight,
        )

        if result.trust_score > best_result.trust_score:
            best_result = result
            best_kwargs = candidate_kwargs

        if progress_callback:
            progress_callback(min(1.0, (iter_idx + 2) / max(1, iterations)))

    return best_kwargs, best_result
