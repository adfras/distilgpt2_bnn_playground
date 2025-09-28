# Agent Playbook (Agent.MD)

This document is a **contract for an automation agent** to set up and run experiments in this repo on a **CPU‑only machine**.

The agent MUST follow these tasks and produce the listed artifacts.

---

## 0) Assumptions
- OS: Linux or Windows with Python 3.9+ available.
- No GPU required.
- Internet access permitted to download model weights from Hugging Face.
- Shell examples are bash; translate for PowerShell on Windows.

---

## 1) Environment Setup (Task: `env_setup`)

**Goal:** Create an isolated environment and install dependencies.

**Steps:**
```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install --upgrade pip
pip install transformers torch pyro-ppl pandas scikit-learn
```

**Artifacts:**
- `.venv/` created
- `pip freeze > reports/pip_freeze.txt`

**Success criteria:**
- `python -c "import torch, transformers, pyro, pandas"` exits 0.

---

## 2) Feature Generation (Task: `build_features`)

**Goal:** Run DistilGPT‑2 to generate token‑level features from a prompt.

**Inputs (configurable):**
- `PROMPT` (string; default in script if not provided)
- `MAX_NEW_TOKENS` (int; default 80)
- `TEMPERATURE` (float; default 0.9)
- `TOP_P` (float; default 0.95)

**Command:**
```bash
python scripts/basic_lm_quickstart.py
```

**Artifacts:**
- `data/token_features.csv`
- Console log with the generated text (optionally tee to `reports/generation.txt`).

**Success criteria:**
- `data/token_features.csv` exists and has columns: `token, position, nll, h*`.

---

## 3) Train Bayesian Head (Task: `train_bnn_head`)

**Goal:** Fit a Bayesian logistic regression (VI) on token features and compute metrics.

**Inputs (configurable):**
- `FEATURES_CSV` (path; default `data/token_features.csv`)
- `VI_STEPS` (int; default 1500)
- `LR` (float; default 5e-3)
- `N_SAMPLES` (int; posterior MC samples; default 100)
- `NLL_PERCENTILE` (float; default 80; weak-label threshold)

**Command:**
```bash
python scripts/bnn_head_pyro_template.py ${FEATURES_CSV:-data/token_features.csv} | tee reports/train_log.txt
```

**Artifacts:**
- `reports/train_log.txt` (must include AUC and Brier lines)
- `reports/metrics.json` (see parsing step below)

**Parsing step (create metrics.json):**
Parse AUC and Brier from `reports/train_log.txt` and write JSON with keys:
```json
{
  "auc": <float>,
  "brier": <float>,
  "nll_percentile": <int>,
  "vi_steps": <int>,
  "lr": <float>,
  "n_samples": <int>,
  "prompt_hash": "<sha1 of prompt text if available>",
  "timestamp": "<ISO8601>"
}
```

**Success criteria:**
- `reports/metrics.json` exists with numeric fields
- AUC and Brier parsed successfully

---

## 4) Optional: Reliability Diagram (Task: `plot_reliability`)

**Goal:** Create a calibration plot from head predictions.

**Method (suggested):**
- Modify `bnn_head_pyro_template.py` to save `mean_p` and `y` arrays to `reports/preds.npz`.
- Then run a small Python snippet (matplotlib) to bin predictions and plot expected vs observed.

**Artifacts:**
- `reports/reliability.png`

**Success criteria:**
- PNG exists and has diagonal reference + binned curve.

---

## 5) Experiment Grid (Task: `grid_run`)

**Goal:** Run a small grid of experiments to explore settings on CPU.

**Grid (example):**
```yaml
experiments:
  - name: prompt_poem_vi1500_p80
    prompt: "Write a short poem about Bayesian neural networks:"
    nll_percentile: 80
    vi_steps: 1500
    n_samples: 100
    seed: 42

  - name: prompt_qa_vi3000_p85
    prompt: "Answer in two sentences: What is a Bayesian neural network?"
    nll_percentile: 85
    vi_steps: 3000
    n_samples: 200
    seed: 7

  - name: prompt_poem_vi1500_p90_pca
    prompt: "Write a short poem about Bayesian neural networks:"
    nll_percentile: 90
    vi_steps: 1500
    n_samples: 100
    seed: 1
    note: "Optional: apply PCA to hidden states before training"
```

**Procedure for each experiment:**
1. Edit `scripts/basic_lm_quickstart.py` to set the prompt (or pass via env var if you extend the script).
2. Run `env_setup` (if not already done).
3. Run `build_features`.
4. Run `train_bnn_head` with configured args.
5. Store `reports/metrics.json` as `reports/metrics_<name>.json` and (optionally) `reliability_<name>.png`.

**Success criteria:**
- One `metrics_*.json` per experiment; non‑empty and parseable.
- Optional reliability plots present for each experiment.

---

## 6) Reporting (Task: `summarize_results`)

**Goal:** Aggregate metrics across experiments into a single summary.

**Steps:**
- Load all `reports/metrics_*.json`.
- Produce `reports/summary.csv` with columns: `name, auc, brier, nll_percentile, vi_steps, n_samples, seed`.
- Print top‑line comparison and write a short markdown `reports/README.md` summarizing findings.

**Success criteria:**
- `reports/summary.csv` exists and contains at least two rows.
- Markdown summary created.

---

## 7) Compute & Stability Constraints

- CPU‑only; prefer **short prompts** and small grids.
- Keep VI steps in the **1.5k–5k** range for speed.
- For large hidden dims, consider **PCA** (retain 64–128 dims) before training the head.
- For reproducibility, set and log random **seeds**; capture `pip_freeze.txt`.

---

## 8) Failure Recovery

- If `data/token_features.csv` is missing, re‑run `build_features`.
- If metrics parsing fails, ensure the expected `AUC:` and `Brier:` lines appear in `train_log.txt`.
- If VI fails to converge, lower LR to `2e-3`, increase steps, or strengthen priors.

---

## 9) Optional Extensions

- Swap base LM to **Pythia-70M** or **Cerebras-GPT-111M** for scaling checks.
- Replace VI with **Laplace approximation** around MAP for speed.
- Add **MC-Dropout** head as a baseline.
- Save `preds.npz` and implement **reliability diagram** plotting.

---

## 10) QA Trust-loop Demo (Task: `qa_trust_loop`)

**Goal:** Compare a vanilla DistilGPT-2 completion with the calibrated trust loop so we can spot hallucinations quickly.

**Steps:**
```bash
source .venv/bin/activate
python scripts/qa_trust_loop_demo.py \
  --question "Question: What is the capital of France?\nAnswer:" \
  --reference "Paris is the capital of France." \
  --max-new-tokens 25 \
  --search-iters 12 \
  --num-candidates 3
```

**Artifacts:**
- Console JSON with `baseline` (raw decode) and `trust_loop` (calibrated decode).
- `trust_loop.generation_kwargs` in the JSON for reuse when running additional rounds.

**Success criteria:**
- `trust_loop.completion` differs from the baseline when low-confidence spans are flagged.
- `trust_loop.trust_score` is >= the baseline's implied confidence (inspect JSON).

Set `--reference ""` if you want to run purely on confidence without any alignment string; expect weaker guidance.

---

**End of Agent contract.**
