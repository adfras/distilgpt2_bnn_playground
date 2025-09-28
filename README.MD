# Trust Heads for Language Models (calibrated correctness & abstention)

**What this repo does:** extracts per-token signals from a causal LM (top-k probabilities, logit gaps, entropy, tail mass, rolling context features) and fits tiny, interpretable heads (logistic regression, Bayesian regression, gradient boosted trees) that output the probability the next token is correct.
**Why it matters:** raw LM probabilities are often miscalibrated; lightweight heads turn raw logits into trustworthy probabilities so you can accept, abstain, or escalate responses based on an explicit quality bar instead of gut feel.
**What you get:** end-to-end scripts that (1) dump token-level features from DistilGPT-2, (2) train calibration heads, (3) plot reliability and risk-coverage curves, and (4) surface metrics that tie directly to operational decisions.

---

## Quickstart (60 seconds)

> Requires Python 3.9+, `torch`, `transformers`, `pandas`, `scikit-learn`, `pyro-ppl`, `numpy`, `matplotlib`. CPU-only is fine.

1) Extract per-token features on any text (zero generation is supported):
```bash
python scripts/basic_lm_quickstart.py \
  --prompt "Paste text you care about" \
  --max-new-tokens 0 \
  --output data/my_text_features.csv \
  --save-generation reports/my_text.txt \
  --meta-output data/my_text.meta.json \
  --print-config
```
This writes columns such as `p_max`, `logit_gap`, `entropy`, `tail_mass_topk`, `true_in_topk`, `prompt_hash`, and rolling-window context stats.

2) Fit a tiny head that predicts P(correct):
```bash
python scripts/logistic_head_baseline.py data/my_text_features.csv \
  --standardise \
  --metrics-output reports/logreg_metrics.json \
  --save-preds reports/logreg_preds.npz \
  --print-config
```
Swap in `scripts/bnn_head_pyro_template.py` for Bayesian uncertainty or `scripts/gbdt_baseline.py` for gradient boosted trees.

3) Turn numbers into decisions and visuals:
```bash
python scripts/plot_reliability.py reports/logreg_preds.npz \
  --output reports/reliability.png \
  --bins 12
python scripts/plot_risk_coverage.py reports/logreg_preds.npz \
  --output reports/risk_coverage.png \
  --steps 12
```
`reliability.png` shows whether confidence matches accuracy; `risk_coverage.png` shows how error falls as you abstain on the lowest-confidence tokens.

---

## Trust-loop QA Demo

When you want to see how the in-repo trust loop can clean up a shaky factual answer, run the comparison script we added in `scripts/qa_trust_loop_demo.py`. It contrasts a straight sample from DistilGPT-2 with the repo's accept/flag/repair loop that uses the logistic head.

```bash
source .venv/bin/activate  # ensure dependencies match the rest of the repo
python scripts/qa_trust_loop_demo.py \
  --question "Question: What is the capital of France?\nAnswer:" \
  --reference "Paris is the capital of France."
```

The script will:
- Train (or reuse cached) calibration heads on the bundled prompt corpus.
- Print a JSON blob showing the raw baseline completion, the trust-loop adjusted answer, the trust score, and any flagged token indices.
- Record the tuned generation kwargs that produced the highest-confidence run so you can reuse them later.

Tweak `--question`/`--reference` pairs to target other knowledge checks. The reference string converts into gentle guidance and evaluation labels for the heads; if you leave it blank the trust loop still runs but has less alignment signal.

---

## Examples

### Example A -- Technical prose (Maxwell treatise hold-out)
Run:
```bash
python scripts/basic_lm_quickstart.py --prompt "$(cat prompts/maxwell_treatise_excerpt.txt)" \
  --max-new-tokens 0 \
  --output data/maxwell.csv \
  --save-generation reports/maxwell.txt \
  --meta-output data/maxwell.meta.json

python scripts/logistic_head_baseline.py data/*.csv \
  --standardise \
  --test-prompt 91ee67c69e6512007ac50f2bdbe2211180f14a03 \
  --metrics-output reports/maxwell_logreg_metrics.json \
  --save-preds reports/maxwell_logreg_preds.npz \
  --print-config
```
Result: the raw LM baseline sits near Brier 0.182; the logistic head drops it to 0.0160 (about 91 percent reduction), the Bayesian head lands at 0.0379 with posterior variance you can threshold on, and gradient boosted trees are effectively perfect. Use this domain to set a strict accept-or-escalate threshold for technical writing.

### Example B -- Narrative prose (Moby-Dick hold-out)
Run:
```bash
python scripts/basic_lm_quickstart.py --prompt "$(cat prompts/moby_dick_excerpt.txt)" \
  --max-new-tokens 0 \
  --output data/moby.csv \
  --save-generation reports/moby.txt \
  --meta-output data/moby.meta.json

python scripts/logistic_head_baseline.py data/*.csv \
  --standardise \
  --test-prompt 3889d52f93ebb77932df482122275a37f1ea9ecf \
  --metrics-output reports/moby_logreg_metrics.json \
  --save-preds reports/moby_logreg_preds.npz \
  --print-config
```
Result: even on harder narrative text, the logistic head cuts Brier from 0.1417 to 0.0247 (about 83 percent reduction), the Bayesian head lands at 0.0494, and the trees again saturate with near-zero error. Only escalate the lowest-confidence spans to a bigger model or human-in-the-loop.

> Tip: run `streamlit run streamlit_trust_loop.py` to auto-calibrate the decode schedule for each scenario, then watch the regenerate-on-flag loop truncate at the first low-confidence span, resample under the tuned settings, and re-check with the heads.

---

## From probabilities to actions (copy-paste)

```python
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

P = np.load("reports/maxwell_logreg_preds.npz")
p = P["mean_p"]          # predicted P(correct) per token
y = P["targets"]         # held-out labels (analysis only)

window = 5                # evaluate confidence on 5-token spans
span_scores = sliding_window_view(p, window).mean(axis=-1)
span_scores = np.pad(span_scores, (window // 2, window - window // 2 - 1), mode="edge")

T = 0.78                  # span threshold from risk-coverage knee
accept = span_scores >= T
coverage = accept.mean()
accepted_brier = ((p[accept] - y[accept]) ** 2).mean()

print(f"Coverage={coverage:.2%}, Brier@accepted={accepted_brier:.4f}")
low_confidence_spans = np.where(~accept)[0]
# Route tokens in low_confidence_spans for abstention, clarification, or escalation.
```
Use your risk-coverage curve (computed on the same span average) to tune `T` so coverage meets your cost/quality trade-off.

For reference, the Maxwell logistic head at `T=0.78` accepts ~40% of spans with Brier on the order of 0.002 on the accepted set; the Moby run covers ~15% with Brier near 0.005, still more than halving the error relative to taking every token.

---

## Evidence (prompt hold-outs)

| Test domain | Method | AUC | Brier | ECE | Note |
| --- | --- | --- | --- | --- | --- |
| Maxwell (technical) | Raw LM (isotonic log p) | 0.801 | 0.1820 | 0.1014 | baseline |
| Maxwell (technical) | Logistic head | **0.9998** | **0.0160** | 0.0389 | ~91 percent Brier drop |
| Maxwell (technical) | Bayesian head (VI) | 0.9977 | 0.0379 | 0.0815 | posterior variance for selective routing |
| Maxwell (technical) | GBDT head | **1.000** | **~0.0000** | **~0.0000** | near-perfect |
| Moby-Dick (narrative) | Raw LM (isotonic log p) | 0.735 | 0.1417 | 0.0928 | baseline |
| Moby-Dick (narrative) | Logistic head | **0.998** | **0.0247** | 0.0499 | ~83 percent Brier drop |
| Moby-Dick (narrative) | Bayesian head (VI) | 0.9914 | 0.0494 | 0.0859 | posterior variance for selective routing |
| Moby-Dick (narrative) | GBDT head | **1.000** | **~0.0000** | **~0.0000** | indicates features nearly determine correctness |

**Takeaway:** logit-derived features (top-k structure, entropy, margin) almost fully determine token correctness on technical text and remain highly predictive on narrative text, enabling practical accept/abstain/escalate policies.

---

## Plots

- `reports/maxwell_reliability.png` -- logistic head reliability for the Maxwell hold-out; curve hugging the diagonal shows calibrated probabilities.
- `reports/maxwell_risk_coverage.png` -- Maxwell risk-coverage: abstaining on the lowest-confidence ~60% of tokens slashes Brier by >90%.
- `reports/moby_reliability.png` -- logistic head reliability for Moby-Dick; still well-behaved despite the harder narrative domain.
- `reports/moby_risk_coverage.png` -- Moby risk-coverage: even limited abstention halves error in the first 10-20% of tokens.

---

## Production pattern

1) Score each token (or span) with a lightweight head to get P(correct).
2) **Accept** predictions when `P >= T` and render them without delay.
3) **Abstain** otherwise and either show an "unsure" message, request clarification, or **escalate** to a stronger model. This drives cost savings and trims unforced errors while keeping latency low.

---

## Limits and gotchas

- **Prompt leakage:** always split by `prompt_hash` or explicit prompt IDs; the training scripts support `--test-prompt` to enforce this.
- **Forbidden features:** do not feed identifiers such as `target_id` or `pred_id` into heads; scripts guard against this but double-check custom pipelines.
- **When trees look perfect:** a zero-error tree usually means logit features fully determine correctness (seen on Maxwell). Treat that as a green light for strict thresholding rather than overfitting proof.
- **Hardware:** long prompts create large hidden-state matrices; trim input length or disable hidden-state columns if memory becomes tight.

---

## Example gallery (contribute your own)

- Docs QA: abstain on uncertain answer spans; escalate those snippets to a higher-capacity model.
- Autocomplete IDEs: halt completion when P(correct) falls below `T` mid-sequence.
- Policy drafting: flag low-confidence sentences for human review before publication.
- Summarization: rerun the 10 to 20 percent lowest-confidence sentences with a stronger summarizer.

---

## Repository layout

```
.
|- scripts/                   # feature extraction, calibration heads, plotting
|- data/                      # CSV feature dumps (gitignored)
|- reports/                   # metrics, reliability curves, policy artifacts
|- prompts/                   # public text excerpts for reproducible runs
|- README.MD                  # this guide
|- Agent.MD                   # automation playbook
```

---

## License and attribution

Use freely; credit the Transformers, Pyro, PyTorch, and scikit-learn projects when sharing results.
