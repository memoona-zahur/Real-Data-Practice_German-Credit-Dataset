# Real Data Practice — German Credit Dataset (Week 06 Bonus · Ungraded)

Applying this week's full pipeline — **baseline first → logistic → tree → forest → defend a winner → error analysis → calibration → statistics on the gaps** — to the real Statlog **German Credit** benchmark (credit-g): 1,000 loan applicants, 20 features, 700 `good` / 300 `bad` credit risk.

Self-paced bonus work. Holds the same bar as the graded week: every claimed number re-verified, every decision written with its *why* (and its *why-not-the-alternative*).

## Key results (test set, positive class = `bad`, stratified split 80/20, seed 42)

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| Baseline (most_frequent) | **0.700** | 0.000 | 0.000 | 0.000 | **0.500** |
| Logistic regression | 0.725 | 0.561 | 0.383 | 0.455 | 0.759 |
| Random forest (200 trees) | **0.790** | **0.750** | 0.450 | **0.562** | **0.800** |
| Decision tree (depth 3) | 0.680 | 0.167 | 0.017 | 0.030 | 0.692 |
| Decision tree (unconstrained) | 0.660 | 0.441 | **0.500** | 0.469 | 0.614 |

**Shipped model: Random forest** — wins ROC-AUC (0.800), F1 (0.562), precision (0.750) and accuracy (0.790); its only loss is recall (0.450 vs the unconstrained tree's 0.500), a trade the unconstrained tree pays for with full training memorization (train acc 1.000 → test 0.660).

**Statistical honesty you won't find in a basic run (all bootstrap 95% CI, seed 42):**
- `AUC(forest) − AUC(logistic)` = +0.040, CI **[−0.013, +0.093] — includes 0** → the two top models are *statistically indistinguishable* on 200 test rows; we ship the forest on point estimates and say so plainly (notebook Section 11).
- Gender default-rate gap = +7.5 pts (female 35.2% vs male 27.7%), CI **[+0.015, +0.136] — excludes 0** → the fairness concern that motivated exclusion is statistically measurable (notebook Section 4).
- Error pattern: only the loan-size gap is conclusive, CI [+0.002, +0.574]; age/duration leans include 0 → reported as directional, not fact (notebook Section 12).

## Deliverables

| File | What it is |
|---|---|
| `german_credit_practice.ipynb` | The full notebook — 16 sections, Restart-and-Run-All clean |
| `test_german_credit.py` | **75 adversarial checks** (data integrity → encoding/level integrity → determinism → fairness canary → every headline number + coefficient/importance/calibration/purpose prose re-computed → bootstrap CIs → parameter claims → charts reopen+overlap → notebook integrity) |
| `SELF_REVIEW.md` | Requirement-by-requirement review vs the task spec + severity-classified findings |
| `technical_summary.md` | Non-technical summary (a reader who never opens the notebook understands the whole story) |
| `REFLECTION.md` | Honest real-vs-synthetic reflection + what surprised me + whether instincts held |
| `charts/` | **8 PNG figures** (class balance, logistic coefficients, forest importances, overfit train-vs-test gap, model comparison, ROC curves, confusion matrix, calibration curve), each `layout="constrained"` and reopened/verified |
| `data/credit_g.csv` + `.sha256` | Pinned raw data (fingerprint `38b6dbf6…`, captured the byte-identity of the OpenML fetch) |
| `requirements.txt` | Pinned environment (pandas 2.3.3, scikit-learn 1.7.2, scipy 1.15.3, …) |

## How to reproduce + fresh-run proof

```bash
pip install -r requirements.txt
jupyter nbconvert --to notebook --execute --inplace german_credit_practice.ipynb
pytest test_german_credit.py -q
```

Verified after the final rebuild (this is the recorded evidence, not a "trust me" line):

```text
$ jupyter nbconvert --to notebook --execute --inplace german_credit_practice.ipynb   # ran twice
[NbConvertApp] Writing 510185 bytes to german_credit_practice.ipynb                  # exit 0, 0 errors
$ pytest test_german_credit.py -q
...........................................................................  [100%]
75 passed in 32.59s
```

Determinism: two consecutive cold executions produced identical metrics (`random_state=42` everywhere); the suite's independent recomputation agrees with every notebook number, including the prose.

## Two decisions worth knowing about

- **Fairness:** `personal_status` bakes gender into marital status (310/310 rows: `female` ≡ `div/dep/mar`), and the female default-rate gap is statistically significant. It is **excluded** from the models — and a canary test asserts no gender/`personal_status` column ever reaches the model (Section 4).
- **Winner by numbers, not reputation:** forest was chosen on ROC-AUC + F1 even though it "loses" recall to the unconstrained tree — because the tree's recall comes with train-test collapse; and the forest-vs-logistic edge is honestly reported as not-statistically-conclusive (honest Section 11 defense).

Run date: 2026-09-08. Env: pandas 2.3.3, numpy 2.2.6, matplotlib 3.10.9, scikit-learn 1.7.2, scipy 1.15.3.