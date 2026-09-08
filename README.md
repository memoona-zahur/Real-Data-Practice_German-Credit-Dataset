# Real Data Practice — German Credit Dataset (Week 06 Bonus · Ungraded)

Applying this week's full pipeline — **baseline first → logistic → tree → forest → defend a winner → error analysis → calibration** — to the real Statlog **German Credit** benchmark (credit-g): 1,000 loan applicants, 20 features, 700 `good` / 300 `bad` credit risk.

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

## Deliverables

| File | What it is |
|---|---|
| `german_credit_practice.ipynb` | The full notebook — 16 sections, Restart-and-Run-All clean |
| `test_german_credit.py` | 36 adversarial checks (data integrity → fairness canary → every headline number → charts → notebook integrity) |
| `REFLECTION.md` | Honest real-vs-synthetic reflection + what I'd do differently |
| `charts/` | 3 PNG figures (class balance, model comparison, calibration curve) |
| `data/credit_g.csv` + `.sha256` | Pinned raw data (fingerprint `38b6dbf6…`) |
| `requirements.txt` | Pinned environment (pandas 2.3.3, scikit-learn 1.7.2, …) |

## How to reproduce

```bash
pip install -r requirements.txt
jupyter nbconvert --to notebook --execute --inplace german_credit_practice.ipynb
pytest test_german_credit.py -q     # 36 passed
```

## Two decisions worth knowing about

- **Fairness:** `personal_status` bakes gender into marital status (310/310 rows: `female` ≡ `div/dep/mar`). It is **excluded** from the models and a canary test asserts no gender/`personal_status` column ever reaches the model (§4).
- **Winner by numbers, not reputation:** forest was chosen on ROC-AUC + F1 even though it "loses" recall to the unconstrained tree — because the tree's recall comes with train-test collapse (honest §11 defense).

Run date: 2026-09-08. Env: pandas 2.3.3, numpy 2.2.6, matplotlib 3.10.9, scikit-learn 1.7.2, scipy 1.15.3.