# REFLECTION — Real Data vs the Synthetic Week

**Run:** Real Data Practice · German Credit (Statlog credit-g) · Week 06 Bonus (ungraded)
**Date of run:** 2026-09-08 · **Environment:** pandas 2.3.3, numpy 2.2.6, scikit-learn 1.7.2, scipy 1.15.3

This week's main lab (Iterate–Diagnose–Defend) was built on *synthetic* data with a planted answer key. This bonus run repeats the same pipeline on **real, public credit data** — 1,000 loan applications, no planted defects. These are my honest notes on what was actually different.

## 1. What I expected vs what actually happened

| | I expected | What really happened |
|---|---|---|
| Messiness | Real data would be dirty (missing values, duplicates) | **Row-level clean:** 0 missing, 0 duplicates. The mess is elsewhere — categoricals, sparse levels, an entangled gender feature |
| Overfitting | Real data "more forgiving" than synthetic | **Exactly as overfit:** unconstrained tree hit train acc = 1.0000 yet only 0.6600 on test (depth 18, 148 leaves) |
| Forest gain | Modest edge over logistic | **Real, small edge:** ROC-AUC 0.800 vs 0.759 (+0.04) — and on *test* F1 (0.562) and accuracy (0.790) too |
| Calibration | Worse than synthetic | **Worse, as feared:** mid-range over-predicts (0.30 → actual 0.20); top bin timid (0.83 → actual 1.00) |

The biggest surprise was the tree. On synthetic data I "knew" the answer; here the unconstrained tree still memorized real data to the last row. Overfitting is not an artifact of clean synthetic data — it is structural.

## 2. The real-data differences worth remembering

1. **The fairness problem is not presentable.** In the synthetic week, "split it into is-married women etc." seemed principled. On real data, `personal_status` has **310/310** overlap between "female" and "div/dep/mar" — there is no marital-status feature separable from gender. The only honest option was **exclusion**, with a documented fairness section (§4) and a *canary test* in the suite asserting no gender/personal_status column reaches the models. I will check for *every* protected column the same way from now on.

2. **Pruning cost me the minority.** The depth-3 tree was "safer" (test acc 0.680 vs 0.660) but its bad-recall collapsed to **0.017** — 1 of 60 defaulters caught. A constraint that improved accuracy destroyed the decision-relevant metric, and only visible because recall was reported, not just accuracy. Lesson reinforced: on imbalanced data, evaluate the minority, always.

3. **Error analysis found a real, non-random pattern.** The shipped forest misclassified 42/200; mistakes were systematically younger (33.6 vs 37.2 y), larger loans (≈3,250 € vs 2,430 €), longer durations (23.4 vs 20.9 mo). On synthetic data errors were noise; here they name a segment — the higher-stress credit profile. That is the kind of finding a lender actually acts on.

4. **Small test sets make metrics noisy.** 200 test rows → ROC-AUC of 0.759 vs 0.800 is a one-split comparison; the calibration curve, at five bins, holds ~20 points/bin in the middle. I state this as a limitation rather than over-claiming (and the suite pins tolerances rather than pretending precision).

5. **The "why not X" habit pays off.** Every engineering call (§5) documented its alternative (ordinal-vs-one-hot, log-vs-scaler, exclude-vs-flag). It made the run reviewable the way the synthetic week never needed to be.

## 3. What I would do differently next time

- **Cross-validation for model choice.** I picked the forest on a single stratified split (mirroring the week's method). On real data a 5-fold CV would make the 0.759-vs-0.800 margin defensible, not just plausible.
- **Calibration plotting with error bars / more bins or grouped bins.** Five bins on 200 rows is qualitative; a production version would use more held-out data or isotonic calibration *and* re-check.
- **Keep `foreign_worker` decision explicit.** It stayed in the model (963/37, sensitive-adjacent) and shows up as the logistic's 2nd-largest coefficient — a reviewed product decision, not a silent inclusion.

## 4. What stuck

- Markdown never self-verifies → 36-check suite (`test_german_credit.py`) re-computes every headline number from the pinned CSV (fingerprint `38b6dbf6…`).
- Every claimed discovery (§8–§13) now cites its actual number, so the notebook cannot quietly drift from reality.
- The ungraded run held to the same bar as the graded week — because the habit, not the grade, is the thing being trained.

**Verdict:** real data made *calibration*, *fairness*, and *error patterns* concrete in a way synthetic data couldn't. The pipeline details transferred nearly unchanged; the judgment calls did not.

---

*Written to be re-read the night before any similar data task: "the numbers decide, not the brand of the model."*