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

**Did my baseline-beating instincts hold up? Partly, and the part that didn't is the most important learning.** My instinct from the controlled datasets was "any model above 0.70 accuracy beats baseline." On real, imbalanced data the forest does beat 0.70 (0.790) — but the *real* battle is the minority (recall/F1/AUC), and the honest number is sharper than my instinct: the forest's ROC-AUC **0.800 vs logistic 0.759** carries a paired-bootstrap 95% CI of **[−0.013, +0.093]** — it **contains 0**, so on 200 test rows the two models are statistically indistinguishable. My instinct to "ship the winner" stayed, but only with a loudly-written caveat instead of a confident claim. What did hold up perfectly: the discipline — verify every number, prune with eyes open, and look at the actual mistakes.

## 2. The real-data differences worth remembering

1. **The fairness problem is not presentable.** In the synthetic week, "split it into is-married women etc." seemed principled. On real data, `personal_status` has **310/310** overlap between "female" and "div/dep/mar" — there is no marital-status feature separable from gender. The only honest option was **exclusion**, with a documented fairness section (Section 4) and a *canary test* in the suite asserting no gender/personal_status column reaches the models. I will check for *every* protected column the same way from now on.

2. **Pruning cost me the minority.** The depth-3 tree was "safer" (test acc 0.680 vs 0.660) but its bad-recall collapsed to **0.017** — 1 of 60 defaulters caught. A constraint that improved accuracy destroyed the decision-relevant metric, and only visible because recall was reported, not just accuracy. Lesson reinforced: on imbalanced data, evaluate the minority, always.

3. **Error analysis found a real, non-random pattern — with the statistics to grade it.** The shipped forest misclassified 42/200; bootstrap 95% CIs showed the mistake/correct gap on loan-size is **significant** ([+0.002, +0.574], excludes 0), while younger/longer-duration leans include 0 — directional, not conclusive at n=42 mistakes. Reporting "looks younger" would have been over-claiming; the CI forced the honest lower-intensity sentence.
4. **The fairness gap is real and measurable.** Female vs male default-rate gap +7.5 pts with bootstrap CI [+0.015, +0.136] (excludes 0) — the exclusion decision rests on a *statistically confirmed* gap, not a vibes-based precaution.

5. **Small test sets make metrics noisy.** 200 test rows → ROC-AUC of 0.759 vs 0.800 is a one-split comparison; the calibration curve, at five bins, holds ~20 points/bin in the middle. I state this as a limitation rather than over-claiming (and the suite pins tolerances rather than pretending precision).

6. **The "why not X" habit pays off.** Every engineering call (Section 5) documented its alternative (ordinal-vs-one-hot, log-vs-scaler, exclude-vs-flag). It made the run reviewable the way the synthetic week never needed to be.

## 3. What I would do differently next time

- **Cross-validation for model choice.** I picked the forest on a single stratified split (mirroring the week's method). On real data, the bootstrap CI showed forest-vs-logistic is *statistically indistinguishable* at n=200 — that is the quantified reason a 5-fold CV (or DeLong test on a bigger holdout) is the honest next step, not a nice-to-have.
- **Calibration plotting with error bars / more bins or grouped bins.** Five bins on 200 rows is qualitative; a production version would use more held-out data or isotonic calibration *and* re-check.
- **Keep `foreign_worker` decision explicit.** It stayed in the model (963/37, sensitive-adjacent) and shows up as the logistic's 2nd-largest coefficient — a reviewed product decision, not a silent inclusion.

## 4. What stuck

- Markdown never self-verifies → 55-check suite (`test_german_credit.py`) re-computes every headline number from the pinned CSV (fingerprint `38b6dbf6…`).
- Every claimed discovery (Sections 8–13) now cites its actual number, so the notebook cannot quietly drift from reality.
- The ungraded run held to the same bar as the graded week — because the habit, not the grade, is the thing being trained.

**Verdict:** real data made *calibration*, *fairness*, and *error patterns* concrete in a way synthetic data couldn't. The pipeline details transferred nearly unchanged; the judgment calls did not.

---

*Written to be re-read the night before any similar data task: "the numbers decide, not the brand of the model."*