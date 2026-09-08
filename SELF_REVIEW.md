# SELF_REVIEW — German Credit Practice Run

*Reviewer mode: I reviewed the deliverable as if a stranger submitted it — evidence-cited, requirements mapped one-by-one, findings severity-classified, then fixed in priority order. Author-vs-reviewer hat switched after completion.*

**Date:** 2026-09-08 · **Suite result at review time:** `pytest test_german_credit.py -q` → **75 passed** · Two cold `nbconvert --execute` re-runs → **0 error cells** (determinism + fresh-run proof).

---

## 1. Task spec (practice-data.md) — point by point

| # | Requirement (quoted/paraphrased) | Delivered | Evidence (file/section) |
|---|---|---|---|
| 1 | Fetched via `fetch_openml(name="credit-g", version=1, as_frame=True, parser="auto")` | ✓ exact call, on first run | notebook §1 md block + §1 code cell; cached to `data/credit_g.csv` + SHA-256 |
| 2 | `df` = 1,000 rows, 21 columns (20 features + `class`) | ✓ verified | notebook §2 output; `TestDataIntegrity::test_shape` |
| 3 | Diagnosis first: `.head()`, `.info()`, `.describe()`, value counts on categoricals, class balance | ✓ all five present in that order | notebook §2–§3 |
| 4 | No planted defects — describe what's genuinely there (clean at row level) | ✓ 0 missing / 0 duplicates stated & asserted | §2 md + `TestDataIntegrity` |
| 5 | Class imbalance 700/300 → baseline isn't 50/50, changes what "beating baseline" means | ✓ baseline chosen as most_frequent; 70% acc / 0% defaulters / AUC 0.50 discussed | §3, §7 |
| 6 | `personal_status` — **notice it, decide: include / exclude / flag, write the choice and why** | ✓ **EXCLUDE** with written why + why-not-flag; fairness evidence incl. 310/310 entanglement + statistically-confirmed gender gap | §4; canary tests `TestFairnessCanary` |
| 7 | "More than one `get_dummies()` call" / decide which categoricals get one-hot as-is | ✓ 6 `get_dummies` calls (5 nominal + grouped `purpose`); explicit per-column list | §5 + engineering cell |
| 8 | "A column you'd simplify or group" | ✓ `purpose` 10 levels → `other_small` grouping (55 rows) with reasons; sparse levels enumerated | §5 |
| 9 | Proper train/test split, own `random_state` (no fixed spec) | ✓ `random_state=42`, `test_size=0.2`, stratified | §6 |
| 10 | Baseline | ✓ `DummyClassifier(most_frequent)` | §7 |
| 11 | At least two real models — logistic **plus decision tree/random forest** | ✓ logistic + decision tree + random forest (three, beyond minimum) | §8–§10 |
| 12 | Report accuracy, precision, recall, F1, ROC-AUC **for each** | ✓ all five metrics for all five models (baseline, logistic, tree d3, tree full, forest) | §11 table |
| 13 | Error analysis — look at the model's actual mistakes | ✓ forest's 42/200 mistakes, numeric-feature comparison, **bootstrap CI grading each gap** | §12 |
| 14 | Calibration — are predicted probabilities trustworthy? | ✓ `calibration_curve` 5 bins + honest sparse-bin read on the *chosen* model | §13 |
| 15 | Short **honest reflection**: harder / surprised / did baseline-beating instincts hold? — all three answered | ✓ REFLECTION.md §1 (instincts), §2 hardened items, surprises table, verdict | `REFLECTION.md` |
| 16 | "MUST review these" (the 4 prior repos for patterns/structure) | ✓ patterns extracted & applied: verify-before-write, canaries, 8-point reviewer pass, layout=constrained + saved-PNG reopen, pinned env, fingerprint, evidence-cited claims | this doc §4 |

---

## 2. Universal Submission Validation

| Guardrail | Status | Evidence |
|---|---|---|
| Restart Kernel & Run All, exit 0, zero error cells | ✓ | two cold runs, 0 errors each (re-ran after final rebuild) |
| Determinism (same seed + logic → same result) | ✓ | metrics identical across consecutive cold runs; every model `random_state=42`; suite recomputation agrees |
| Fresh-run exit 0 **shown as evidence** | ✓ | README "Reproducibility" section records both commands + outputs |
| Dependencies pinned | ✓ | `requirements.txt` full pins (pandas 2.3.3, scikit-learn 1.7.2, scipy 1.15.3, …) |
| Every number from live variable / self-audit table | ✓ | notebook §14 self-audit (14 claims incl. CI bounds); suite `TestMarkdownNumberIntegrity` |
| **Markdown-number check = reads every markdown cell, asserts numbers match fresh recompute** | ✓ | `test_german_credit.py` Part J: recompute-then-assert-in-prose for all headline metrics + 4 CI sets + gender rates + percentages; plus "value was live-printed" cross-check |
| Decisions/parameters in prose asserted against objects, not just metrics | ✓ | Part M: `max_iter=1000`, `n_estimators=200`, `random_state=42`, `max_depth=3`, `test_size=0.2`, threshold 0.5 |
| Requirement → delivery mapped (none skipped) | ✓ | table above (16/16) |
| Adversarial checks on "looks correct" things | ✓ | fairness canary; NaN/non-numeric guards; by-hand baseline arithmetic (different method); CI zero-inclusion assertions on prose |
| Edge cases (empty/null/extremes) | ✓ | NaN + dtype guards; sparse categories grouped; imbalance; skew logged |
| Honest limitations with impact | ✓ | notebook §15 (7 bullets incl. quantified CI finding); REFLECTION.md |
| Charts: correct type, `layout="constrained"`, saved PNG reopened + overlap-free | ✓ | all 8 `layout="constrained"`; Part I reopens every PNG (magic bytes, header/body dims, non-blank) + re-renders the two busiest charts (metric comparison + ROC legend) and asserts **no text overlaps** |
| README + technical summary + self-review present | ✓ | README.md, technical_summary.md, this file |
| Git: feature branch, incremental commits, PR open | ✓ | `feature/real-data-practice-review` ← 3 incremental commits; PR #1 open |
| Adversarial + self-created checks run, all passing | ✓ | 75 passed |
| Clean git tree before submit | ✓ | working tree clean after `git commit` |

---

## 3. Reviewer findings during the pass (severity-classified) + fixes applied

**P1 — comparison gaps lacked statistics (fixed).** §4 gender gap, §11 forest-vs-logistic AUC gap, §12 mistake-vs-correct features were point-estimates in prose. Added **bootstrap 95% CIs** (scipy, percentile, seed 42, paired where required) to all three; prose now states zero-inclusion for each and the prose was *weakened* where the data demanded it (forest edge → "statistically indistinguishable"; age/duration leans → "directional, not conclusive").

**P1 — markdown-number verification was only 4 pins (fixed).** A 4-token check is weaker than the previous repo's Part T standard (a trajectory regression). Rebuilt Part J to recompute every headline metric, all four CI bands, gender rates/percentages, and error counts in the test and assert each appears in the notebook prose; added Part M so prose hyperparameters are checked against the actual fitted objects.

**P1 — "skew ≈ 1.95" was a hand-typed prose number (fixed).** Now computed live in the engineering cell (1.947) and the suite asserts it.

**P2 — §2 said "13 features are categorical" while `.info()` counts 14 object columns (fixed).** Prose now explains the target `class` is the 14th object column — no ambiguity for a skeptic.

**P2 — chart QA checked only magic bytes (fixed).** Part I now reopens each saved PNG (magic, header-vs-body dims, non-blank render) and re-renders the busiest chart to assert zero overlapping text labels.

**P2 — git done as one commit on main (fixed).** Moved to `feature/real-data-practice-review`, three incremental commits (notebook rigor → suite → docs), PR #1 opened with description + verification section.

**P1 — honest limitation added (fixed).** §15 now quantifies the "single split" limitation with the actual CI ([−0.013, +0.093] contains 0) instead of a generic sentence.

No P0 (blocks-submission) findings — no wrong numbers, no error cells, no missing spec points were found at review time.

---

## 4. Best-practice transfer from the four reference repos

| Lesson (source repo) | Applied here |
|---|---|
| Verify-before-write; markdown numbers need their own automated check (Iterate-Diagnose-Defend Part T) | Part J + Part M |
| Adversarial suite w/ canaries + raw-immutability + PNG magic bytes (EDA-Assessment) | TestFairnessCanary, TestRawImmutability, TestCharts |
| Narrative notebook = report; live f-strings + self-audit; baseline-first (First-ML-Pipeline) | §14 self-audit, question→chart→finding throughout, §7 baseline |
| Reviewer pass: evidence-cited, severity-classified, fix in copy, adapt checklist to project (EDA-Peer-Review-Lab) | this document + statistical-rigor additions |
| Shuffle-test / relationship-vs-comparison, bootstrap CI on gaps (EDA-Peer-Review-Lab #8) | bootstrap CIs on the three key comparison gaps (this project has no *relationship* charts, so #1/#2/#4/#5 of the 8-point list are n/a by design) |

---

## 5. Project-specific additions (adapted reviewer checklist for a *ML* task)

- [x] Every claimed headline metric re-computed from the pinned CSV (not the notebook's memory)
- [x] Hyperparameters in prose == actual model objects (Part M)
- [x] Chosen-model test on **imbalance-critical** metrics (bad-recall, F1, AUC), not just accuracy
- [x] Fairness: feature excluded AND canary asserts no leak; no strong gender proxy among remaining numerics
- [x] Overfit honesty pinned on real objects: depth=18, leaves=148, train acc=1.0000 (Part G)
- [x] Calibration assessed on the **shipped** model only, with sparse-bin caveat
- [x] Determinism proven by consecutive cold re-runs + byte-fingerprinted raw data

## 6. Reviewer's note (summary verdict)

What's genuinely good: the whole artifact runs cold with zero errors, every prose number is now anchored to an independent recomputation (75 checks including the prose itself, plus coefficient/importance/calibration and encoding-level integrity), the fairness call is evidence-first (entanglement proof + a significant gap) rather than policy-talk, and the two statistical surprises — the forest edge is *not* significant at n=200, and only the loan-size error pattern is conclusive — are reported honestly instead of spun. Top fixes applied in priority order: bootstrap CIs on every comparison gap, full markdown-number verification, live skew, chart-QA reopen+overlap proof, and a proper feature-branch/PR git flow. As a stranger reading top-to-bottom, the story is complete: question → method → numbers → interpretation → limitation, at every step.