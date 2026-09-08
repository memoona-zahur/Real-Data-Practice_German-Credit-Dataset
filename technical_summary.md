# Technical Summary — German Credit: Did We Build a Loan-Risk Model That Works? (non-technical)

**What we were asked to do.** The week's earlier data was made-up, with a hidden answer key. This task says: run the *same* pipeline — look at the data, build a model, check it — on **1,000 real loan applications** (Statlog German Credit), where there is no hidden answer. Decide things for real and write down why.

**What the data looks like.** 1,000 people who applied for a loan. 20 facts about each person (age, loan amount, how long the loan runs, job, housing, savings, and 9 more). Each person was later labeled `good` (700 people, paid back fine) or `bad` (300 people, defaulted). Not a single missing value or duplicate row — this data was already clean; the "behind-the-scenes" problems were in the way the facts are worded, not in missing entries.

**One alarming column.** `personal_status` squashes gender and marital status into one value ("male single", "female div/dep/mar", …). We proved, row-by-row, that "female" and "div/dep/mar" are the *exact same 310 rows* — so any "marital" column is secretly a gender column. Female applicants also default noticeably more often (35.2% vs 27.7% of males — a gap our statistics showed is real, not a fluke). Because a loan model should not silently train on gender, we **took this column out entirely** and wrote a check that makes sure it never sneaks back in.

**What we did with the rest.** We turned words into numbers two different ways (ordered facts keep their order; named facts become on/off switches), lumped 4 rare loan purposes into one "other-special" bucket, and used the log of the loan amount because amounts were very skewed.

**Models we built and compared.** We always start with a "dumb" baseline that just predicts *everyone is good* — it scores 70% accuracy (echoing the 700/300 split) but catches **zero** defaulters, which shows accuracy alone can quietly lie. Then we trained a logistic regression (simple, readable rules), a decision tree, and a random forest (many trees working together). Every model was measured on test data it had never seen, on accuracy, precision, recall, F1, and ROC-AUC.

**Results.** The random forest was "best": 79% accuracy, catches 45% of defaulters, and ranks applicants best (ROC-AUC 0.80). But honesty check: a bootstrap 95% confidence band for the forest-vs-logistic ranking difference is **[−0.013, +0.093]** — it crosses zero, meaning *200 test rows are too few to say for certain which model is better*. We say that out loud rather than claiming a win.

**Why we trust our own numbers.** Plain-text explanations in notebooks are written by hand, so they can drift from reality. To stop that, an automated suite (`test_german_credit.py`) **recomputes every headline number independently from the raw file** — 55 checks — including the claims in the prose itself, the exact settings used (test size 20%, seed 42, 200 trees), and the statistics behind the fairness and error findings.

**What we learned (the point of the ungraded exercise).** Real data is not "dirtier"; it's *messier in judgment*: you must decide to drop a gender-adjacent column, decide which word-categories to merge, and resist overclaiming a model win the sample size can't support. The pipeline skills transferred unchanged; the judgment calls did not.

**Files:** `german_credit_practice.ipynb` (the full report), `test_german_credit.py` (55 verification checks), `REFLECTION.md` (honest lessons), `charts/` (3 figures), `data/credit_g.csv` (pinned raw data with fingerprint).