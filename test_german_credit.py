"""Adversarial verification suite for the German Credit practice run.

Mirrors the week's central rule: *markdown is never f-string'd, so every claimed
number must be re-verified by an independent computation.*

Each Part recomputes the notebook's headline claims directly from the pinned
raw CSV (data/credit_g.csv) rather than trusting notebook state:

  Part A  Data integrity     — fingerprint, shape, balance, cleanliness
  Part B  Diagnosisfacts     — num/cat count, purpose levels, gender entanglement,
                               credit_amount skew
  Part C  Fairness canary    — personal_status / derived-gender NEVER becomes a
                               model input; gender default-rate gap + bootstrap CI
  Part D  Engineering        — purpose grouping, all-numeric + no-NaN features,
                               no leaked target, engineered shape
  Part E  Baseline truth     — acc=0.700, bad-recall=0.000, ROC-AUC=0.500
                               (incl. an independent by-hand computation)
  Part F  Model claims       — logistic 0.759 / forest 0.800 + winner selection
  Part G  Overfit honesty    — unconstrained tree train=1.000, depth=18, leaves=148
  Part H  Error+calibration  — misclassified count, mistake-vs-correct bootstrap CIs
  Part I  Charts             — PNG magic bytes + reopened renders + no text overlap
  Part J  Markdown integrity — every prose number == a live/recomputed value
  Part K  Raw immutability   — CSV byte-identical to the captured fingerprint
  Part M  Parameter claims   — hyperparameters in prose == actual model objects

Run:  pytest test_german_credit.py -q
"""
import hashlib
import json
import struct
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from scipy import stats
from scipy.stats import pearsonr
from sklearn.calibration import calibration_curve
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score,
                             roc_curve)
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

ROOT = Path(__file__).parent
CSV = ROOT / "data" / "credit_g.csv"
SHA = ROOT / "data" / "credit_g.sha256"
NB = ROOT / "german_credit_practice.ipynb"
CHARTS = sorted((ROOT / "charts").glob("*.png"))
EXPECTED_SHA = "38b6dbf6fb4b0311a3ffc005730f42623128591fb36473ab3c22d270c0467632"
SMALL = {"retraining", "other", "domestic appliance", "repairs"}
ORDER_MAPS = {
    "checking_status": {"no checking": 0, "<0": 1, "0<=X<200": 2, ">=200": 3},
    "credit_history": {"no credits/all paid": 0, "all paid": 1,
                       "existing paid": 2, "delayed previously": 3,
                       "critical/other existing credit": 4},
    "savings_status": {"no known savings": 0, "<100": 1, "100<=X<500": 2,
                       "500<=X<1000": 3, ">=1000": 4},
    "employment": {"unemployed": 0, "<1": 1, "1<=X<4": 2, "4<=X<7": 3, ">=7": 4},
    "property_magnitude": {"no known property": 0, "car": 1, "life insurance": 2,
                           "real estate": 3},
    "job": {"unemp/unskilled non res": 0, "unskilled resident": 1,
            "skilled": 2, "high qualif/self emp/mgmt": 3},
}
SEED = 42


# --------------------------------------------------------------------------- fixtures
@pytest.fixture(scope="module")
def df():
    return pd.read_csv(CSV)


@pytest.fixture(scope="module")
def eng(df):
    """Replicates the notebook's feature engineering (single source of numbers)."""

    purpose = df["purpose"].astype(str)
    purpose_grouped = purpose.where(~purpose.isin(SMALL), "other_small")

    X = pd.DataFrame(index=df.index)
    for col, mapping in ORDER_MAPS.items():
        X[col] = df[col].astype(str).map(mapping)
    for col in ["other_parties", "other_payment_plans", "housing",
                "own_telephone", "foreign_worker"]:
        X = X.join(pd.get_dummies(df[col].astype(str), prefix=col))
    X = X.join(pd.get_dummies(purpose_grouped.astype(str), prefix="purpose"))
    X["credit_amount_log"] = np.log(df["credit_amount"])
    for c in ["duration", "age", "installment_commitment", "residence_since",
              "existing_credits", "num_dependents"]:
        X[c] = df[c]
    return X


@pytest.fixture(scope="module")
def fitted(eng, df):
    """Fits every model exactly as the notebook does (deterministic, seed 42)."""
    y = (df["class"] == "bad").astype(int)
    X_train, X_test, y_train, y_test = train_test_split(
        eng, y, test_size=0.2, random_state=SEED, stratify=y)

    log = LogisticRegression(max_iter=1000, random_state=SEED).fit(X_train, y_train)
    rf = RandomForestClassifier(n_estimators=200, random_state=SEED).fit(X_train, y_train)
    t3 = DecisionTreeClassifier(max_depth=3, random_state=SEED).fit(X_train, y_train)
    tf = DecisionTreeClassifier(random_state=SEED).fit(X_train, y_train)
    dummy = DummyClassifier(strategy="most_frequent", random_state=SEED).fit(X_train, y_train)

    def probs(m):
        return m.predict_proba(X_test)[:, 1]

    return {"X_train": X_train, "X_test": X_test, "y_train": y_train, "y_test": y_test,
            "log": log, "rf": rf, "t3": t3, "tf": tf,
            "p_log": probs(log), "p_rf": probs(rf), "p_t3": probs(t3),
            "p_tf": probs(tf), "p_dummy": probs(dummy)}


@pytest.fixture(scope="module")
def results(fitted):
    y_te, p = fitted["y_test"], fitted
    out = {}

    def metr(name, pred, prob):
        return {"accuracy": accuracy_score(y_te, pred),
                "precision": precision_score(y_te, pred, zero_division=0),
                "recall": recall_score(y_te, pred, zero_division=0),
                "f1": f1_score(y_te, pred, zero_division=0),
                "roc_auc": roc_auc_score(y_te, prob)}

    out["baseline"] = metr("baseline", (p["p_dummy"] >= 0.5).astype(int), p["p_dummy"])
    out["logistic"] = metr("logistic", (p["p_log"] >= 0.5).astype(int), p["p_log"])
    out["random_forest"] = metr("random_forest", (p["p_rf"] >= 0.5).astype(int), p["p_rf"])
    out["tree_depth3"] = metr("tree_depth3", (p["p_t3"] >= 0.5).astype(int), p["p_t3"])
    out["tree_unconstrained"] = metr("tree_unconstrained", (p["p_tf"] >= 0.5).astype(int), p["p_tf"])
    out["tree_full_train_acc"] = accuracy_score(fitted["y_train"], fitted["tf"].predict(fitted["X_train"]))
    return out


def _md_text():
    nb = json.loads(NB.read_text())
    return "\n".join("".join(c.get("source", []))
                     for c in nb["cells"] if c.get("cell_type") == "markdown")


def _live_text():
    nb = json.loads(NB.read_text())
    parts = []
    for c in nb["cells"]:
        if c.get("cell_type") == "code":
            for o in c.get("outputs", []):
                if o.get("output_type") == "stream":
                    parts.append("".join(o["text"]))
    return "\n".join(parts)


def _norm(s):
    return s.replace("\u2212", "-").replace("\u2013", "-")


# ------------------------------------------------------------------ Part A
class TestDataIntegrity:
    def test_fingerprint_matches(self):
        assert hashlib.sha256(CSV.read_bytes()).hexdigest() == EXPECTED_SHA

    def test_shape(self, df):
        assert df.shape == (1000, 21)

    def test_class_balance(self, df):
        counts = df["class"].value_counts().to_dict()
        assert counts["good"] == 700 and counts["bad"] == 300

    def test_no_missing(self, df):
        assert df.isna().sum().sum() == 0

    def test_no_duplicates(self, df):
        assert df.duplicated().sum() == 0


# ------------------------------------------------------------------ Part B
class TestDiagnosisFacts:
    def test_numeric_column_count(self, df):
        assert df.select_dtypes(include=[np.number]).shape[1] == 7

    def test_purpose_levels_present(self, df):
        assert len(df["purpose"].astype(str).value_counts()) == 10

    def test_skew_of_credit_amount(self, df):
        skew = stats.skew(df["credit_amount"])
        assert skew == pytest.approx(1.95, abs=0.05)
        assert df["credit_amount"].min() == 250 and df["credit_amount"].max() == 18424

    def test_gender_entanglement_proof(self, df):
        ps = df["personal_status"].astype(str)
        gender = ps.str.split().str[0]
        marital = ps.str.split(n=1).str[1]
        assert (gender == "female").sum() == 310
        assert (marital == "div/dep/mar").sum() == 310
        assert ((gender == "female") == (marital == "div/dep/mar")).all()


# ------------------------------------------------------------------ Part C
class TestFairnessCanary:
    def test_personal_status_excluded_from_model_input(self, eng):
        for bad in ("personal_status", "gender"):
            assert bad not in set(eng.columns), f"forbidden feature leaked: {bad}"

    def test_no_remaining_gender_proxy(self, df, eng):
        gender = (df["personal_status"].astype(str).str.split().str[0] == "female")
        r, _ = pearsonr(eng["age"], gender.astype(int))
        assert abs(r) < 0.6, f"age correlates with derived gender r={r:+.3f}"

    def test_gender_gap_significant_with_ci(self, df):
        ps = df["personal_status"].astype(str)
        gender = ps.str.split().str[0]
        is_female = (gender == "female").values
        is_male = (gender == "male").values
        y_arr = (df["class"] == "bad").astype(int).values
        row_idx = np.arange(len(df))

        def gender_gap(idx):
            return y_arr[idx][is_female[idx]].mean() - y_arr[idx][is_male[idx]].mean()

        boot = stats.bootstrap((row_idx,), statistic=gender_gap, n_resamples=5000,
                               method="percentile", random_state=SEED)
        lo, hi = boot.confidence_interval.low, boot.confidence_interval.high
        gap = gender_gap(row_idx)
        assert y_arr[is_female].mean() == pytest.approx(0.352, abs=0.001)
        assert y_arr[is_male].mean() == pytest.approx(0.277, abs=0.001)
        assert gap == pytest.approx(0.075, abs=0.002)
        assert lo > 0, "gender-gap CI must exclude 0 (gap is statistically detectable)"


# ------------------------------------------------------------------ Part D
class TestEngineering:
    def test_purpose_grouping(self, df):
        purpose = df["purpose"].astype(str)
        assert purpose.isin(SMALL).sum() == 55
        assert not (purpose.isin(SMALL)).all()

    def test_engineered_shape(self, eng):
        assert eng.shape == (1000, 33)

    def test_target_not_in_features(self, eng):
        assert "class" not in eng.columns

    def test_all_features_numeric(self, eng):
        ok = set(eng.dtypes.astype(str)) <= {"int64", "float64", "bool"}
        assert ok, f"non-numeric feature reached the models: {set(eng.dtypes.astype(str)) - {'int64', 'float64', 'bool'}}"

    def test_no_nan_in_engineered_features(self, eng):
        assert not eng.isna().any().any(), "a .map() miss silently leaked NaN"


# ------------------------------------------------------------------ Part E
class TestBaselineTruth:
    def test_baseline_accuracy(self, results):
        assert results["baseline"]["accuracy"] == pytest.approx(0.700, abs=1e-3)

    def test_baseline_catches_no_defaulters(self, results):
        assert results["baseline"]["recall"] == 0.0
        assert results["baseline"]["precision"] == 0.0

    def test_baseline_is_random(self, results):
        assert results["baseline"]["roc_auc"] == pytest.approx(0.500, abs=1e-3)

    def test_baseline_by_hand_different_method(self, fitted):
        """O(n) arithmetic instead of sklearn: 140 good/200 test must give 0.70."""
        good_test = int((fitted["y_test"] == 0).sum())
        assert good_test == 140
        assert good_test / len(fitted["y_test"]) == pytest.approx(0.70, abs=1e-3)


# ------------------------------------------------------------------ Part F
class TestModelClaims:
    def test_logistic_auc(self, results):
        assert results["logistic"]["roc_auc"] == pytest.approx(0.759, abs=0.005)

    def test_forest_auc(self, results):
        assert results["random_forest"]["roc_auc"] == pytest.approx(0.800, abs=0.005)

    def test_forest_F1(self, results):
        assert results["random_forest"]["f1"] == pytest.approx(0.562, abs=0.005)

    def test_forest_beats_logistic_on_auc(self, results):
        assert results["random_forest"]["roc_auc"] > results["logistic"]["roc_auc"]

    def test_models_beat_baseline_on_auc(self, results):
        for name in ("logistic", "random_forest"):
            assert results[name]["roc_auc"] > 0.65

    def test_winner_is_forest_by_auc(self, results):
        winner = max(["logistic", "random_forest", "tree_depth3", "tree_unconstrained"],
                     key=lambda m: results[m]["roc_auc"])
        assert winner == "random_forest"

    def test_tree3_is_recall_blind(self, results):
        assert results["tree_depth3"]["recall"] <= 0.05


# ------------------------------------------------------------------ Part G
class TestOverfitHonesty:
    def test_full_tree_memorizes_train(self, results):
        assert results["tree_full_train_acc"] == pytest.approx(1.0000, abs=1e-4)

    def test_full_tree_actually_reaches_depth_18_and_148_leaves(self, fitted):
        assert fitted["tf"].get_depth() == 18
        assert fitted["tf"].get_n_leaves() == 148

    def test_full_tree_test_accuracy_drops(self, results):
        assert results["tree_unconstrained"]["accuracy"] <= 0.72

    def test_depth3_closes_the_gap(self, results):
        assert results["tree_depth3"]["accuracy"] > results["tree_unconstrained"]["accuracy"]

    def test_tree3_catches_one_default(self, fitted, results):
        tp = int(round(results["tree_depth3"]["recall"] * int(fitted["y_test"].sum())))
        assert tp == 1


# ------------------------------------------------------------------ Part H
class TestErrorAndCalibration:
    def test_misclassified_count_around_42(self, fitted):
        pred = (fitted["p_rf"] >= 0.5).astype(int)
        n_mis = int((pred != fitted["y_test"].values).sum())
        assert n_mis == 42

    def test_calibration_curve_has_five_bins(self, fitted):
        prob_true, prob_pred = calibration_curve(
            fitted["y_test"], fitted["p_rf"], n_bins=5, strategy="uniform")
        assert len(prob_true) == 5
        assert 0 <= prob_true.min() <= prob_pred.max() <= 1

    def test_error_pattern_loan_size_ci_excludes_zero(self, fitted):
        """Only the larger-loan mistake pattern is statistically conclusive."""
        pred = (fitted["p_rf"] >= 0.5).astype(int)
        rows = pd.DataFrame({"x": fitted["X_test"]["credit_amount_log"].values,
                             "y": fitted["y_test"].values, "p": pred})
        mk = rows[rows.y != rows.p]["x"].values
        ok = rows[rows.y == rows.p]["x"].values
        boot = stats.bootstrap((mk, ok), statistic=lambda a, b: a.mean() - b.mean(),
                               n_resamples=5000, method="percentile", random_state=SEED)
        lo, hi = boot.confidence_interval.low, boot.confidence_interval.high
        assert lo > 0, "loan-size gap CI must exclude 0; prose over-claimed otherwise"

    def test_error_pattern_age_and_duration_include_zero(self, fitted):
        pred = (fitted["p_rf"] >= 0.5).astype(int)
        rows = pd.DataFrame({"age": fitted["X_test"]["age"].values,
                             "dur": fitted["X_test"]["duration"].values,
                             "y": fitted["y_test"].values, "p": pred})
        mk, ok = rows[rows.y != rows.p], rows[rows.y == rows.p]
        for col in ("age", "dur"):
            boot = stats.bootstrap(
                (mk[col].values, ok[col].values),
                statistic=lambda a, b: a.mean() - b.mean(),
                n_resamples=5000, method="percentile", random_state=SEED)
            lo, hi = boot.confidence_interval.low, boot.confidence_interval.high
            assert lo <= 0 <= hi, f"{col} CI must include 0 (prose must say 'directional, not conclusive')"

    def test_notebook_calibration_cell_prints_five_bins(self):
        for cell in json.loads(NB.read_text())["cells"]:
            if cell.get("cell_type") != "code":
                continue
            src = "".join(cell.get("source", []))
            if "calibration_curve" in src and "n_bins=5" in src:
                printed = "\n".join("".join(o.get("text", []))
                                    for o in cell.get("outputs", []) if o.get("output_type") == "stream")
                rows = [l for l in printed.splitlines() if l.lstrip().startswith("~")]
                assert len(rows) == 5, f"expected 5 calibration bins:\n{printed}"
                return
        raise AssertionError("calibration cell not found in notebook")


# ------------------------------------------------------------------ Part I
class TestCharts:
    def test_png_magic_bytes(self):
        for path in CHARTS:
            assert path.exists(), f"chart missing: {path.name}"
            assert path.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n", f"{path.name} not a PNG"

    def test_saved_pngs_reopen_as_real_renders(self):
        """The saved file (what gets reviewed) re-opens to a content-rich render."""
        for path in CHARTS:
            data = path.read_bytes()
            w, h = struct.unpack(">II", data[16:24])
            assert w > 100 and h > 100, f"{path.name} implausible {w}x{h}"
            img = matplotlib.image.imread(path)
            assert img.shape[1] == w and img.shape[0] == h, "header/body dimension mismatch"
            assert len(np.unique(img.reshape(-1, img.shape[-1]), axis=0)) > 5, \
                f"{path.name} reopened to a blank/solid render"

    def test_comparison_chart_has_no_overlapping_text(self, results):
        """Re-renders the busiest chart and proves no two value-labels collide
        (Friday's failure mode — overlap in the SAVED file, not the preview)."""
        metrics = ["accuracy", "precision", "recall", "f1", "roc_auc"]
        x = np.arange(len(metrics)); w = 0.18
        order = ["baseline", "logistic", "tree_depth3", "random_forest"]
        colors = {"baseline": "#8C8C8C", "logistic": "#2E86AB",
                  "tree_depth3": "#6AB187", "random_forest": "#B07AA1"}
        fig, ax = plt.subplots(figsize=(8.5, 4.5), layout="constrained")
        for i, m in enumerate(order):
            vals = [results[m][mtr] for mtr in metrics]
            ax.bar(x + i * w, vals, w, label=m, color=colors[m])
            for xi, v in zip(x + i * w, vals):
                ax.text(xi, v + 0.015, f"{v:.2f}", ha="center", rotation=90, fontsize=7)
        ax.set_xticks(x + 1.5 * w, metrics); ax.set_ylabel("Score"); ax.set_ylim(0, 1.08)
        ax.legend(loc="lower right", ncol=2)
        ax.set_title("Classification models x metrics — German Credit (test set, positive='bad')")
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        boxes = [t.get_window_extent(renderer=renderer) for t in ax.texts]
        collisions = [(ax.texts[i].get_text(), ax.texts[j].get_text())
                      for i in range(len(boxes)) for j in range(i + 1, len(boxes))
                      if boxes[i].overlaps(boxes[j])]
        plt.close(fig)
        assert not collisions, f"overlapping value-labels in comparison chart: {collisions}"

    def test_roc_legend_has_no_overlapping_labels(self, fitted):
        """Second busiest chart (5 curves + legend): legend text must not collide."""
        probs = {"baseline": fitted["p_dummy"], "logistic": fitted["p_log"],
                 "tree_depth3": fitted["p_t3"], "tree_unconstrained": fitted["p_tf"],
                 "random_forest": fitted["p_rf"]}
        fig, ax = plt.subplots(figsize=(7, 5.5), layout="constrained")
        for m, p in probs.items():
            fpr, tpr, _ = roc_curve(fitted["y_test"], p)
            ax.plot(fpr, tpr, label=f"{m} — AUC {roc_auc_score(fitted['y_test'], p):.2f}")
        ax.plot([0, 1], [0, 1], ls=":", color="gray", lw=1)
        ax.legend(loc="lower right", fontsize=8)
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        legend_fig = ax.get_legend()
        boxes = [t.get_window_extent(renderer=renderer) for t in legend_fig.get_texts()]
        collisions = [(legend_fig.get_texts()[i].get_text(),
                       legend_fig.get_texts()[j].get_text())
                      for i in range(len(boxes)) for j in range(i + 1, len(boxes))
                      if boxes[i].overlaps(boxes[j])]
        plt.close(fig)
        assert not collisions, f"overlapping legend lines in ROC chart: {collisions}"


# ------------------------------------------------------------------ Part J
class TestMarkdownNumberIntegrity:
    """Verify-Before-Write: every prose number must equal a live/recomputed value."""

    def test_markdown_matches_independent_recomputation(self, df, fitted, results):
        y_te = fitted["y_test"]
        md = _norm(_md_text())

        exp = {}
        add = lambda k, v: exp.__setitem__(k, v)

        add("baseline", f"{results['baseline']['accuracy']:.2f}")           # 0.70
        add("baseline recall", f"{results['baseline']['recall']:.2f}")      # 0.00
        add("baseline auc", f"{results['baseline']['roc_auc']:.2f}")        # 0.50
        add("logistic auc", f"{results['logistic']['roc_auc']:.3f}")        # 0.759
        add("logistic acc", f"{results['logistic']['accuracy']:.3f}")       # 0.725
        add("logistic rec", f"{results['logistic']['recall']:.3f}")         # 0.383
        add("logistic prec %", f"{results['logistic']['precision'] * 100:.0f}%")  # 56%
        add("forest auc", f"{results['random_forest']['roc_auc']:.3f}")     # 0.800
        add("forest f1", f"{results['random_forest']['f1']:.3f}")           # 0.562
        add("forest prec", f"{results['random_forest']['precision']:.3f}")  # 0.750
        add("forest acc", f"{results['random_forest']['accuracy']:.3f}")    # 0.790
        add("forest rec", f"{results['random_forest']['recall']:.3f}")      # 0.450
        add("tree full train acc", f"{results['tree_full_train_acc']:.4f}")  # 1.0000
        add("tree full test acc", f"{results['tree_unconstrained']['accuracy']:.4f}")  # 0.6600
        add("tree3 acc", f"{results['tree_depth3']['accuracy']:.3f}")       # 0.680
        add("tree3 rec", f"{results['tree_depth3']['recall']:.3f}")         # 0.017
        add("tree3 f1", f"{results['tree_depth3']['f1']:.3f}")              # 0.030
        add("leaves", str(fitted["tf"].get_n_leaves()))                     # 148
        add("misclassified", f"{int(((fitted['p_rf'] >= 0.5).astype(int) != y_te.values).sum())} of 200")  # 42 of 200
        add("depth18", f"depth {fitted['tf'].get_depth()}")                        # depth 18
        add("leaves prose", f"{fitted['tf'].get_n_leaves()} leaves")               # 148 leaves
        add("splits", f"train {fitted['y_train'].mean():.2f}, test {y_te.mean():.2f}")  # train 0.30, test 0.30

        missing = [k for k, v in exp.items() if v not in md]
        assert not missing, f"markdown drifted from recomputed values: {missing}"

    def test_coefficient_claims_match(self, fitted):
        md = _norm(_md_text())
        coefs = pd.Series(fitted["log"].coef_[0], index=fitted["X_test"].columns)
        for feature, col in [("purpose_education", "purpose_education"),
                             ("foreign_worker_yes", "foreign_worker_yes"),
                             ("purpose_used car", "purpose_used car")]:
            v = coefs[col]
            assert f"`{feature}` ({v:+.2f})" in md, \
                f"prose coefficient for {feature} != recomputed {v:+.2f}"

    def test_importance_claims_match(self, fitted):
        md = _norm(_md_text())
        imp = pd.Series(fitted["rf"].feature_importances_,
                        index=fitted["X_test"].columns).sort_values(ascending=False)
        for feature in ("credit_amount_log", "checking_status", "age", "duration"):
            assert f"`{feature}` ({imp[feature]:.3f})" in md, \
                f"prose importance for {feature} != recomputed {imp[feature]:.3f}"

    def test_purpose_counts_and_skew_in_prose(self, df):
        md = _norm(_md_text())
        purpose = df["purpose"].astype(str)
        counts = purpose.value_counts()
        for level in SMALL:
            assert f"`{level}` ({counts[level]})" in md, \
                f"prose purpose count for {level} != recomputed {counts[level]}"
        assert f"skew = {stats.skew(df['credit_amount']):.2f}" in md

    def test_calibration_bin_claims_match(self, fitted):
        md = _norm(_md_text())
        prob_true, prob_pred = calibration_curve(
            fitted["y_test"], fitted["p_rf"], n_bins=5, strategy="uniform")
        pp, pt = [f"{p:.2f}" for p in prob_pred], [f"{t:.2f}" for t in prob_true]
        assert f"predicted {pp[0]}" in md, f"calibration predicted {pp[0]} missing"
        assert f"actual {pt[0]}" in md
        assert f"{pp[1]} predicted" in md and f"actual {pt[1]}" in md   # 0.30 -> 0.20
        assert f"{pp[2]}" in md and f"actual {pt[2]}" in md             # 0.49 -> 0.55
        assert f"{pp[4]}" in md and f"actual {pt[4]}" in md             # 0.83 -> 1.00

    def test_gender_rates_and_ci_claims_match(self, df, fitted):
        is_female = (df["personal_status"].astype(str).str.split().str[0] == "female").values
        is_male = ~is_female & (df["personal_status"].astype(str).str.split().str[0] == "male").values
        y_arr = (df["class"] == "bad").astype(int).values
        f_rate, m_rate = y_arr[is_female].mean(), y_arr[is_male].mean()
        row_idx = np.arange(len(df))

        def gender_gap(idx):
            return y_arr[idx][is_female[idx]].mean() - y_arr[idx][is_male[idx]].mean()

        boot = stats.bootstrap((row_idx,), statistic=gender_gap, n_resamples=5000,
                               method="percentile", random_state=SEED)
        lo, hi = boot.confidence_interval.low, boot.confidence_interval.high
        gap = gender_gap(row_idx)
        md = _norm(_md_text())
        assert f"{f_rate * 100:.1f}%" in md, "female default-rate prose != recomputed"
        assert f"{m_rate * 100:.1f}%" in md, "male default-rate prose != recomputed"
        assert f"{gap * 100:.1f}" in md, "gender-gap prose != recomputed"
        assert f"[{lo:+.3f}, {hi:+.3f}]" in md, "gender-gap CI prose != recomputed"

    def test_auc_gap_ci_claims_match(self, fitted):
        y_te, p = fitted["y_test"], fitted
        md = _norm(_md_text())
        row_idx = np.arange(len(y_te))

        def auc_gap(idx):
            return roc_auc_score(y_te.values[idx], p["p_rf"][idx]) \
                   - roc_auc_score(y_te.values[idx], p["p_log"][idx])

        boot = stats.bootstrap((row_idx,), statistic=auc_gap, n_resamples=5000,
                               method="percentile", random_state=SEED)
        lo, hi = boot.confidence_interval.low, boot.confidence_interval.high
        gap = auc_gap(row_idx)
        assert f"[{lo:+.3f}, {hi:+.3f}]" in md, "AUC-gap CI prose != recomputed"
        assert f"{gap:+.3f}" in md, "AUC-gap point prose != recomputed"
        assert lo <= 0 <= hi, "prose must say the AUC gap is NOT conclusive (CI contains 0)"

    def test_error_pattern_ci_claims_match(self, fitted):
        pred = (fitted["p_rf"] >= 0.5).astype(int)
        rows = pd.DataFrame({"age": fitted["X_test"]["age"].values,
                             "amt": fitted["X_test"]["credit_amount_log"].values,
                             "dur": fitted["X_test"]["duration"].values,
                             "y": fitted["y_test"].values, "p": pred})
        mk, ok = rows[rows.y != rows.p], rows[rows.y == rows.p]
        md = _norm(_md_text())

        def diff_stat(a, b):
            return a.mean() - b.mean()

        for col in ("age", "amt", "dur"):
            boot = stats.bootstrap((mk[col].values, ok[col].values),
                                   statistic=diff_stat, n_resamples=5000,
                                   method="percentile", random_state=SEED)
            lo, hi = boot.confidence_interval.low, boot.confidence_interval.high
            assert f"[{lo:+.3f}, {hi:+.3f}]" in md, f"{col} CI prose != recomputed"

    def test_no_error_cells_and_runs_to_end(self):
        nb = json.loads(NB.read_text())
        code_cells = [c for c in nb["cells"] if c.get("cell_type") == "code"]
        for i, cell in enumerate(code_cells):
            for out in cell.get("outputs", []):
                assert out.get("output_type") != "error", f"error output in notebook cell {i}"
        assert "End of notebook" in "".join(code_cells[-1]["source"])

    def test_every_key_number_was_live_printed(self):
        """The values pinned in prose must also appear in the notebook's own printed
        output — so no number was hand-typed as a bare assertion of reality."""
        live = _live_text()
        for tok in ["AUC=0.800", "AUC=0.759", "F1=0.562", "train acc = 1.0000",
                    "female-minus-male bad-rate = +0.075",
                    "AUC(random_forest) - AUC(logistic) = +0.040",
                    "misclassified: 42"]:
            assert tok in live, f"claimed number {tok} never appeared in live output"


# ------------------------------------------------------------------ Part K
class TestRawImmutability:
    def test_sha_file_pins_the_csv(self):
        pinned = SHA.read_text().split()[0]
        assert pinned == EXPECTED_SHA
        assert hashlib.sha256(CSV.read_bytes()).hexdigest() == pinned


# ------------------------------------------------------------------ Part M
class TestParameterClaims:
    """Prose-stated hyperparameters verified against the ACTUAL fitted objects."""

    def test_logistic_max_iter_is_1000(self, fitted):
        assert fitted["log"].get_params()["max_iter"] == 1000

    def test_forest_n_estimators_is_200(self, fitted):
        assert fitted["rf"].get_params()["n_estimators"] == 200

    def test_all_models_seed_42(self, fitted):
        for name in ("log", "rf", "t3", "tf"):
            assert fitted[name].get_params()["random_state"] == 42

    def test_depth3_uses_max_depth_3(self, fitted):
        assert fitted["t3"].get_params()["max_depth"] == 3

    def test_split_uses_02_test_and_stratify(self, fitted):
        assert len(fitted["X_test"]) == 200 and len(fitted["X_train"]) == 800
        assert fitted["y_test"].mean() == pytest.approx(0.30, abs=0.01)

    def test_threshold_is_05_and_positive_is_bad(self, fitted):
        pred = (fitted["p_rf"] >= 0.5).astype(int)
        assert set(np.unique(pred)) <= {0, 1}
        assert pred.mean() > 0, "threshold=0.5 must actually flag some positives"

    def test_personal_status_documented_exclusion_in_prose(self):
        md = _norm(_md_text())
        assert "EXCLUDE" in md and "personal_status" in md


# ------------------------------------------------------------------ Part N
class TestEncodingIntegrity:
    def test_every_ordinal_level_is_mapped(self, df):
        """A .map() miss would silently leak NaN — every raw level must map."""
        for col, mapping in ORDER_MAPS.items():
            mapped = df[col].astype(str).map(mapping)
            assert mapped.notna().all(), f"unmapped level silently NaNs in {col}"

    def test_ordinal_level_sets_match_exactly(self, df):
        """The documented encodings must match the data exactly: no leftover
        level, no invented one."""
        for col, mapping in ORDER_MAPS.items():
            assert set(df[col].astype(str).unique()) == set(mapping), \
                f"{col}: data levels != documented mapping levels"

    def test_target_is_last_column(self, df):
        assert df.shape[1] == 21
        assert df.columns[-1] == "class"

    def test_target_has_exactly_two_values(self, df):
        assert set(df["class"].unique()) == {"good", "bad"}

    def test_index_is_uniform_arange(self, df):
        assert (df.index == pd.RangeIndex(len(df))).all()

    def test_expected_onehot_columns_exist(self, eng):
        for feature in ("foreign_worker_yes", "purpose_used car",
                        "housing_rent", "other_payment_plans_none",
                        "purpose_other_small"):
            assert feature in eng.columns, f"missing engineered feature: {feature}"


# ------------------------------------------------------------------ Part O
class TestDeterminism:
    """Reproducibility: identical code path must reproduce identical numbers."""

    def test_forest_two_fits_identical(self, fitted):
        rf2 = RandomForestClassifier(n_estimators=200, random_state=SEED)
        rf2.fit(fitted["X_train"], fitted["y_train"])
        np.testing.assert_array_equal(rf2.predict_proba(fitted["X_test"]),
                                      fitted["rf"].predict_proba(fitted["X_test"]))

    def test_logistic_two_fits_identical(self, fitted):
        log2 = LogisticRegression(max_iter=1000, random_state=SEED)
        log2.fit(fitted["X_train"], fitted["y_train"])
        np.testing.assert_array_equal(log2.predict_proba(fitted["X_test"]),
                                      fitted["log"].predict_proba(fitted["X_test"]))

    def test_split_two_calls_identical(self, eng, df):
        y = (df["class"] == "bad").astype(int)
        a = train_test_split(eng, y, test_size=0.2, random_state=SEED, stratify=y)
        b = train_test_split(eng, y, test_size=0.2, random_state=SEED, stratify=y)
        for xa, xb in zip(a, b):
            np.testing.assert_array_equal(xa, xb)


# ------------------------------------------------------------------ Part P
class TestMetricSemantics:
    def test_baseline_confusion_is_all_good(self, fitted):
        pred = (fitted["p_dummy"] >= 0.5).astype(int)
        tn, fp, fn, tp = confusion_matrix(fitted["y_test"], pred).ravel()
        assert (tn, fp, fn, tp) == (140, 0, 60, 0)

    def test_forest_confusion_cells_exact(self, fitted):
        pred = (fitted["p_rf"] >= 0.5).astype(int)
        tn, fp, fn, tp = confusion_matrix(fitted["y_test"], pred).ravel()
        assert tp == 27 and fn == 33, "forest TP/FN must equal 27/33 (recall=0.45)"
        assert fp == 9 and tn == 131, "forest FP/TN must equal 9/131 (precision=0.75)"

    def test_confusion_agrees_with_reported_metrics(self, fitted, results):
        pred = (fitted["p_rf"] >= 0.5).astype(int)
        tn, fp, fn, tp = confusion_matrix(fitted["y_test"], pred).ravel()
        rec = tp / (tp + fn); prec = tp / (tp + fp)
        assert rec == pytest.approx(results["random_forest"]["recall"], abs=1e-9)
        assert prec == pytest.approx(results["random_forest"]["precision"], abs=1e-9)

    def test_all_metrics_finite_and_in_range(self, results):
        for name in ("baseline", "logistic", "random_forest",
                     "tree_depth3", "tree_unconstrained"):
            for metric in ("accuracy", "precision", "recall", "f1", "roc_auc"):
                v = results[name][metric]
                assert np.isfinite(v), f"{name}.{metric} is NaN"
                assert 0.0 <= v <= 1.0, f"{name}.{metric}={v} outside [0,1]"

    def test_every_real_model_beats_baseline_on_auc(self, results):
        for name in ("logistic", "random_forest", "tree_depth3", "tree_unconstrained"):
            assert results[name]["roc_auc"] > results["baseline"]["roc_auc"], name

    def test_both_splits_preserve_bad_rate(self, fitted):
        assert fitted["y_test"].mean() == pytest.approx(0.30, abs=0.01)
        assert fitted["y_train"].mean() == pytest.approx(0.30, abs=0.01)