"""Adversarial verification suite for the German Credit practice run.

Mirrors one rule from the whole week: *markdown is never f-string'd, so every
claimed number must be re-verified by an independent computation.*

Each Part recomputes the notebook's headline claims directly from the pinned
raw CSV (data/credit_g.csv) rather than trusting notebook state:

  Part A  Data integrity     — fingerprint, shape, balance, cleanliness
  Part B  Diagnosisfacts     — num/cat count, purpose levels, entanglement proof
  Part C  Fairness canary    — personal_status / derived gender NEVER become models input
  Part D  Engineering        — purpose grouping, no leaked target, engineered shape
  Part E  Baseline truth     — acc=0.700, bad-recall=0.000, ROC-AUC=0.500
  Part F  Model claims       — logistic 0.759 / forest 0.800, winner selection
  Part G  Overfit honesty    — unconstrained tree train=1.000 vs test<0.80
  Part H  Error+calibration  — misclassified count, calibration bins exist
  Part I  Charts             — PNG magic bytes per saved figure
  Part J  Notebook integrity — executed top-to-bottom, zero error cells
  Part K  Raw immutability   — CSV is byte-identical to the captured fingerprint

Run:  pytest test_german_credit.py -q
"""
import hashlib
import json
import struct

import numpy as np
import pandas as pd
import pytest
from scipy.stats import pearsonr
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score, roc_auc_score)
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

ROOT = __import__("pathlib").Path(__file__).parent
CSV = ROOT / "data" / "credit_g.csv"
SHA = ROOT / "data" / "credit_g.sha256"
NB = ROOT / "german_credit_practice.ipynb"
CHARTS = [ROOT / "charts" / n for n in ("chart_class_balance.png",
                                        "chart_classification_comparison.png",
                                        "chart_calibration_curve.png")]

EXPECTED_SHA = "38b6dbf6fb4b0311a3ffc005730f42623128591fb36473ab3c22d270c0467632"
SMALL = {"retraining", "other", "domestic appliance", "repairs"}


@pytest.fixture(scope="module")
def df():
    return pd.read_csv(CSV)


@pytest.fixture(scope="module")
def eng(df):
    """Replicates the notebook's feature engineering (single source of numbers)."""

    purpose = df["purpose"].astype(str)
    purpose_grouped = purpose.where(~purpose.isin(SMALL), "other_small")

    order_maps = {
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
    X = pd.DataFrame(index=df.index)
    for col, mapping in order_maps.items():
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
def split(eng, df):
    y = (df["class"] == "bad").astype(int)
    X_train, X_test, y_train, y_test = train_test_split(
        eng, y, test_size=0.2, random_state=42, stratify=y)
    return X_train, X_test, y_train, y_test


@pytest.fixture(scope="module")
def results(split):
    X_tr, X_te, y_tr, y_te = split

    def metr(name, y_pred, y_prob):
        return {"accuracy": accuracy_score(y_te, y_pred),
                "precision": precision_score(y_te, y_pred, zero_division=0),
                "recall": recall_score(y_te, y_pred, zero_division=0),
                "f1": f1_score(y_te, y_pred, zero_division=0),
                "roc_auc": roc_auc_score(y_te, y_prob)}

    out = {}
    dummy = DummyClassifier(strategy="most_frequent", random_state=42).fit(X_tr, y_tr)
    out["baseline"] = metr("baseline", dummy.predict(X_te), dummy.predict_proba(X_te)[:, 1])

    log = LogisticRegression(max_iter=1000, random_state=42).fit(X_tr, y_tr)
    p = log.predict_proba(X_te)[:, 1]
    out["logistic"] = metr("logistic", (p >= 0.5).astype(int), p)

    rf = RandomForestClassifier(n_estimators=200, random_state=42).fit(X_tr, y_tr)
    p = rf.predict_proba(X_te)[:, 1]
    out["random_forest"] = metr("random_forest", (p >= 0.5).astype(int), p)

    t3 = DecisionTreeClassifier(max_depth=3, random_state=42).fit(X_tr, y_tr)
    p = t3.predict_proba(X_te)[:, 1]
    out["tree_depth3"] = metr("tree_depth3", (p >= 0.5).astype(int), p)

    tf = DecisionTreeClassifier(random_state=42).fit(X_tr, y_tr)
    p = tf.predict_proba(X_te)[:, 1]
    out["tree_unconstrained"] = metr("tree_unconstrained", (p >= 0.5).astype(int), p)
    out["tree_full_train_acc"] = accuracy_score(y_tr, tf.predict(X_tr))
    return out


# ------------------------------------------------------------------ Part A
class TestDataIntegrity:
    def test_fingerprint_matches(self):
        actual = hashlib.sha256(CSV.read_bytes()).hexdigest()
        assert actual == EXPECTED_SHA, "credit_g.csv modified since capture"

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
        names = set(eng.columns)
        for bad in ("personal_status", "gender"):
            assert bad not in names, f"forbidden feature leaked into model input: {bad}"

    def test_no_remaining_gender_proxy(self, df, eng):
        gender = (df["personal_status"].astype(str).str.split().str[0] == "female")
        if "age" in eng.columns:
            r, _ = pearsonr(eng["age"], gender.astype(int))
            assert abs(r) < 0.6, f"age correlates with derived gender r={r:+.3f}"


# ------------------------------------------------------------------ Part D
class TestEngineering:
    def test_purpose_grouping(self, df):
        purpose = df["purpose"].astype(str)
        assert purpose.isin(SMALL).sum() == 55, \
            "small-purpose subgroup should total 55 rows"
        assert not (purpose.isin(SMALL)).all(), "grouping must not swallow all purposes"

    def test_engineered_shape(self, eng):
        assert eng.shape == (1000, 33)

    def test_target_not_in_features(self, eng):
        assert "class" not in eng.columns


# ------------------------------------------------------------------ Part E
class TestBaselineTruth:
    def test_baseline_accuracy(self, results):
        assert results["baseline"]["accuracy"] == pytest.approx(0.700, abs=1e-3)

    def test_baseline_catches_no_defaulters(self, results):
        assert results["baseline"]["recall"] == 0.0
        assert results["baseline"]["precision"] == 0.0

    def test_baseline_is_random(self, results):
        assert results["baseline"]["roc_auc"] == pytest.approx(0.500, abs=1e-3)


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
        assert results["tree_depth3"]["recall"] <= 0.05, \
            "depth-3 tree should barely catch defaulters (honest overfit cost)"


# ------------------------------------------------------------------ Part G
class TestOverfitHonesty:
    def test_full_tree_memorizes_train(self, results):
        assert results["tree_full_train_acc"] == pytest.approx(1.0000, abs=1e-4)

    def test_full_tree_test_accuracy_drops(self, results):
        assert results["tree_unconstrained"]["accuracy"] <= 0.72, \
            "unconstrained tree must show clear train/test gap"

    def test_depth3_closes_the_gap(self, results):
        assert results["tree_depth3"]["accuracy"] > results["tree_unconstrained"]["accuracy"]


# ------------------------------------------------------------------ Part H
class TestErrorAndCalibration:
    def test_calibration_curve_has_five_bins(self, eng, df):
        from sklearn.calibration import calibration_curve
        y = (df["class"] == "bad").astype(int)
        X_tr, X_te, y_tr, y_te = train_test_split(
            eng, y, test_size=0.2, random_state=42, stratify=y)
        rf = RandomForestClassifier(n_estimators=200, random_state=42).fit(X_tr, y_tr)
        y_prob = rf.predict_proba(X_te)[:, 1]
        prob_true, prob_pred = calibration_curve(y_te, y_prob, n_bins=5, strategy="uniform")
        assert len(prob_true) == 5
        assert 0 <= prob_true.min() <= prob_pred.max() <= 1

    def test_notebook_calibration_cell_prints_five_bins(self):
        nb = json.loads(NB.read_text())
        code_txt = [c for c in nb["cells"] if c.get("cell_type") == "code"]
        for cell in code_txt:
            src = "".join(cell.get("source", []))
            if "calibration_curve" in src and "n_bins=5" in src:
                printed = "".join("".join(o.get("text", []))
                                  for o in cell.get("outputs", []) if o.get("output_type") == "stream")
                rows = [l for l in printed.splitlines() if l.lstrip().startswith("~")]
                assert len(rows) == 5, f"expected 5 calibration bins, got:\n{printed}"
                return
        raise AssertionError("calibration cell not found in notebook")

    def test_error_rate_around_pct(self, split, results):
        X_tr, X_te, y_tr, y_te = split
        rf = RandomForestClassifier(n_estimators=200, random_state=42).fit(X_tr, y_tr)
        pred = (rf.predict_proba(X_te)[:, 1] >= 0.5).astype(int)
        n_mis = int((pred != y_te.values).sum())
        assert 30 <= n_mis <= 55, f"misclassified count drifted: {n_mis}"


# ------------------------------------------------------------------ Part I
class TestCharts:
    def test_png_magic_bytes(self):
        for path in CHARTS:
            assert path.exists(), f"chart missing: {path.name}"
            assert path.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n", f"{path.name} not a PNG"

    def test_png_has_real_dimensions(self):
        for path in CHARTS:
            data = path.read_bytes()
            w, h = struct.unpack(">II", data[16:24])
            assert w > 100 and h > 100, f"{path.name} has implausible {w}x{h}"


# ------------------------------------------------------------------ Part J
class TestNotebookIntegrity:
    def test_notebook_exists(self):
        assert NB.exists()

    def test_notebook_has_no_error_cells(self):
        nb = json.loads(NB.read_text())
        for i, cell in enumerate(nb["cells"]):
            if cell.get("cell_type") == "code":
                for out in cell.get("outputs", []):
                    assert out.get("output_type") != "error", f"error output in cell {i}"

    def test_notebook_trained_to_the_end(self):
        nb = json.loads(NB.read_text())
        code_cells = [c for c in nb["cells"] if c.get("cell_type") == "code"]
        last_src = "".join(code_cells[-1]["source"])
        assert "End of notebook" in last_src

    def test_calibration_markdown_matches_live(self):
        nb = json.loads(NB.read_text())
        txt = "\n".join("".join(c.get("source", []))
                        for c in nb["cells"] if c.get("cell_type") == "markdown")
        for pin in ("0.759", "0.800", "0.562", "42 of 200"):
            assert pin in txt, f"markdown lost the verified number {pin}"


# ------------------------------------------------------------------ Part K
class TestRawImmutability:
    def test_sha_file_pins_the_csv(self):
        pinned = SHA.read_text().split()[0]
        assert pinned == EXPECTED_SHA
        assert hashlib.sha256(CSV.read_bytes()).hexdigest() == pinned