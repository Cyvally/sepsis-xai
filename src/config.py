"""Settings shared by every step. Change things here, not in the step files."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
SET_A = DATA / "setA.parquet"
SET_B = DATA / "setB.parquet"

# Quick mode: a small sample of patients and small models, to check in a few
# minutes that everything is installed and working. Its output goes to a
# separate folder so it can never be mistaken for the real results.
QUICK = os.environ.get("SEPSIS_QUICK") == "1"
OUT = ROOT / "quick_test" if QUICK else ROOT
RESULTS = OUT / "results"
MODELS = OUT / "models"
FIGURES = OUT / "figures"
for d in (RESULTS, MODELS, FIGURES):
    d.mkdir(parents=True, exist_ok=True)

SEED = 42                 # data splits, model fitting, SHAP sampling
BOOT_SEED = 7             # bootstrap
N_FOLDS = 5
TARGET_SENS = 0.80        # operating point fixed in advance (Section 3.12)
MARGIN = -0.05            # pre-registered acceptability margin (Section 3.12)
N_JOBS = -1               # use every processor core

N_BOOT = 100 if QUICK else 2000
TUNE_ROWS = 250_000       # rows used for the tuning fits only
QUICK_PATIENTS = 1500     # patients per dataset in quick mode

ALGORITHMS = ["logistic_regression", "decision_tree", "xgboost", "random_forest"]
FEATURE_SETS = ["restricted", "full"]
ALGO_LABEL = {"logistic_regression": "Logistic Regression",
              "decision_tree": "Decision Tree",
              "random_forest": "Random Forest",
              "xgboost": "XGBoost"}
SET_LABEL = {"restricted": "restricted", "full": "full"}

# Patient-hours sampled for the SHAP calculations. Every row gets exact
# values; the sample size only affects the precision of the averages.
SHAP_GLOBAL_ROWS = 500 if QUICK else 5000
SHAP_FOLD_ROWS = 300 if QUICK else 2000


def tag(algo, feature_set):
    return f"{algo}_{feature_set}"


def label(t):
    for a in ALGORITHMS:
        if t.startswith(a):
            return f"{ALGO_LABEL[a]}, {t[len(a) + 1:].replace('_', ' ')} set"
    return t
