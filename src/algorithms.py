"""The four learning algorithms (Section 3.10) and how each is explained.

Every algorithm here is an existing, published implementation taken from a
standard library. Nothing in this file implements a learning algorithm; it
only says which settings each one is given and which settings are tuned.

  Logistic Regression   scikit-learn   (Pedregosa et al., 2011)
  Decision Tree         scikit-learn   (Pedregosa et al., 2011)
  Random Forest         scikit-learn   (Breiman, 2001; Pedregosa et al., 2011)
  XGBoost               xgboost        (Chen & Guestrin, 2016)

Class imbalance is handled the same way for all four, by weighting the
classes inversely to their frequency in the training data. No synthetic
patients are created and no patient-hours are discarded.
"""
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier, DMatrix

import config

# Hyperparameter grids. Every combination is tried inside each training fold
# and the one with the best AUPRC on a held-out tuning split is kept.
GRIDS = {
    "logistic_regression": [{"C": c} for c in (0.01, 0.1, 1.0)],
    "decision_tree": [{"max_depth": d, "min_samples_leaf": l}
                      for d in (4, 6, 8, 10) for l in (50, 200)],
    "random_forest": [{"max_depth": d, "min_samples_leaf": l}
                      for d in (10, 16, 24) for l in (10, 50)],
    "xgboost": [{"max_depth": d, "learning_rate": lr}
                for d in (4, 6) for lr in (0.05, 0.1)],
}

# Number of trees. Fewer trees are used for the tuning fits than for the
# model that is actually evaluated, to keep the tuning affordable.
TREES = {"tune": 20, "final": 40} if config.QUICK else {"tune": 100, "final": 300}


def fit_model(algo, params, X, y, stage="final"):
    """Fit one algorithm with one set of hyperparameters."""
    seed, jobs = config.SEED, config.N_JOBS
    if algo == "logistic_regression":
        # Inputs are standardised for Logistic Regression only. The scaler is
        # part of the fitted model, so it is always estimated on training data.
        m = Pipeline([
            ("scale", StandardScaler()),
            ("clf", LogisticRegression(class_weight="balanced", max_iter=2000,
                                       solver="lbfgs", **params)),
        ])
    elif algo == "decision_tree":
        m = DecisionTreeClassifier(class_weight="balanced", random_state=seed,
                                   **params)
    elif algo == "random_forest":
        m = RandomForestClassifier(n_estimators=TREES[stage],
                                   max_features="sqrt",
                                   class_weight="balanced", n_jobs=jobs,
                                   random_state=seed, **params)
    elif algo == "xgboost":
        pos = float(y.sum())
        m = XGBClassifier(n_estimators=TREES[stage], tree_method="hist",
                          subsample=0.8, colsample_bytree=0.8,
                          scale_pos_weight=(len(y) - pos) / max(pos, 1.0),
                          eval_metric="logloss", n_jobs=jobs,
                          random_state=seed, **params)
    else:
        raise ValueError(f"unknown algorithm {algo}")
    m.fit(X, y)
    return m


def predict(model, X):
    """Score for the positive class, one value per patient-hour."""
    return model.predict_proba(X)[:, 1].astype(np.float64)


# ------------------------------------------------------------------ explain
EXPLANATION_SCALE = {
    "logistic_regression": "log-odds",
    "xgboost": "log-odds",
    "decision_tree": "probability",
    "random_forest": "probability",
}


def explain(model, algo, X):
    """Exact Shapley values for every row of X.

    Returns (values, baseline). values has one row per patient-hour and one
    column per feature; baseline plus the sum of a row equals the model's
    output for that row, on the scale given in EXPLANATION_SCALE.

    Logistic Regression: for a linear model the Shapley value of a feature is
    its coefficient times the distance of the standardised value from the
    training mean, which is zero after standardisation.
    XGBoost: TreeSHAP as implemented inside the xgboost library.
    Decision Tree and Random Forest: TreeSHAP from the shap library.
    All three are exact, not sampled approximations.
    """
    if algo == "logistic_regression":
        z = model.named_steps["scale"].transform(X)
        clf = model.named_steps["clf"]
        return z * clf.coef_[0], float(clf.intercept_[0])
    if algo == "xgboost":
        c = model.get_booster().predict(DMatrix(X), pred_contribs=True)
        return c[:, :-1].astype(np.float64), float(c[0, -1])
    import shap
    exp = shap.TreeExplainer(model)
    sv = np.asarray(exp.shap_values(X, check_additivity=False))
    if sv.ndim == 3:                       # (rows, features, classes)
        sv = sv[:, :, 1] if sv.shape[2] == 2 else sv[1]
    return sv, float(np.asarray(exp.expected_value).ravel()[-1])


def global_importance(values):
    """Mean absolute Shapley value per feature."""
    return np.abs(values).mean(axis=0)
