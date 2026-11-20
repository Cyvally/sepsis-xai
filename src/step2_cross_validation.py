"""Step 2. Internal validation on Dataset A (Sections 3.10 and 3.12).

Five-fold cross-validation grouped by patient, for every algorithm on both
feature sets. Inside each training fold a further patient-grouped split is
held back: hyperparameters are chosen on it, and so is the decision threshold,
because a threshold taken from predictions on the model's own training data is
biased upward. Nothing from the held-out fold enters tuning.

Two sensitivity analyses are run on the restricted feature set:
  median_only      median imputation without forward filling
  no_indicators    no availability indicators

A finished run is saved, so if the computer sleeps or the script is stopped,
running it again continues from where it stopped.
"""
import time

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score
from sklearn.model_selection import GroupKFold, GroupShuffleSplit

import algorithms as alg
import config
import evaluate as ev
import prep
from common import log, took

R = config.RESULTS

# (name used in file names, feature set, median only, availability indicators)
CONFIGS = [
    ("restricted", "restricted", False, True),
    ("full", "full", False, True),
    ("restricted_median_only", "restricted", True, True),
    ("restricted_no_indicators", "restricted", False, False),
]


def tune(algo, Xtr, ytr, gtr, rng):
    """Choose hyperparameters and the decision threshold on a held-out split."""
    gss = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=config.SEED)
    inner_tr, inner_va = next(gss.split(Xtr, ytr, gtr))
    if len(inner_tr) > config.TUNE_ROWS:
        inner_tr = rng.choice(inner_tr, config.TUNE_ROWS, replace=False)
    best, best_score, best_p = None, -np.inf, None
    for params in alg.GRIDS[algo]:
        m = alg.fit_model(algo, params, Xtr[inner_tr], ytr[inner_tr], "tune")
        p_va = alg.predict(m, Xtr[inner_va])
        s = average_precision_score(ytr[inner_va], p_va)
        if s > best_score:
            best, best_score, best_p = params, s, p_va
    thr = ev.threshold_at_sensitivity(ytr[inner_va], best_p, config.TARGET_SENS)
    return best, best_score, thr


def run(algo, name, X, y, groups, columns, with_shap):
    t = config.tag(algo, name)
    done = (R / f"cv_{t}.csv").exists() and (R / f"oof_{t}.npy").exists()
    if done:
        log(f"  {t}: already done, skipped")
        return
    log(f"  {t}")
    rng = np.random.default_rng(config.SEED)
    oof = np.zeros(len(y))
    rows, chosen, importance = [], [], {}
    folds = GroupKFold(n_splits=config.N_FOLDS).split(X, y, groups)
    for k, (tr, te) in enumerate(folds, 1):
        t0 = time.time()
        params, tscore, thr = tune(algo, X[tr], y[tr], groups[tr], rng)
        m = alg.fit_model(algo, params, X[tr], y[tr], "final")
        p_te = alg.predict(m, X[te])
        oof[te] = p_te
        r = {"fold": k, **ev.core_metrics(y[te], p_te),
             **ev.at_threshold(y[te], p_te, thr),
             "n_test_rows": len(te),
             "n_test_patients": len(np.unique(groups[te]))}
        rows.append(r)
        chosen.append({"fold": k, **params, "tuning_auprc": tscore,
                       "threshold": thr})
        if with_shap:
            # Explanation consistency (Section 3.11): global importance in
            # this fold's model, on a sample of this fold's held-out hours.
            srng = np.random.default_rng(config.SEED + k)
            samp = srng.choice(te, min(config.SHAP_FOLD_ROWS, len(te)),
                               replace=False)
            values, _ = alg.explain(m, algo, X[samp])
            importance[f"fold{k}"] = alg.global_importance(values)
        log(f"    fold {k}/{config.N_FOLDS}: AUROC {r['auroc']:.3f}  "
            f"AUPRC {r['auprc']:.3f}  sensitivity {r['sensitivity']:.2f}  "
            f"alert rate {r['alert_rate']:.2f}   [{took(t0)}]")

    df = pd.DataFrame(rows)
    body = df.drop(columns=["fold"])
    mean, sd = body.mean().to_dict(), body.std().to_dict()
    mean["fold"], sd["fold"] = "mean", "sd"
    pd.concat([df, pd.DataFrame([mean, sd])], ignore_index=True).to_csv(
        R / f"cv_{t}.csv", index=False)
    pd.DataFrame(chosen).to_csv(R / f"hyperparameters_{t}.csv", index=False)
    if with_shap:
        pd.DataFrame(importance, index=columns).to_csv(
            R / f"shap_fold_importance_{t}.csv")
    np.save(R / f"oof_{t}.npy", oof)      # saved last: marks the run complete
    log(f"    mean AUROC {df.auroc.mean():.4f}   mean AUPRC {df.auprc.mean():.4f}")


def main():
    log("STEP 2  cross-validation on Dataset A")
    t0 = time.time()
    a, _, _ = prep.load_datasets()
    y = a.SepsisLabel.to_numpy()
    groups = pd.factorize(a.patient_id)[0]
    for name, fs, median_only, indicators in CONFIGS:
        p = prep.Preprocessor(fs, use_indicators=indicators,
                              median_only=median_only)
        Xdf = p.fit_transform(a)
        X = Xdf.to_numpy()
        log(f"feature configuration '{name}': {X.shape[1]} columns"
            + (f", dropped for having no data: {p.dropped_}" if p.dropped_ else ""))
        for algo in config.ALGORITHMS:
            run(algo, name, X, y, groups, list(Xdf.columns),
                with_shap=name in config.FEATURE_SETS)
    np.save(R / "y_A.npy", y)
    np.save(R / "groups_A.npy", groups)
    log(f"STEP 2 finished in {took(t0)}")


if __name__ == "__main__":
    main()
