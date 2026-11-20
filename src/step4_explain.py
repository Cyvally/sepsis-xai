"""Step 4. Explanations (Section 3.11).

Global importance: mean absolute Shapley value of each feature, for every
final model, on a documented random sample of Dataset A patient-hours.

Explanation consistency: the Spearman rank correlation of global importance
between the five cross-validation fold models (computed in step 2). It asks
whether an explanation is a property of the data or of one particular fit.

Individual explanations: four patient-hours scored by one model, each fully
decomposed, so that a correct alert, two kinds of false alert and a correct
non-alert can be compared.
"""
import json
import time

import joblib
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

import algorithms as alg
import config
import prep
from common import log, took

R, M = config.RESULTS, config.MODELS

# The model whose individual explanations are shown in Chapter Four.
LOCAL_MODEL = ("random_forest", "restricted")


def consistency():
    rows = []
    for fs in config.FEATURE_SETS:
        for algo in config.ALGORITHMS:
            t = config.tag(algo, fs)
            f = R / f"shap_fold_importance_{t}.csv"
            if not f.exists():
                continue
            d = pd.read_csv(f, index_col=0)
            cors = [spearmanr(d.iloc[:, i], d.iloc[:, j]).statistic
                    for i in range(d.shape[1]) for j in range(i + 1, d.shape[1])]
            cors = np.array(cors, dtype=float)
            rows.append({"model": t, "pairs_of_folds": len(cors),
                         "spearman_median": float(np.nanmedian(cors)),
                         "spearman_min": float(np.nanmin(cors)),
                         "spearman_max": float(np.nanmax(cors))})
    pd.DataFrame(rows).to_csv(R / "shap_consistency.csv", index=False)
    return rows


def local_cases(model, algo, X, y, columns, thr, patient_ids):
    """Four illustrative patient-hours, scored and explained by one model."""
    p = alg.predict(model, X)
    alert = p >= thr
    picks = {}
    tp = np.flatnonzero(alert & (y == 1))
    fp = np.flatnonzero(alert & (y == 0))
    tn = np.flatnonzero(~alert & (y == 0))
    if len(tp):
        picks["confident_true_positive"] = int(tp[np.argmax(p[tp])])
    if len(fp):
        picks["confident_false_positive"] = int(fp[np.argmax(p[fp])])
        # The alerting negative whose score is closest to what the model
        # returns before any feature is considered.
        _, base = alg.explain(model, algo, X[:1])
        base_p = base if alg.EXPLANATION_SCALE[algo] == "probability" \
            else 1 / (1 + np.exp(-base))
        picks["default_false_positive"] = int(fp[np.argmin(np.abs(p[fp] - base_p))])
    if len(tn):
        picks["correct_rejection"] = int(tn[np.argmin(p[tn])])
    rows = []
    for case, i in picks.items():
        values, base = alg.explain(model, algo, X[[i]])
        for c, v, s in zip(columns, X[i], values[0]):
            rows.append({"case": case, "patient_id": patient_ids[i],
                         "label": int(y[i]), "score": float(p[i]),
                         "alerted": bool(alert[i]), "threshold": thr,
                         "baseline": base, "feature": c,
                         "feature_value": float(v), "shap_value": float(s)})
    return pd.DataFrame(rows)


def main():
    log("STEP 4  explanations")
    t_all = time.time()
    a, _, _ = prep.load_datasets()
    ya = a.SepsisLabel.to_numpy()
    art = json.load(open(M / "artefacts.json"))
    rng = np.random.default_rng(config.SEED)

    for fs in config.FEATURE_SETS:
        Xdf = prep.Preprocessor(fs).fit_transform(a)
        X, cols = Xdf.to_numpy(), list(Xdf.columns)
        samp = rng.choice(len(X), min(config.SHAP_GLOBAL_ROWS, len(X)),
                          replace=False)
        for algo in config.ALGORITHMS:
            t = config.tag(algo, fs)
            out = R / f"shap_global_{t}.csv"
            if out.exists():
                log(f"  {t}: already done, skipped")
                continue
            t0 = time.time()
            model = joblib.load(M / f"{t}.joblib")
            values, _ = alg.explain(model, algo, X[samp])
            gi = (pd.DataFrame({"feature": cols,
                                "mean_abs_shap": alg.global_importance(values)})
                  .sort_values("mean_abs_shap", ascending=False))
            gi["scale"] = alg.EXPLANATION_SCALE[algo]
            gi.to_csv(out, index=False)
            log(f"  {t}: top features {', '.join(gi.feature.head(4))}   [{took(t0)}]")

        t = config.tag(*LOCAL_MODEL)
        if fs == LOCAL_MODEL[1] and not (R / "shap_local_cases.csv").exists():
            local_cases(joblib.load(M / f"{t}.joblib"), LOCAL_MODEL[0], X, ya,
                        cols, art[t]["decision_threshold"],
                        a.patient_id.to_numpy()).to_csv(
                R / "shap_local_cases.csv", index=False)
            log(f"  wrote four individual explanations for {t}")

    for r in consistency():
        log(f"  consistency {r['model']}: Spearman median "
            f"{r['spearman_median']:.3f} (range {r['spearman_min']:.3f} to "
            f"{r['spearman_max']:.3f})")
    with open(R / "shap_provenance.json", "w") as f:
        json.dump({"global_sample_hours": config.SHAP_GLOBAL_ROWS,
                   "per_fold_sample_hours": config.SHAP_FOLD_ROWS,
                   "random_state": config.SEED,
                   "sampled_from": "Dataset A",
                   "scale": alg.EXPLANATION_SCALE,
                   "individual_explanations_model": config.tag(*LOCAL_MODEL)},
                  f, indent=2)
    log(f"STEP 4 finished in {took(t_all)}")


if __name__ == "__main__":
    main()
