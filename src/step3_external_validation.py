"""Step 3. Final models, external validation on Dataset B, comparisons.

Sections 3.12 and 3.13.

Each algorithm is fitted on all of Dataset A for each feature set, using the
hyperparameters selected most often in cross-validation. The decision
threshold comes from the Dataset A out-of-fold predictions at the 80 per cent
sensitivity operating point, and is fixed before Dataset B is scored. Dataset
B is processed with Dataset A's medians and carry limits, unchanged.

Confidence intervals come from a patient-level cluster bootstrap in which all
models and SIRS are evaluated on the same resampled patients, so every
difference reported is a paired difference.
"""
import json
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

import algorithms as alg
import config
import evaluate as ev
import prep
from common import log, took

R, M = config.RESULTS, config.MODELS


def modal_hyperparameters(t):
    """The configuration selected most often across the five folds."""
    h = pd.read_csv(R / f"hyperparameters_{t}.csv")
    keys = [c for c in h.columns if c not in ("fold", "tuning_auprc", "threshold")]
    best = h.groupby(keys).size().sort_values(ascending=False).index[0]
    best = best if isinstance(best, tuple) else (best,)
    whole = ("max_depth", "min_samples_leaf")
    return {k: int(v) if k in whole else float(v) for k, v in zip(keys, best)}


def sirs_scores(df):
    cols = prep.feature_columns(df)
    score, evaluable = prep.sirs(prep.forward_fill(df, cols))
    return score, evaluable


def first_hour_mask(df):
    m = np.zeros(len(df), bool)
    m[df.groupby("patient_id", sort=False).head(1).index.to_numpy()] = True
    return m


def main():
    log("STEP 3  final models and external validation on Dataset B")
    t_all = time.time()
    a, b, _ = prep.load_datasets()
    ya, yb = a.SepsisLabel.to_numpy(), b.SepsisLabel.to_numpy()
    ga, gb = pd.factorize(a.patient_id)[0], pd.factorize(b.patient_id)[0]
    first_a = first_hour_mask(a)
    never_a = a.groupby("patient_id").SepsisLabel.transform("max").to_numpy() == 0

    preds = {"A": {}, "B": {}}
    artefacts, perf, calib, first_rows = {}, [], [], []

    for fs in config.FEATURE_SETS:
        p = prep.Preprocessor(fs)
        Xa_df = p.fit_transform(a)
        Xa, Xb = Xa_df.to_numpy(), p.transform(b).to_numpy()
        for algo in config.ALGORITHMS:
            t = config.tag(algo, fs)
            t0 = time.time()
            oof = np.load(R / f"oof_{t}.npy")
            params = modal_hyperparameters(t)
            model_file = M / f"{t}.joblib"
            if model_file.exists() and (R / f"pred_b_{t}.npy").exists():
                model = joblib.load(model_file)
                pb = np.load(R / f"pred_b_{t}.npy")
                note = "loaded"
            else:
                model = alg.fit_model(algo, params, Xa, ya, "final")
                pb = alg.predict(model, Xb)
                joblib.dump(model, model_file, compress=3)
                np.save(R / f"pred_b_{t}.npy", pb)
                note = "fitted"

            # Threshold fixed on Dataset A before Dataset B is evaluated.
            thr = ev.threshold_at_sensitivity(ya, oof, config.TARGET_SENS)

            # Recalibration fitted on Dataset A only (Section 3.13).
            rc = LogisticRegression(penalty=None, solver="lbfgs", max_iter=1000)
            rc.fit(ev.logit(oof).reshape(-1, 1), ya)
            joblib.dump(rc, M / f"recalibrator_{t}.joblib")

            preds["A"][t], preds["B"][t] = oof, pb
            for ds, y, pr in [("A", ya, oof), ("B", yb, pb)]:
                at = ev.at_threshold(y, pr, thr)
                perf.append({"model": t, "algorithm": algo, "feature_set": fs,
                             "dataset": ds, **ev.core_metrics(y, pr), **at,
                             "false_alerts_per_true_alert":
                                 at["fp"] / max(at["tp"], 1)})
                pc = rc.predict_proba(ev.logit(pr).reshape(-1, 1))[:, 1]
                np.save(R / f"pred_calibrated_{t}_{ds}.npy", pc)
                cm = ev.core_metrics(y, pc)
                calib.append({"model": t, "dataset": ds,
                              "mean_score_as_fitted": float(pr.mean()),
                              "mean_risk_recalibrated": float(pc.mean()),
                              "observed_rate": float(y.mean()),
                              "brier_as_fitted": float(np.mean((pr - y) ** 2)),
                              "brier_recalibrated": cm["brier"],
                              "slope_recalibrated": cm["calibration_slope"],
                              "intercept_recalibrated": cm["calibration_intercept"]})
                ev.calibration_curve_points(y, pc).to_csv(
                    R / f"calibration_{t}_{ds}.csv", index=False)
                ev.decision_curve(y, pc).to_csv(R / f"dca_{t}_{ds}.csv", index=False)

            first_rows.append({
                "model": t, "threshold": thr,
                "alert_rate_first_recorded_hour": float((oof[first_a] >= thr).mean()),
                "alert_rate_later_hours": float((oof[~first_a] >= thr).mean()),
                "alert_rate_never_septic_patients": float((oof[never_a] >= thr).mean()),
                "highest_score": float(oof.max()),
                "highest_score_is_sepsis": bool(ya[int(np.argmax(oof))] == 1)})

            artefacts[t] = {
                "algorithm": algo, "feature_set": fs, "hyperparameters": params,
                "columns": list(Xa_df.columns), "medians": p.medians_,
                "dropped_no_data": p.dropped_,
                "ffill_limits": prep.ffill_limits(p.base_cols_),
                "decision_threshold": thr,
                "target_sensitivity": config.TARGET_SENS,
                "random_state": config.SEED}
            pa, pbm = perf[-2], perf[-1]
            log(f"  {t} ({note}): AUROC A {pa['auroc']:.3f}  B {pbm['auroc']:.3f}"
                f"   alert rate A {pa['alert_rate']:.2f}  B {pbm['alert_rate']:.2f}"
                f"   [{took(t0)}]")

    pd.DataFrame(perf).to_csv(R / "performance.csv", index=False)
    pd.DataFrame(calib).to_csv(R / "calibration_summary.csv", index=False)
    pd.DataFrame(first_rows).to_csv(R / "alert_behaviour.csv", index=False)
    with open(M / "artefacts.json", "w") as f:
        json.dump(artefacts, f, indent=2, default=float)

    # ------------------------------------------------------------- SIRS
    log("  SIRS comparator")
    rows = []
    for ds, df, y in [("A", a, ya), ("B", b, yb)]:
        score, evaluable = sirs_scores(df)
        preds[ds]["sirs"] = score / 4.0
        np.save(R / f"sirs_{ds}.npy", score / 4.0)
        at = ev.at_threshold(y, score, 2)
        rows.append({"dataset": ds, "coverage_all_four": float(evaluable.mean()),
                     **{k: v for k, v in ev.core_metrics(y, score / 4.0).items()
                        if k in ("auroc", "auprc")},
                     **{k: at[k] for k in ("sensitivity", "specificity", "ppv",
                                           "npv", "alert_rate")}})
    pd.DataFrame(rows).to_csv(R / "sirs.csv", index=False)

    # -------------------------------------------------------- bootstrap
    log(f"  patient-level bootstrap, {config.N_BOOT} resamples per dataset "
        f"(the slowest part of this step)")
    ci_rows, cmp_rows, sirs_rows = [], [], []
    for ds, y, g in [("A", ya, ga), ("B", yb, gb)]:
        t0 = time.time()
        bs = ev.cluster_bootstrap(y, preds[ds], g, n_boot=config.N_BOOT,
                                  seed=config.BOOT_SEED)
        np.savez_compressed(R / f"bootstrap_raw_{ds}.npz", **bs)
        for name in preds[ds]:
            for metric in ("auroc", "auprc"):
                arr = bs[f"{name}_{metric}"]
                lo, hi = ev.ci(arr)
                ci_rows.append({"dataset": ds, "model": name, "metric": metric,
                                "mean": float(np.nanmean(arr)),
                                "ci_low": lo, "ci_high": hi})
        for algo in config.ALGORITHMS:
            r, f = config.tag(algo, "restricted"), config.tag(algo, "full")
            row = {"dataset": ds, "algorithm": algo}
            for metric in ("auroc", "auprc"):
                d = bs[f"{r}_{metric}"] - bs[f"{f}_{metric}"]
                lo, hi = ev.ci(d)
                row.update({f"delta_{metric}": float(np.nanmean(d)),
                            f"delta_{metric}_ci_low": lo,
                            f"delta_{metric}_ci_high": hi})
            row["margin"] = config.MARGIN
            row["criterion_met"] = bool(row["delta_auroc_ci_low"] > config.MARGIN)
            cmp_rows.append(row)
        for name in preds[ds]:
            if name == "sirs":
                continue
            d = bs[f"{name}_auroc"] - bs["sirs_auroc"]
            lo, hi = ev.ci(d)
            sirs_rows.append({"dataset": ds, "model": name,
                              "delta_auroc_vs_sirs": float(np.nanmean(d)),
                              "ci_low": lo, "ci_high": hi,
                              "interval_excludes_zero": bool(lo > 0 or hi < 0)})
        log(f"    Dataset {ds} done   [{took(t0)}]")
    pd.DataFrame(ci_rows).to_csv(R / "confidence_intervals.csv", index=False)
    cmp = pd.DataFrame(cmp_rows)
    cmp.to_csv(R / "restricted_vs_full.csv", index=False)
    pd.DataFrame(sirs_rows).to_csv(R / "model_vs_sirs.csv", index=False)

    log("  restricted minus full, AUROC (pre-registered margin "
        f"{config.MARGIN}):")
    for r in cmp.itertuples():
        log(f"    Dataset {r.dataset}  {config.ALGO_LABEL[r.algorithm]:20s} "
            f"{r.delta_auroc:+.4f} [{r.delta_auroc_ci_low:+.4f}, "
            f"{r.delta_auroc_ci_high:+.4f}]  "
            f"{'criterion met' if r.criterion_met else 'criterion NOT met'}")
    log(f"STEP 3 finished in {took(t_all)}")


if __name__ == "__main__":
    main()
