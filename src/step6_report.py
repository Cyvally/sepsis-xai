"""Step 6. Collect every number into one document for writing Chapter Four.

Only reads the files the earlier steps produced. Nothing is recomputed, so
this document cannot disagree with the results files.
"""
import json

import numpy as np
import pandas as pd

import config
from common import log

R, M = config.RESULTS, config.MODELS
ALGOS = ["logistic_regression", "decision_tree", "random_forest", "xgboost"]
NAME = config.ALGO_LABEL
L = []


def w(s=""):
    L.append(s)


def md_table(header, rows):
    w("| " + " | ".join(header) + " |")
    w("|" + "---|" * len(header))
    for r in rows:
        w("| " + " | ".join(str(x) for x in r) + " |")
    w()


def model_name(t):
    if t == "sirs":
        return "SIRS"
    a = next(x for x in ALGOS if t.startswith(x))
    return f"{NAME[a]}, {t[len(a) + 1:].replace('_', ' ')}"


def cohort():
    c = pd.read_csv(R / "cohort.csv")
    w("## 1. Cohort (Section 4.1)")
    w()
    spec = [("Patients", "patients", "{:,.0f}"),
            ("Patient-hours", "patient_hours", "{:,.0f}"),
            ("Patients who developed sepsis", "septic_patients", "{:,.0f}"),
            ("Patients who developed sepsis (%)", "septic_patients_pct", "{:.2f}"),
            ("Positive patient-hours", "positive_hours", "{:,.0f}"),
            ("Positive patient-hours (%)", "positive_hours_pct", "{:.2f}"),
            ("Class imbalance, 1 positive to n negative", "imbalance_1_to", "{:.1f}"),
            ("Age, median", "age_median", "{:.1f}"),
            ("Age, interquartile range low", "age_iqr_low", "{:.1f}"),
            ("Age, interquartile range high", "age_iqr_high", "{:.1f}"),
            ("Male (%)", "male_pct", "{:.1f}"),
            ("Record length in hours, median", "los_hours_median", "{:.0f}"),
            ("Hours of record before first positive hour, median",
             "hours_before_first_positive_median", "{:.1f}"),
            ("Patients under 18 excluded", "under_18_excluded", "{:.0f}")]
    md_table(["", "Dataset A", "Dataset B"],
             [(lab, f.format(c.loc[0, col]), f.format(c.loc[1, col]))
              for lab, col, f in spec])


def missingness():
    m = pd.read_csv(R / "missingness.csv")
    w("## 2. Missingness (Section 4.2)")
    w()
    never = m[m.n_observed_A == 0].variable.tolist()
    w(f"Eligible predictor variables: {len(m)}. "
      + (f"Never recorded in Dataset A and therefore dropped: {', '.join(never)}. "
         f"Variables in the full feature set: {len(m) - len(never)}."
         if never else "All were recorded at least once in Dataset A."))
    w()
    w("Restricted feature set:")
    w()
    md_table(["Variable", "Carry limit (h)", "Missing A (%)", "After fill A (%)",
              "Missing B (%)", "After fill B (%)"],
             [(r.variable,
               "static" if pd.isna(r.ffill_limit_h) else f"{r.ffill_limit_h:.0f}",
               f"{r.pct_missing_A:.1f}", f"{r.pct_missing_A_after_ffill:.1f}",
               f"{r.pct_missing_B:.1f}", f"{r.pct_missing_B_after_ffill:.1f}")
              for r in m[m.in_restricted_set].itertuples()])
    w("All other variables:")
    w()
    md_table(["Variable", "Missing A (%)", "After fill A (%)", "Missing B (%)",
              "After fill B (%)"],
             [(r.variable, f"{r.pct_missing_A:.1f}",
               f"{r.pct_missing_A_after_ffill:.1f}", f"{r.pct_missing_B:.1f}",
               f"{r.pct_missing_B_after_ffill:.1f}")
              for r in m[~m.in_restricted_set].itertuples()])
    d = pd.read_csv(R / "descriptives_restricted.csv")
    w("Restricted predictors, median (interquartile range) of recorded values:")
    w()
    md_table(["Variable", "Dataset A", "Dataset B"],
             [(r.variable,
               f"{r.A_median:.1f} ({r.A_iqr_low:.1f} to {r.A_iqr_high:.1f})",
               f"{r.B_median:.1f} ({r.B_iqr_low:.1f} to {r.B_iqr_high:.1f})")
              for r in d.itertuples()])


def internal():
    w("## 3. Internal validation: five-fold cross-validation on Dataset A "
      "(Section 4.3)")
    w()
    w("Mean across the five folds, with the standard deviation in brackets.")
    w()
    rows = []
    for fs in config.FEATURE_SETS:
        for a in ALGOS:
            d = pd.read_csv(R / f"cv_{config.tag(a, fs)}.csv").set_index("fold")
            mu, sd = d.loc["mean"], d.loc["sd"]
            rows.append((NAME[a], fs,
                         f"{mu.auroc:.3f} ({sd.auroc:.3f})",
                         f"{mu.auprc:.3f} ({sd.auprc:.3f})",
                         f"{mu.sensitivity:.3f}", f"{mu.specificity:.3f}",
                         f"{mu.alert_rate:.3f}"))
    md_table(["Algorithm", "Feature set", "AUROC", "AUPRC", "Sensitivity",
              "Specificity", "Alert rate"], rows)
    w("Hyperparameters selected in each fold, and those used for the final model "
      "(the combination selected most often):")
    w()
    art = json.load(open(M / "artefacts.json"))
    rows = []
    for fs in config.FEATURE_SETS:
        for a in ALGOS:
            t = config.tag(a, fs)
            h = pd.read_csv(R / f"hyperparameters_{t}.csv")
            keys = [c for c in h.columns if c not in ("fold", "tuning_auprc", "threshold")]
            per_fold = "; ".join(", ".join(f"{k}={r[k]:g}" for k in keys)
                                 for _, r in h.iterrows())
            final = ", ".join(f"{k}={v:g}" for k, v in art[t]["hyperparameters"].items())
            rows.append((NAME[a], fs, per_fold, final))
    md_table(["Algorithm", "Feature set", "Folds 1 to 5", "Final model"], rows)


def performance():
    p = pd.read_csv(R / "performance.csv")
    ci = pd.read_csv(R / "confidence_intervals.csv").set_index(["dataset", "model", "metric"])
    s = pd.read_csv(R / "sirs.csv").set_index("dataset")
    w("## 4. Performance on Dataset A (out-of-fold) and Dataset B (external) "
      "(Sections 4.4 and 4.5)")
    w()
    w(f"Decision thresholds were fixed on Dataset A at {100*config.TARGET_SENS:.0f}% "
      f"sensitivity before Dataset B was scored. Confidence intervals: "
      f"{config.N_BOOT:,} patient-level bootstrap resamples.")
    w()
    for ds in ("A", "B"):
        w(f"### Dataset {ds}")
        w()
        rows = []
        for fs in config.FEATURE_SETS:
            for a in ALGOS:
                t = config.tag(a, fs)
                r = p[(p.model == t) & (p.dataset == ds)].iloc[0]
                au, ap = ci.loc[(ds, t, "auroc")], ci.loc[(ds, t, "auprc")]
                rows.append((NAME[a], fs,
                             f"{r.auroc:.3f} [{au.ci_low:.3f}, {au.ci_high:.3f}]",
                             f"{r.auprc:.3f} [{ap.ci_low:.3f}, {ap.ci_high:.3f}]",
                             f"{r.sensitivity:.3f}", f"{r.specificity:.3f}",
                             f"{100*r.ppv:.1f}%", f"{100*r.alert_rate:.1f}%",
                             f"{r.false_alerts_per_true_alert:.0f}"))
        au, ap = ci.loc[(ds, "sirs", "auroc")], ci.loc[(ds, "sirs", "auprc")]
        r = s.loc[ds]
        rows.append(("SIRS (score of 2 or more)", "-",
                     f"{r.auroc:.3f} [{au.ci_low:.3f}, {au.ci_high:.3f}]",
                     f"{r.auprc:.3f} [{ap.ci_low:.3f}, {ap.ci_high:.3f}]",
                     f"{r.sensitivity:.3f}", f"{r.specificity:.3f}",
                     f"{100*r.ppv:.1f}%", f"{100*r.alert_rate:.1f}%",
                     f"{(1 - r.ppv) / r.ppv:.0f}"))
        md_table(["Model", "Feature set", "AUROC [95% CI]", "AUPRC [95% CI]",
                  "Sensitivity", "Specificity", "PPV", "Alert rate",
                  "False alerts per true alert"], rows)
    w(f"SIRS could be scored on all four of its components on "
      f"{100*s.loc['A','coverage_all_four']:.1f}% of hours in Dataset A and "
      f"{100*s.loc['B','coverage_all_four']:.1f}% in Dataset B. A component "
      f"with no available value was counted as not met.")
    w()
    w("Change from Dataset A to Dataset B:")
    w()
    rows = []
    for fs in config.FEATURE_SETS:
        for a in ALGOS:
            t = config.tag(a, fs)
            ra = p[(p.model == t) & (p.dataset == "A")].iloc[0]
            rb = p[(p.model == t) & (p.dataset == "B")].iloc[0]
            rows.append((NAME[a], fs, f"{ra.auroc:.3f}", f"{rb.auroc:.3f}",
                         f"{rb.auroc - ra.auroc:+.3f}",
                         f"{ra.calibration_slope:.2f}",
                         f"{rb.calibration_slope:.2f}"))
    md_table(["Algorithm", "Feature set", "AUROC A", "AUROC B", "Change",
              "Calibration slope A", "Calibration slope B"], rows)


def comparisons():
    c = pd.read_csv(R / "restricted_vs_full.csv")
    w("## 5. Restricted against full feature set: the pre-registered criterion "
      "(Section 4.6)")
    w()
    w(f"Criterion fixed in Section 3.12: the restriction is acceptable if the "
      f"lower bound of the 95% confidence interval for the AUROC difference "
      f"(restricted minus full) is above {config.MARGIN}.")
    w()
    md_table(["Dataset", "Algorithm", "AUROC difference [95% CI]",
              "Lower bound", "Criterion", "AUPRC difference [95% CI]"],
             [(r.dataset, NAME[r.algorithm],
               f"{r.delta_auroc:+.4f} [{r.delta_auroc_ci_low:+.4f}, "
               f"{r.delta_auroc_ci_high:+.4f}]",
               f"{r.delta_auroc_ci_low:+.4f}",
               "met" if r.criterion_met else "**not met**",
               f"{r.delta_auprc:+.4f} [{r.delta_auprc_ci_low:+.4f}, "
               f"{r.delta_auprc_ci_high:+.4f}]") for r in c.itertuples()])
    n_met = int(c.criterion_met.sum())
    w(f"The criterion was met in {n_met} of {len(c)} algorithm and dataset "
      f"combinations.")
    w()
    s = pd.read_csv(R / "model_vs_sirs.csv")
    w("## 6. Every model against SIRS (Section 4.7)")
    w()
    md_table(["Dataset", "Model", "AUROC difference against SIRS [95% CI]",
              "Interval excludes zero"],
             [(r.dataset, model_name(r.model),
               f"{r.delta_auroc_vs_sirs:+.4f} [{r.ci_low:+.4f}, {r.ci_high:+.4f}]",
               "yes" if r.interval_excludes_zero else "**no**")
              for r in s.itertuples()])


def calibration():
    c = pd.read_csv(R / "calibration_summary.csv")
    w("## 7. Calibration and clinical utility (Section 4.8)")
    w()
    w("Class weighting inflates the raw scores, so they are not risks. The "
      "recalibration is a logistic model fitted on Dataset A only and applied "
      "unchanged to Dataset B. It does not change AUROC or AUPRC.")
    w()
    md_table(["Model", "Dataset", "Mean score as fitted", "Mean recalibrated risk",
              "Observed rate", "Brier as fitted", "Brier recalibrated",
              "Slope after recalibration"],
             [(model_name(r.model), r.dataset, f"{r.mean_score_as_fitted:.4f}",
               f"{r.mean_risk_recalibrated:.4f}", f"{r.observed_rate:.4f}",
               f"{r.brier_as_fitted:.4f}", f"{r.brier_recalibrated:.4f}",
               f"{r.slope_recalibrated:.2f}") for r in c.itertuples()])
    w("Net benefit on recalibrated risks (treat none = 0). A model is useful at "
      "a threshold only if its net benefit is above both treat all and zero.")
    w()
    for ds in ("A", "B"):
        rows = []
        for fs in config.FEATURE_SETS:
            for a in ALGOS:
                d = pd.read_csv(R / f"dca_{config.tag(a, fs)}_{ds}.csv")
                cells = []
                for th in (0.01, 0.02, 0.05, 0.10):
                    r = d.iloc[(d.threshold - th).abs().argsort().iloc[0]]
                    cells.append(f"{r.net_benefit_model:.5f}")
                rows.append((NAME[a], fs, *cells))
        d = pd.read_csv(R / f"dca_{config.tag(ALGOS[0], 'restricted')}_{ds}.csv")
        rows.append(("Treat all", "-", *[
            f"{d.iloc[(d.threshold - th).abs().argsort().iloc[0]].net_benefit_treat_all:.5f}"
            for th in (0.01, 0.02, 0.05, 0.10)]))
        w(f"Dataset {ds}:")
        w()
        md_table(["Model", "Feature set", "at 1%", "at 2%", "at 5%", "at 10%"], rows)


def alerts():
    a = pd.read_csv(R / "alert_behaviour.csv")
    w("## 8. Alert behaviour on Dataset A (Section 4.9)")
    w()
    w("The first recorded hour is the hour at which the model knows least "
      "about a patient. 'Never septic' means hours belonging to patients who "
      "did not meet the sepsis definition at any point.")
    w()
    md_table(["Model", "Threshold", "Alert rate, first recorded hour",
              "Alert rate, later hours", "Alert rate, never-septic patients",
              "Highest score given", "That hour was sepsis"],
             [(model_name(r.model), f"{r.threshold:.4f}",
               f"{100*r.alert_rate_first_recorded_hour:.1f}%",
               f"{100*r.alert_rate_later_hours:.1f}%",
               f"{100*r.alert_rate_never_septic_patients:.1f}%",
               f"{r.highest_score:.4f}",
               "yes" if r.highest_score_is_sepsis else "no")
              for r in a.itertuples()])


def sensitivity():
    w("## 9. Sensitivity analyses on the restricted feature set (Section 4.10)")
    w()
    w("Each row removes one preprocessing decision and keeps everything else "
      "the same. Mean cross-validated AUROC on Dataset A.")
    w()
    rows = []
    for a in ALGOS:
        def auroc(name):
            return pd.read_csv(R / f"cv_{config.tag(a, name)}.csv").set_index(
                "fold").loc["mean", "auroc"]
        base, mo, ni = (auroc("restricted"), auroc("restricted_median_only"),
                        auroc("restricted_no_indicators"))
        rows.append((NAME[a], f"{base:.4f}", f"{mo:.4f}", f"{mo - base:+.4f}",
                     f"{ni:.4f}", f"{ni - base:+.4f}"))
    md_table(["Algorithm", "As specified", "Median imputation only",
              "Change", "No availability indicators", "Change"], rows)


def explanations():
    prov = json.load(open(R / "shap_provenance.json"))
    w("## 10. Explanations (Section 4.11)")
    w()
    w(f"Exact Shapley values. Global importance is the mean absolute value "
      f"over {prov['global_sample_hours']:,} randomly sampled Dataset A "
      f"patient-hours (random state {prov['random_state']}). Values are on the "
      f"log-odds scale for Logistic Regression and XGBoost and on the "
      f"probability scale for Decision Tree and Random Forest, so compare "
      f"rankings between algorithms, not magnitudes.")
    w()
    for fs in config.FEATURE_SETS:
        tops = {a: pd.read_csv(R / f"shap_global_{config.tag(a, fs)}.csv")
                for a in ALGOS}
        k = min(10, min(len(v) for v in tops.values()))
        w(f"Ten most important features, {fs} feature set:")
        w()
        md_table(["Rank"] + [NAME[a] for a in ALGOS],
                 [(i + 1, *[f"{tops[a].feature.iloc[i]} "
                            f"({tops[a].mean_abs_shap.iloc[i]:.4f})" for a in ALGOS])
                  for i in range(k)])
    c = pd.read_csv(R / "shap_consistency.csv")
    w(f"Consistency of explanations: Spearman rank correlation of global "
      f"importance between the five cross-validation fold models "
      f"({prov['per_fold_sample_hours']:,} held-out hours per fold).")
    w()
    md_table(["Model", "Median", "Lowest pair", "Highest pair"],
             [(model_name(r.model), f"{r.spearman_median:.3f}",
               f"{r.spearman_min:.3f}", f"{r.spearman_max:.3f}")
              for r in c.itertuples()])
    f = R / "shap_local_cases.csv"
    if f.exists():
        d = pd.read_csv(f)
        w(f"Individual explanations, {model_name(prov['individual_explanations_model'])}. "
          f"Baseline plus the sum of all contributions equals the score.")
        w()
        for case in d.case.unique():
            g = d[d.case == case].copy()
            g["absv"] = g.shap_value.abs()
            r0 = g.iloc[0]
            w(f"{case.replace('_', ' ').capitalize()}: sepsis label {r0.label}, "
              f"score {r0.score:.4f}, threshold {r0.threshold:.4f}, baseline "
              f"{r0.baseline:.4f}, {'alerted' if r0.alerted else 'no alert'}.")
            w()
            md_table(["Feature", "Value", "Contribution"],
                     [(r.feature, f"{r.feature_value:.4g}", f"{r.shap_value:+.4f}")
                      for r in g.sort_values("absv", ascending=False).head(8).itertuples()])


def files():
    w("## 11. Where each number comes from")
    w()
    md_table(["File in the results folder", "Contents"], [
        ("cohort.csv, missingness.csv, descriptives_restricted.csv", "Sections 1 and 2"),
        ("cv_*.csv, hyperparameters_*.csv", "Section 3, one file per model"),
        ("performance.csv, confidence_intervals.csv, sirs.csv", "Section 4"),
        ("restricted_vs_full.csv", "Section 5"),
        ("model_vs_sirs.csv", "Section 6"),
        ("calibration_summary.csv, calibration_*.csv, dca_*.csv", "Section 7"),
        ("alert_behaviour.csv", "Section 8"),
        ("shap_global_*.csv, shap_consistency.csv, shap_local_cases.csv", "Section 10"),
        ("bootstrap_raw_A.npz, bootstrap_raw_B.npz", "every bootstrap resample"),
        ("../models/artefacts.json", "columns, medians, thresholds, hyperparameters"),
    ])


def main():
    log("STEP 6  writing the Chapter Four numbers")
    w("# Chapter Four: every number")
    w()
    if config.QUICK:
        w("**QUICK TEST RUN on a small sample of patients. These numbers are "
          "not results and must not be used in the thesis.**")
        w()
    w("Four algorithms (Logistic Regression, Decision Tree, Random Forest, "
      "XGBoost), each trained on the restricted and on the full feature set, "
      "with SIRS as the rule-based comparator. Dataset A is used for "
      "development and internal validation, Dataset B for external validation.")
    w()
    for part in (cohort, missingness, internal, performance, comparisons,
                 calibration, alerts, sensitivity, explanations, files):
        part()
    out = R / "chapter_four_numbers.md"
    out.write_text("\n".join(L) + "\n", encoding="utf-8")
    log(f"STEP 6 finished: {out}")


if __name__ == "__main__":
    main()
