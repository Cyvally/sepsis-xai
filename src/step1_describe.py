"""Step 1. Describe the data: cohort, missingness, distributions (Chapter 4)."""
import numpy as np
import pandas as pd

import config
import prep
from common import log

R = config.RESULTS


def cohort(a, b):
    rows = []
    for name, d in [("A", a), ("B", b)]:
        pat = d.groupby("patient_id")
        septic = pat.SepsisLabel.max()
        age, sex, los = pat.Age.max(), pat.Gender.max(), pat.size()
        # Hours of record before each septic patient's first positive hour.
        hour_index = d.groupby("patient_id").cumcount()
        hours_before = (hour_index[d.SepsisLabel == 1]
                        .groupby(d.patient_id[d.SepsisLabel == 1]).min())
        rows.append({
            "dataset": name,
            "patients": int(d.patient_id.nunique()),
            "patient_hours": int(len(d)),
            "septic_patients": int(septic.sum()),
            "septic_patients_pct": 100 * float(septic.mean()),
            "positive_hours": int(d.SepsisLabel.sum()),
            "positive_hours_pct": 100 * float(d.SepsisLabel.mean()),
            "imbalance_1_to": float((1 - d.SepsisLabel.mean()) / d.SepsisLabel.mean()),
            "age_median": float(age.median()),
            "age_iqr_low": float(age.quantile(.25)),
            "age_iqr_high": float(age.quantile(.75)),
            "age_max": float(age.max()),
            "male_pct": 100 * float(sex.mean()),
            "los_hours_median": float(los.median()),
            "los_hours_iqr_low": float(los.quantile(.25)),
            "los_hours_iqr_high": float(los.quantile(.75)),
            "hours_before_first_positive_median":
                float(hours_before.median()) if len(hours_before) else np.nan,
        })
    return pd.DataFrame(rows)


def missing_table(a, b):
    cols = prep.feature_columns(a)
    ma = prep.missingness(a, cols).set_index("variable")
    mb = prep.missingness(b, cols).set_index("variable")
    fa, fb = prep.forward_fill(a, cols), prep.forward_fill(b, cols)
    lim = prep.ffill_limits(cols)
    out = pd.DataFrame({
        "variable": cols,
        "pct_missing_A": [ma.loc[c, "pct_missing"] for c in cols],
        "pct_missing_B": [mb.loc[c, "pct_missing"] for c in cols],
        "pct_missing_A_after_ffill": [100 * fa[c].isna().mean() for c in cols],
        "pct_missing_B_after_ffill": [100 * fb[c].isna().mean() for c in cols],
        "ffill_limit_h": [lim.get(c, np.nan) for c in cols],
        "in_restricted_set": [c in prep.RESTRICTED for c in cols],
        "n_observed_A": [int(ma.loc[c, "n_observed"]) for c in cols],
    })
    return out.sort_values("pct_missing_A").reset_index(drop=True)


def descriptives(a, b):
    rows = []
    for c in prep.RESTRICTED:
        r = {"variable": c}
        for name, d in [("A", a), ("B", b)]:
            s = d[c].dropna()
            r[f"{name}_median"] = float(s.median())
            r[f"{name}_iqr_low"] = float(s.quantile(.25))
            r[f"{name}_iqr_high"] = float(s.quantile(.75))
        rows.append(r)
    return pd.DataFrame(rows)


def main():
    log("STEP 1  describing the data")
    a, b, n_under_18 = prep.load_datasets()
    log(f"  Dataset A: {a.patient_id.nunique():,} patients, {len(a):,} hours")
    log(f"  Dataset B: {b.patient_id.nunique():,} patients, {len(b):,} hours "
        f"({n_under_18} patients under 18 excluded)")
    c = cohort(a, b)
    c["under_18_excluded"] = [0, n_under_18]
    c.to_csv(R / "cohort.csv", index=False)
    missing_table(a, b).to_csv(R / "missingness.csv", index=False)
    descriptives(a, b).to_csv(R / "descriptives_restricted.csv", index=False)
    prep.range_flags(a).to_csv(R / "range_flags_A.csv", index=False)
    prep.range_flags(b).to_csv(R / "range_flags_B.csv", index=False)
    log("  wrote cohort, missingness, descriptives and range checks")


if __name__ == "__main__":
    main()
