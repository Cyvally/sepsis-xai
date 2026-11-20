"""Preprocessing for the sepsis prediction pipeline.

Implements Section 3.9 of the methodology:
  - causal forward fill within patient, with per-variable carry-forward limits
  - availability indicators recorded before filling
  - median imputation with medians fitted on training data only
  - SIRS computed from forward-filled values (Section 3.13)
"""
import numpy as np
import pandas as pd

import config

# Workflow / elapsed-time proxies, excluded from both feature sets (Section 3.6.2)
EXCLUDE = ["Unit1", "Unit2", "HospAdmTime", "ICULOS"]
META = ["SepsisLabel", "patient_id", "hospital_system"]

STATIC = ["Age", "Gender"]

# Carry-forward limits in hours (Section 3.9)
FFILL_4H = ["HR", "O2Sat", "SBP", "MAP", "DBP", "Resp"]
FFILL_8H = ["Temp"]
# everything else (laboratory and other intermittently recorded variables): 24h

RESTRICTED = ["HR", "Resp", "Temp", "SBP", "DBP", "O2Sat", "Age", "Gender"]

# Plausible physiological ranges, for flagging only (Section 3.9: flagged, not deleted)
RANGES = {
    "HR": (10, 300), "O2Sat": (10, 100), "Temp": (20, 45), "SBP": (20, 300),
    "MAP": (10, 250), "DBP": (10, 200), "Resp": (0, 90), "Age": (0, 120),
}


def load(path):
    if not path.exists():
        raise SystemExit(
            f"\nData file not found: {path}\n"
            f"Put setA.parquet and setB.parquet inside the 'data' folder.\n")
    df = pd.read_parquet(path)
    return df.sort_values(["patient_id", "ICULOS"], kind="mergesort").reset_index(drop=True)


def load_datasets():
    """Dataset A, Dataset B (adults only) and the number of patients excluded.

    Patients recorded as under 18 are removed from Dataset B before any
    analysis. In quick mode a random sample of patients is kept from each.
    """
    a = load(config.SET_A)
    b = load(config.SET_B)
    age = b.groupby("patient_id")["Age"].max()
    keep = age[age >= 18].index
    n_under_18 = int(b.patient_id.nunique() - len(keep))
    b = b[b.patient_id.isin(keep)].reset_index(drop=True)
    if config.QUICK:
        rng = np.random.default_rng(config.SEED)
        out = []
        for d in (a, b):
            ids = d.patient_id.unique()
            pick = rng.choice(ids, min(config.QUICK_PATIENTS, len(ids)), replace=False)
            out.append(d[d.patient_id.isin(pick)].reset_index(drop=True))
        a, b = out
    return a, b, n_under_18


def feature_columns(df):
    """The 36 eligible variables: everything but workflow proxies and metadata."""
    return [c for c in df.columns if c not in EXCLUDE + META]


def ffill_limits(cols):
    out = {}
    for c in cols:
        if c in STATIC:
            continue
        if c in FFILL_4H:
            out[c] = 4
        elif c in FFILL_8H:
            out[c] = 8
        else:
            out[c] = 24
    return out


def range_flags(df):
    """Count implausible values per variable. Flagged and reported, never deleted."""
    rows = []
    for c, (lo, hi) in RANGES.items():
        if c in df.columns:
            s = df[c]
            n = int(((s < lo) | (s > hi)).sum())
            rows.append({"variable": c, "lower": lo, "upper": hi,
                         "n_outside": n,
                         "pct_outside": 100 * n / s.notna().sum() if s.notna().any() else 0.0})
    return pd.DataFrame(rows)


def missingness(df, cols):
    n = len(df)
    return pd.DataFrame({
        "variable": cols,
        "n_observed": [int(df[c].notna().sum()) for c in cols],
        "pct_missing": [100 * df[c].isna().sum() / n for c in cols],
    }).sort_values("pct_missing").reset_index(drop=True)


def build_indicators(df, cols):
    """Availability indicator: 1 if the value was measured at this hour, 0 if not.

    Computed from the raw data, before any filling (Section 3.9).
    Static variables are excluded: they have no missingness, so an indicator
    would be a constant column.
    """
    ind_cols = [c for c in cols if c not in STATIC and df[c].isna().any()]
    ind = pd.DataFrame(
        {f"{c}_obs": df[c].notna().to_numpy(dtype=np.float32) for c in ind_cols},
        index=df.index,
    )
    return ind, ind_cols


def forward_fill(df, cols):
    """Causal forward fill within patient, with per-variable limits.

    No backward fill: a prediction at hour t uses only hours <= t.
    """
    out = df[cols].copy()
    limits = ffill_limits(cols)
    g = df["patient_id"]
    for lim in sorted(set(limits.values())):
        group = [c for c, L in limits.items() if L == lim]
        if group:
            out[group] = out[group].groupby(g).ffill(limit=lim)
    # Static variables are constant within patient; fill both directions.
    for c in cols:
        if c in STATIC:
            out[c] = out[c].groupby(g).ffill().groupby(g).bfill()
    return out


def fit_medians(filled, cols):
    """Medians fitted on training data only (Section 3.9, leakage control)."""
    return {c: float(filled[c].median()) for c in cols}


def apply_medians(filled, medians):
    return filled.fillna(value=medians)


# --------------------------------------------------------------- SIRS
def sirs(filled_raw):
    """SIRS score 0-4 from forward-filled values, no median imputation.

    Section 3.13. A criterion with no available value counts as not met.
    Band cells are not recorded in this dataset, so that part of the WBC
    criterion cannot be applied.

    Returns the score and a coverage flag: whether all four criteria were
    evaluable at that hour.
    """
    t = filled_raw["Temp"]
    hr = filled_raw["HR"]
    rr = filled_raw["Resp"]
    pc = filled_raw["PaCO2"]
    wbc = filled_raw["WBC"]

    c_temp = ((t > 38) | (t < 36)).fillna(False)
    c_hr = (hr > 90).fillna(False)
    c_resp = ((rr > 20) | (pc < 32)).fillna(False)
    c_wbc = ((wbc > 12) | (wbc < 4)).fillna(False)

    score = (c_temp.astype(int) + c_hr.astype(int)
             + c_resp.astype(int) + c_wbc.astype(int))
    evaluable = (t.notna() & hr.notna() & (rr.notna() | pc.notna()) & wbc.notna())
    return score.to_numpy(), evaluable.to_numpy()


# --------------------------------------------------------------- assembly
class Preprocessor:
    """Fitted on Dataset A, applied unchanged to Dataset B (Section 3.12)."""

    def __init__(self, feature_set="restricted", use_indicators=True,
                 median_only=False):
        self.feature_set = feature_set
        self.use_indicators = use_indicators
        self.median_only = median_only
        self.medians_ = None
        self.columns_ = None

    def _base_cols(self, df):
        if self.feature_set == "restricted":
            return list(RESTRICTED)
        return feature_columns(df)

    def fit_transform(self, df):
        cols = self._base_cols(df)
        # A variable never recorded in the training data has no median and
        # carries no information. Dropped here and reported (Section 3.9).
        self.dropped_ = [c for c in cols if df[c].notna().sum() == 0]
        cols = [c for c in cols if c not in self.dropped_]
        self.base_cols_ = cols
        ind, ind_cols = build_indicators(df, cols)
        if self.median_only:
            filled = df[cols].copy()
            for c in cols:
                if c in STATIC:
                    filled[c] = filled[c].groupby(df["patient_id"]).ffill().groupby(df["patient_id"]).bfill()
        else:
            filled = forward_fill(df, cols)
        self.medians_ = fit_medians(filled, cols)
        X = apply_medians(filled, self.medians_)
        if self.use_indicators and ind_cols:
            X = pd.concat([X, ind], axis=1)
        self.columns_ = list(X.columns)
        return X.astype(np.float32)

    def transform(self, df):
        if self.medians_ is None:
            raise RuntimeError("Preprocessor must be fitted first")
        cols = self.base_cols_
        ind, ind_cols = build_indicators(df, cols)
        if self.median_only:
            filled = df[cols].copy()
            for c in cols:
                if c in STATIC:
                    filled[c] = filled[c].groupby(df["patient_id"]).ffill().groupby(df["patient_id"]).bfill()
        else:
            filled = forward_fill(df, cols)
        X = apply_medians(filled, self.medians_)
        if self.use_indicators and ind_cols:
            X = pd.concat([X, ind], axis=1)
        # Align to the columns seen at fit time; a missing indicator becomes 0.
        for c in self.columns_:
            if c not in X.columns:
                X[c] = 0.0
        X = X[self.columns_]
        return X.astype(np.float32)
