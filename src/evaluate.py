"""Evaluation metrics for the sepsis prediction pipeline (Sections 3.13, 3.15)."""
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
from sklearn.linear_model import LogisticRegression


def threshold_at_sensitivity(y, p, target=0.80):
    """Smallest threshold achieving at least `target` sensitivity.

    Chosen on training data only, then applied unchanged to held-out data
    (Section 3.13).
    """
    pos = np.sort(p[y == 1])[::-1]
    if pos.size == 0:
        return 0.5
    k = int(np.ceil(target * pos.size)) - 1
    k = min(max(k, 0), pos.size - 1)
    return float(pos[k])


def at_threshold(y, p, thr):
    pred = (p >= thr).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    tn = int(((pred == 0) & (y == 0)).sum())
    sens = tp / (tp + fn) if tp + fn else np.nan
    spec = tn / (tn + fp) if tn + fp else np.nan
    ppv = tp / (tp + fp) if tp + fp else np.nan
    npv = tn / (tn + fn) if tn + fn else np.nan
    f1 = 2 * ppv * sens / (ppv + sens) if (ppv + sens) else np.nan
    return {"threshold": thr, "sensitivity": sens, "specificity": spec,
            "ppv": ppv, "npv": npv, "f1": f1,
            "alert_rate": pred.mean(), "tp": tp, "fp": fp, "fn": fn, "tn": tn}


def calibration(y, p):
    """Calibration slope and intercept from a logistic recalibration model."""
    eps = 1e-7
    lp = np.log(np.clip(p, eps, 1 - eps) / (1 - np.clip(p, eps, 1 - eps)))
    lr = LogisticRegression(penalty=None, solver="lbfgs", max_iter=1000)
    lr.fit(lp.reshape(-1, 1), y)
    return {"calibration_slope": float(lr.coef_[0][0]),
            "calibration_intercept": float(lr.intercept_[0]),
            "brier": float(brier_score_loss(y, p))}


def calibration_curve_points(y, p, n_bins=10):
    """Observed vs predicted risk by decile of predicted risk."""
    q = np.quantile(p, np.linspace(0, 1, n_bins + 1))
    q[0], q[-1] = -np.inf, np.inf
    idx = np.digitize(p, q[1:-1])
    rows = []
    for b in range(n_bins):
        m = idx == b
        if m.sum() == 0:
            continue
        rows.append({"bin": b + 1, "n": int(m.sum()),
                     "mean_predicted": float(p[m].mean()),
                     "observed": float(y[m].mean())})
    return pd.DataFrame(rows)


def core_metrics(y, p):
    out = {"auroc": float(roc_auc_score(y, p)),
           "auprc": float(average_precision_score(y, p))}
    out.update(calibration(y, p))
    return out


def decision_curve(y, p, thresholds=None):
    """Net benefit against treat-all and treat-none (Vickers & Elkin, 2006)."""
    if thresholds is None:
        thresholds = np.concatenate([np.arange(0.005, 0.05, 0.005),
                                     np.arange(0.05, 0.55, 0.025)])
    n = len(y)
    prev = y.mean()
    rows = []
    for t in thresholds:
        pred = p >= t
        tp = np.sum(pred & (y == 1))
        fp = np.sum(pred & (y == 0))
        nb = tp / n - (fp / n) * (t / (1 - t))
        nb_all = prev - (1 - prev) * (t / (1 - t))
        rows.append({"threshold": float(t), "net_benefit_model": float(nb),
                     "net_benefit_treat_all": float(nb_all),
                     "net_benefit_treat_none": 0.0})
    return pd.DataFrame(rows)


# ------------------------------------------------------- cluster bootstrap
def _prepare(y, p, inv):
    """Sort once by descending score and mark where tied scores start."""
    order = np.argsort(-p, kind="stable")
    ps = p[order]
    starts = np.flatnonzero(np.r_[True, ps[1:] != ps[:-1]])
    return y[order].astype(np.float64), inv[order], starts


def _weighted_auroc_auprc(ys, w, starts):
    """AUROC and average precision with patient weights, ties handled exactly.

    Rows are already sorted by descending score. `starts` marks the first row
    of each group of tied scores.
    """
    wp = w * ys
    pos = np.add.reduceat(wp, starts)
    neg = np.add.reduceat(w - wp, starts)
    P, N = pos.sum(), neg.sum()
    if P == 0 or N == 0:
        return np.nan, np.nan
    cneg = np.cumsum(neg)
    ctp = np.cumsum(pos)
    auroc = float((pos * (N - cneg + 0.5 * neg)).sum() / (P * N))
    denom = ctp + cneg
    prec = np.divide(ctp, denom, out=np.zeros_like(ctp), where=denom > 0)
    auprc = float((pos / P * prec).sum())
    return auroc, auprc


def cluster_bootstrap(y, preds, groups, n_boot=2000, seed=7):
    """Patient-level cluster bootstrap (Section 3.13).

    Patients, not patient-hours, are resampled with replacement, because hours
    from the same patient are not independent. Drawing a patient k times is
    the same as giving every one of that patient's hours a weight of k, which
    is how it is computed here: each model's scores are sorted once, and each
    replicate only changes the weights. Every model sees the same patients in
    each replicate, so differences between models are paired.

    `preds` is a dict of name -> score array. Returns name_auroc and
    name_auprc arrays with one value per replicate.
    """
    rng = np.random.default_rng(seed)
    uniq, inv = np.unique(groups, return_inverse=True)
    G = len(uniq)
    prepared = {n: _prepare(y, np.asarray(p, dtype=np.float64), inv)
                for n, p in preds.items()}
    out = {f"{n}_{m}": np.empty(n_boot) for n in preds for m in ("auroc", "auprc")}
    for b in range(n_boot):
        counts = np.bincount(rng.integers(0, G, G), minlength=G).astype(np.float64)
        for n, (ys, inv_sorted, starts) in prepared.items():
            a, ap = _weighted_auroc_auprc(ys, counts[inv_sorted], starts)
            out[f"{n}_auroc"][b] = a
            out[f"{n}_auprc"][b] = ap
    return out


def ci(arr, level=0.95):
    lo, hi = (1 - level) / 2, 1 - (1 - level) / 2
    return float(np.nanquantile(arr, lo)), float(np.nanquantile(arr, hi))


def logit(p, eps=1e-7):
    p = np.clip(p, eps, 1 - eps)
    return np.log(p / (1 - p))
