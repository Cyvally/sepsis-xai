"""Step 5. Chapter Four figures, saved as PNG files at 300 dpi."""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve, roc_curve

import config
import prep
from common import log

R, F = config.RESULTS, config.FIGURES

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Liberation Serif", "DejaVu Serif"],
    "font.size": 10, "axes.titlesize": 11, "axes.labelsize": 10,
    "savefig.dpi": 300, "savefig.bbox": "tight",
    "axes.spines.top": False, "axes.spines.right": False,
})

COLOR = {"logistic_regression": "#1F6FB4", "decision_tree": "#7B4EA3",
         "random_forest": "#C55A11", "xgboost": "#2E8B57", "sirs": "#444444"}
ALGOS = ["logistic_regression", "decision_tree", "random_forest", "xgboost"]
NAME = dict(config.ALGO_LABEL, sirs="SIRS")
SETS = config.FEATURE_SETS
SET_TITLE = {"restricted": "Restricted feature set", "full": "Full feature set"}


def save(fig, name):
    fig.savefig(F / f"{name}.png")
    plt.close(fig)
    log(f"  wrote {name}.png")


def thin(n, k=3000):
    return np.unique(np.linspace(0, n - 1, min(n, k)).astype(int))


def fig_missingness(n):
    d = pd.read_csv(R / "missingness.csv").sort_values("pct_missing_A")
    fig, ax = plt.subplots(figsize=(7.5, 0.24 * len(d) + 1.2))
    yy = np.arange(len(d))
    ax.barh(yy - 0.2, d.pct_missing_A, height=0.4, color="#1F6FB4", label="Dataset A")
    ax.barh(yy + 0.2, d.pct_missing_B, height=0.4, color="#C55A11", label="Dataset B")
    ax.set_yticks(yy)
    ax.set_yticklabels(d.variable, fontsize=8)
    ax.set(xlabel="Patient-hours with no recorded value (%)", xlim=(0, 100))
    ax.legend(frameon=False, loc="lower right")
    ax.set_title(f"Figure 4.{n}: Missingness by variable and hospital system",
                 fontweight="bold", loc="left")
    save(fig, f"fig4_{n}_missingness")


def fig_auroc(n):
    ci = pd.read_csv(R / "confidence_intervals.csv")
    ci = ci[ci.metric == "auroc"]
    order = [config.tag(a, s) for s in SETS for a in ALGOS] + ["sirs"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.6), sharey=True)
    for ax, ds in zip(axes, ["A", "B"]):
        d = ci[ci.dataset == ds].set_index("model").loc[order]
        yy = np.arange(len(order))[::-1]
        for y, (m, r) in zip(yy, d.iterrows()):
            algo = next((a for a in ALGOS if m.startswith(a)), "sirs")
            ax.errorbar(r["mean"], y, xerr=[[r["mean"] - r.ci_low], [r.ci_high - r["mean"]]],
                        fmt="o", color=COLOR[algo], capsize=3, ms=5)
            ax.text(r.ci_high + 0.004, y, f"{r['mean']:.3f}", va="center", fontsize=8)
        ax.set_yticks(yy)
        ax.set_yticklabels([("SIRS" if m == "sirs" else
                             f"{NAME[next(a for a in ALGOS if m.startswith(a))]}, "
                             f"{'restricted' if m.endswith('restricted') else 'full'}")
                            for m in order], fontsize=8.5)
        ax.axvline(0.5, ls=":", color="#999999", lw=1)
        ax.margins(x=0.12)
        ax.set(xlabel="AUROC with 95% confidence interval",
               title="Dataset A (internal, out-of-fold)" if ds == "A"
               else "Dataset B (external)")
    fig.suptitle(f"Figure 4.{n}: Discrimination of every model and of SIRS",
                 fontweight="bold", y=1.0)
    fig.tight_layout()
    save(fig, f"fig4_{n}_auroc_all_models")


def fig_margin(n):
    d = pd.read_csv(R / "restricted_vs_full.csv")
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.4), sharey=True, sharex=True)
    for ax, ds in zip(axes, ["A", "B"]):
        g = d[d.dataset == ds].set_index("algorithm").loc[ALGOS]
        yy = np.arange(len(g))[::-1]
        for y, (algo, r) in zip(yy, g.iterrows()):
            ax.errorbar(r.delta_auroc, y,
                        xerr=[[r.delta_auroc - r.delta_auroc_ci_low],
                              [r.delta_auroc_ci_high - r.delta_auroc]],
                        fmt="o", color=COLOR[algo], capsize=3, ms=5)
            ax.text(r.delta_auroc_ci_high + 0.004, y, f"{r.delta_auroc:+.3f}",
                    fontsize=8, va="center")
        line = ax.axvline(config.MARGIN, ls="--", color="#B00020", lw=1.2)
        ax.axvline(0, ls=":", color="#999999", lw=1)
        ax.set_yticks(yy)
        ax.set_yticklabels([NAME[a] for a in g.index])
        ax.set_ylim(-0.6, len(g) - 0.4)
        ax.margins(x=0.12)
        ax.set(xlabel="AUROC, restricted minus full (95% CI)",
               title=f"Dataset {ds}")
    fig.suptitle(f"Figure 4.{n}: Cost of restricting to bedside variables, "
                 f"by algorithm", fontweight="bold", y=1.0)
    fig.legend([line], [f"Pre-registered margin ({config.MARGIN}): the "
                        f"criterion is met only if the whole interval lies "
                        f"to the right of this line"],
               frameon=False, fontsize=8.5, loc="lower center")
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    save(fig, f"fig4_{n}_restricted_vs_full")


def load_scores():
    y = {"A": np.load(R / "y_A.npy")}
    _, b, _ = prep.load_datasets()
    y["B"] = b.SepsisLabel.to_numpy()
    s = {"A": {"sirs": np.load(R / "sirs_A.npy")},
         "B": {"sirs": np.load(R / "sirs_B.npy")}}
    for fs in SETS:
        for a in ALGOS:
            t = config.tag(a, fs)
            s["A"][t] = np.load(R / f"oof_{t}.npy")
            s["B"][t] = np.load(R / f"pred_b_{t}.npy")
    return y, s


def fig_curves(n, y, s, kind):
    fig, axes = plt.subplots(2, 2, figsize=(10, 8.6))
    for i, ds in enumerate(["A", "B"]):
        for j, fs in enumerate(SETS):
            ax = axes[i, j]
            for name in ALGOS + ["sirs"]:
                key = "sirs" if name == "sirs" else config.tag(name, fs)
                style = dict(color=COLOR[name], lw=1.6,
                             ls="--" if name == "sirs" else "-", label=NAME[name])
                if kind == "roc":
                    fpr, tpr, _ = roc_curve(y[ds], s[ds][key])
                    k = thin(len(fpr))
                    ax.plot(fpr[k], tpr[k], **style)
                else:
                    pr, rc, _ = precision_recall_curve(y[ds], s[ds][key])
                    k = thin(len(pr))
                    ax.plot(rc[k], pr[k], **style)
            if kind == "roc":
                ax.plot([0, 1], [0, 1], ls=":", color="#999999", lw=1)
                ax.set(xlabel="False positive rate", ylabel="True positive rate",
                       xlim=(0, 1), ylim=(0, 1))
            else:
                ax.axhline(y[ds].mean(), ls=":", color="#999999", lw=1)
                ax.set(xlabel="Recall", ylabel="Precision", xlim=(0, 1),
                       ylim=(0, max(0.2, 6 * y[ds].mean())))
            ax.set_title(f"Dataset {ds}: {SET_TITLE[fs].lower()}")
            ax.legend(frameon=False, fontsize=8)
    what = ("Receiver operating characteristic curves" if kind == "roc"
            else "Precision-recall curves (dotted line: base rate)")
    fig.suptitle(f"Figure 4.{n}: {what}", fontweight="bold", y=1.0)
    fig.tight_layout()
    save(fig, f"fig4_{n}_{kind}_curves")


def fig_calibration(n):
    fig, axes = plt.subplots(2, 2, figsize=(10, 8.6))
    for i, ds in enumerate(["A", "B"]):
        for j, fs in enumerate(SETS):
            ax, lim = axes[i, j], 0
            for a in ALGOS:
                d = pd.read_csv(R / f"calibration_{config.tag(a, fs)}_{ds}.csv")
                ax.plot(d.mean_predicted, d.observed, "o-", color=COLOR[a],
                        lw=1.4, ms=3.5, label=NAME[a])
                lim = max(lim, d.mean_predicted.max(), d.observed.max())
            ax.plot([0, lim], [0, lim], ls=":", color="#999999", lw=1,
                    label="Perfect calibration")
            ax.set(xlabel="Mean predicted risk", ylabel="Observed proportion",
                   title=f"Dataset {ds}: {SET_TITLE[fs].lower()}")
            ax.legend(frameon=False, fontsize=8)
    fig.suptitle(f"Figure 4.{n}: Calibration by decile of predicted risk, "
                 f"after recalibration on Dataset A", fontweight="bold", y=1.0)
    fig.tight_layout()
    save(fig, f"fig4_{n}_calibration")


def fig_dca(n):
    fig, axes = plt.subplots(2, 2, figsize=(10, 8.6))
    for i, ds in enumerate(["A", "B"]):
        for j, fs in enumerate(SETS):
            ax, top = axes[i, j], 0.0
            ref = pd.read_csv(R / f"dca_{config.tag(ALGOS[0], fs)}_{ds}.csv")
            ref = ref[(ref.threshold >= 0.005) & (ref.threshold <= 0.15)]
            ax.plot(ref.threshold, ref.net_benefit_treat_all, ls="--",
                    color="#999999", lw=1.2, label="Treat all")
            ax.axhline(0, ls=":", color="#555555", lw=1.2, label="Treat none")
            for a in ALGOS:
                d = pd.read_csv(R / f"dca_{config.tag(a, fs)}_{ds}.csv")
                d = d[(d.threshold >= 0.005) & (d.threshold <= 0.15)]
                ax.plot(d.threshold, d.net_benefit_model, color=COLOR[a],
                        lw=1.6, label=NAME[a])
                top = max(top, d.net_benefit_model.max())
            ax.set_ylim(-0.3 * top, 1.25 * top)
            ax.set(xlabel="Threshold probability", ylabel="Net benefit",
                   title=f"Dataset {ds}: {SET_TITLE[fs].lower()}")
            ax.legend(frameon=False, fontsize=8)
    fig.suptitle(f"Figure 4.{n}: Decision curve analysis on recalibrated risks",
                 fontweight="bold", y=1.0)
    fig.tight_layout()
    save(fig, f"fig4_{n}_decision_curves")


def fig_shap_global(n, fs, top):
    fig, axes = plt.subplots(2, 2, figsize=(10, 0.5 * top + 3.2))
    for ax, a in zip(axes.ravel(), ALGOS):
        d = pd.read_csv(R / f"shap_global_{config.tag(a, fs)}.csv").head(top).iloc[::-1]
        ax.barh(d.feature, d.mean_abs_shap, color=COLOR[a], height=0.7)
        ax.set_title(NAME[a])
        ax.set_xlabel(f"Mean absolute SHAP value ({d.scale.iloc[0]})", fontsize=8.5)
        ax.tick_params(axis="y", labelsize=8)
    extra = "" if fs == "restricted" else f", top {top} features"
    fig.suptitle(f"Figure 4.{n}: Global feature importance, "
                 f"{SET_TITLE[fs].lower()}{extra}", fontweight="bold", y=1.0)
    fig.tight_layout()
    save(fig, f"fig4_{n}_shap_global_{fs}")


def fig_consistency(n):
    d = pd.read_csv(R / "shap_consistency.csv").set_index("model")
    fig, ax = plt.subplots(figsize=(7.5, 3.6))
    order = [config.tag(a, s) for s in SETS for a in ALGOS]
    yy = np.arange(len(order))[::-1]
    for y, m in zip(yy, order):
        r = d.loc[m]
        algo = next(a for a in ALGOS if m.startswith(a))
        ax.errorbar(r.spearman_median, y,
                    xerr=[[r.spearman_median - r.spearman_min],
                          [r.spearman_max - r.spearman_median]],
                    fmt="o", color=COLOR[algo], capsize=3, ms=5)
        ax.text(min(r.spearman_max + 0.01, 1.0), y + 0.2,
                f"{r.spearman_median:.3f}", fontsize=8)
    ax.set_yticks(yy)
    ax.set_yticklabels([f"{NAME[next(a for a in ALGOS if m.startswith(a))]}, "
                        f"{'restricted' if m.endswith('restricted') else 'full'}"
                        for m in order], fontsize=8.5)
    ax.set_xlim(right=1.03)
    ax.set_xlabel("Spearman rank correlation of global importance between fold "
                  "models\n(point: median; bar: lowest to highest pair)")
    ax.set_title(f"Figure 4.{n}: Consistency of explanations across "
                 f"cross-validation folds", fontweight="bold", loc="left")
    save(fig, f"fig4_{n}_explanation_consistency")


def fig_local(n):
    f = R / "shap_local_cases.csv"
    if not f.exists():
        return
    d = pd.read_csv(f)
    which = json.load(open(R / "shap_provenance.json"))["individual_explanations_model"]
    cases = [c for c in ("confident_true_positive", "confident_false_positive",
                         "default_false_positive", "correct_rejection")
             if c in set(d.case)]
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 8.6))
    for ax, case in zip(axes.ravel(), cases):
        g = d[d.case == case].copy()
        g["absv"] = g.shap_value.abs()
        g = g.sort_values("absv").tail(8)
        ax.barh([f"{r.feature} = {r.feature_value:.4g}" for r in g.itertuples()],
                g.shap_value,
                color=["#C55A11" if v > 0 else "#1F6FB4" for v in g.shap_value],
                height=0.7)
        ax.axvline(0, color="#333333", lw=1)
        r = g.iloc[0]
        ax.set_title(f"{case.replace('_', ' ').capitalize()}: "
                     f"{'alerted' if r.alerted else 'no alert'}\n"
                     f"baseline {r.baseline:.3f}, score {r.score:.3f}, "
                     f"threshold {r.threshold:.3f}", fontsize=9.5)
        ax.set_xlabel("SHAP value (contribution to the score)", fontsize=8.5)
        ax.tick_params(axis="y", labelsize=8)
    for ax in axes.ravel()[len(cases):]:
        ax.set_visible(False)
    fig.suptitle(f"Figure 4.{n}: Individual explanations, {config.label(which)}.\n"
                 f"Orange raises the score, blue lowers it.",
                 fontweight="bold", y=1.0)
    fig.tight_layout()
    save(fig, f"fig4_{n}_individual_explanations")


def main():
    log("STEP 5  figures")
    fig_missingness(1)
    fig_auroc(2)
    fig_margin(3)
    y, s = load_scores()
    fig_curves(4, y, s, "roc")
    fig_curves(5, y, s, "pr")
    fig_calibration(6)
    fig_dca(7)
    fig_shap_global(8, "restricted", 14)
    fig_shap_global(9, "full", 15)
    fig_consistency(10)
    fig_local(11)
    log(f"STEP 5 finished: figures are in {F}")


if __name__ == "__main__":
    main()
