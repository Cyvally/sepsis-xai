"""Scoring prototype (Section 3.14).

Loads one saved model with its preprocessing parameters, decision threshold
and recalibration map, scores a patient's hourly record, and explains the
score. Run the full analysis first, so that the models exist.

    python src/predict.py                              Random Forest, restricted
    python src/predict.py xgboost full                 any algorithm and feature set
"""
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import algorithms as alg      # noqa: E402
import config                 # noqa: E402
import evaluate as ev         # noqa: E402
import prep                   # noqa: E402


class SepsisScorer:
    def __init__(self, algorithm="random_forest", feature_set="restricted"):
        t = config.tag(algorithm, feature_set)
        art = json.load(open(config.MODELS / "artefacts.json"))[t]
        self.algorithm, self.tag = algorithm, t
        self.columns = art["columns"]
        self.medians = art["medians"]
        self.limits = art["ffill_limits"]
        self.threshold = art["decision_threshold"]
        self.model = joblib.load(config.MODELS / f"{t}.joblib")
        self.recalibrator = joblib.load(config.MODELS / f"recalibrator_{t}.joblib")

    def design_matrix(self, df):
        """The training preprocessing, reproduced for one patient.

        Availability indicators are read from the raw record first, then
        values are carried forward within their limits, then anything still
        missing takes the Dataset A median.
        """
        d = df.sort_values("ICULOS").reset_index(drop=True)
        out = pd.DataFrame(index=d.index)
        for c in self.columns:
            if c.endswith("_obs"):
                out[c] = d[c[:-4]].notna().astype(float)
            elif c in prep.STATIC:
                out[c] = d[c].ffill().bfill().fillna(self.medians[c])
            else:
                out[c] = d[c].ffill(limit=int(self.limits[c])).fillna(self.medians[c])
        return out[self.columns].to_numpy(dtype=np.float32), d.ICULOS.to_numpy()

    def score(self, df):
        X, hours = self.design_matrix(df)
        s = alg.predict(self.model, X)
        risk = self.recalibrator.predict_proba(ev.logit(s).reshape(-1, 1))[:, 1]
        return X, pd.DataFrame({"hour": hours, "score": s, "calibrated_risk": risk,
                                "alert": s >= self.threshold})

    def report(self, df, top=6):
        X, h = self.score(df)
        i = int(np.flatnonzero(h.alert)[0]) if h.alert.any() else int(h.score.idxmax())
        values, base = alg.explain(self.model, self.algorithm, X[[i]])
        scale = alg.EXPLANATION_SCALE[self.algorithm]
        lines = [f"Model: {config.label(self.tag)}",
                 (f"First alert at hour {int(h.hour[i])} of the record."
                  if h.alert.any() else
                  f"No alert in this record. Highest score was at hour {int(h.hour[i])}."),
                 f"Score {h.score[i]:.3f} against a threshold of {self.threshold:.3f}.",
                 f"Calibrated risk of sepsis within 6 hours: {100*h.calibrated_risk[i]:.1f}%.",
                 f"Alerts in {int(h.alert.sum())} of {len(h)} recorded hours.",
                 "",
                 f"What moved the score at that hour ({scale} scale, baseline {base:.3f}):"]
        for j in np.argsort(-np.abs(values[0]))[:top]:
            c, v = self.columns[j], values[0][j]
            name = f"{c[:-4]} recorded this hour (1 = yes)" if c.endswith("_obs") else c
            lines.append(f"  {name} = {X[i, j]:.4g}: "
                         f"{'raises' if v > 0 else 'lowers'} it by {abs(v):.4f}")
        lines += ["", "This is a post-hoc attribution of the score to the inputs. "
                      "It does not show that the score is correct."]
        return "\n".join(lines)


if __name__ == "__main__":
    algorithm = sys.argv[1] if len(sys.argv) > 1 else "random_forest"
    feature_set = sys.argv[2] if len(sys.argv) > 2 else "restricted"
    a, _, _ = prep.load_datasets()
    scorer = SepsisScorer(algorithm, feature_set)
    septic = a.groupby("patient_id").SepsisLabel.max()
    for what, pid in [("a patient who later met the sepsis definition",
                       septic[septic == 1].index[0]),
                      ("a patient who never met it", septic[septic == 0].index[0])]:
        print("=" * 70)
        print(f"Patient {pid}: {what}")
        print("=" * 70)
        print(scorer.report(a[a.patient_id == pid]))
        print()
