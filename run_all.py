"""Run the whole analysis, step by step.

    python run_all.py --quick     a 5-minute test on a small sample of patients
    python run_all.py             the real run on all patients (several hours)

If the real run is interrupted, run the same command again. Finished parts
are kept and it continues from where it stopped.
"""
import os
import sys
import time
from pathlib import Path

if "--quick" in sys.argv:
    os.environ["SEPSIS_QUICK"] = "1"

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

try:
    import numpy, pandas, sklearn, xgboost, shap, matplotlib, joblib, pyarrow, scipy  # noqa
except ImportError as e:
    raise SystemExit(
        f"\nA required library is missing: {e.name}\n"
        f"Install the libraries first:   pip install -r requirements.txt\n")

import config                      # noqa: E402
from common import log, took       # noqa: E402
import step1_describe              # noqa: E402
import step2_cross_validation      # noqa: E402
import step3_external_validation   # noqa: E402
import step4_explain               # noqa: E402
import step5_figures               # noqa: E402
import step6_report                # noqa: E402


def main():
    t0 = time.time()
    mode = ("QUICK TEST (small sample, results are NOT for the thesis)"
            if config.QUICK else "FULL RUN (all patients)")
    print("=" * 72)
    print(f"Sepsis prediction analysis: {mode}")
    print(f"Output folder: {config.OUT}")
    print("=" * 72, flush=True)
    for step in (step1_describe, step2_cross_validation,
                 step3_external_validation, step4_explain, step5_figures,
                 step6_report):
        step.main()
        print("-" * 72, flush=True)
    print("=" * 72)
    print(f"ALL DONE in {took(t0)}.")
    print(f"Numbers for Chapter Four: {config.RESULTS / 'chapter_four_numbers.md'}")
    print(f"Figures:                  {config.FIGURES}")
    print(f"Trained models:           {config.MODELS}")
    if config.QUICK:
        print("\nThe quick test worked. Now start the real run with:"
              "\n    python run_all.py")
    print("=" * 72)


if __name__ == "__main__":
    main()
