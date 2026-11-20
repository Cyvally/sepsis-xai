# How to run the analysis

You will run one command for a 5-minute test, then one command for the real
run. The real run takes roughly 1.5 to 3 hours on a normal laptop.

## What is in this folder

| Item | What it is |
|---|---|
| `data/` | Your two data files, `setA.parquet` and `setB.parquet` |
| `src/` | The code. `algorithms.py` is where the four algorithms are called |
| `run_all.py` | The one file you run |
| `requirements.txt` | The list of libraries to install |

After the run, three new folders appear: `results/` (all the numbers),
`figures/` (the Chapter Four figures) and `models/` (the eight trained models).

## Step 1. Install Python (once)

VS Code is only an editor. It needs Python installed separately.

1. Go to python.org, open Downloads, and download **Python 3.13**.
2. Run the installer.
   **On Windows, tick the box "Add python.exe to PATH" on the first screen
   before you click Install.** This is the step people miss.
3. Close VS Code completely and open it again.

If you already have Python 3.11 or newer, skip this step.

## Step 2. Open the project in VS Code

1. Unzip `sepsis_project.zip`. Put the `sepsis_project` folder somewhere easy,
   such as Documents.
2. In VS Code: **File > Open Folder**, choose the `sepsis_project` folder.
   You should see `data`, `src` and `run_all.py` in the left panel.
3. If VS Code offers to install the "Python" extension, accept.
4. Open the terminal: **Terminal > New Terminal**. A panel opens at the
   bottom. Every command below is typed there, followed by Enter.

## Step 3. Check Python works

    python --version

You should see `Python 3.13.x` (or 3.11, 3.12).

If you see an error instead:
- Windows: try `py --version`. If that works, use `py` in place of `python`
  in every command below.
- Mac: use `python3` in place of `python` in every command below.

## Step 4. Install the libraries (once, needs internet, a few minutes)

    python -m pip install -r requirements.txt

Wait until the prompt comes back. Yellow warnings are fine. Red text ending
in "ERROR" is not; see Problems below.

## Step 5. Run the 5-minute test

    python run_all.py --quick

This runs everything on a small sample of patients to prove your setup
works. It is finished when you see:

    ALL DONE in ...
    The quick test worked. Now start the real run with:
        python run_all.py

The test's output goes into a `quick_test` folder. **Those numbers are not
results. Never use them in the thesis.**

## Step 6. The real run

Before you start: plug the laptop in, and set it not to go to sleep
(Windows: Settings > System > Power > "When plugged in, put my device to
sleep after: Never").

    python run_all.py

Lines keep appearing as each fold of each model finishes. Random Forest is
the slowest: its folds take a few minutes each, so a pause there is normal.

It is finished when you see `ALL DONE in ...`.

**If it stops for any reason** (laptop slept, you closed VS Code, power
cut): open the terminal and run `python run_all.py` again. Everything already
finished is kept and it continues from where it stopped.

## Step 7. What you get

| Where | What |
|---|---|
| `results/chapter_four_numbers.md` | Every number for Chapter Four, in tables. Open it in VS Code and press Ctrl+Shift+V to view it formatted |
| `figures/` | Eleven figures (`fig4_1` to `fig4_11`), ready to insert in Word |
| `models/` | The eight trained models, with thresholds and settings in `artefacts.json` |
| `results/*.csv` | The same numbers as spreadsheets; they open in Excel |

To see a trained model score a patient and explain itself:

    python src/predict.py
    python src/predict.py xgboost full

## What the run does, in order

1. Describes the data: cohort, missing values.
2. Cross-validation on Dataset A: 4 algorithms x 2 feature sets, 5 folds
   each, patients kept together. Also two sensitivity analyses.
3. Trains the 8 final models on all of Dataset A, tests them on Dataset B,
   compares each with SIRS, and bootstraps the confidence intervals.
4. Explains every model with exact Shapley values.
5. Draws the figures.
6. Writes all numbers into one document.

## Problems

**"python is not recognized"** (Windows): Python is not on PATH. Reinstall
Python and tick "Add python.exe to PATH", then restart VS Code.

**"No module named ..."**: Step 4 did not finish. Run it again.

**pip fails on `shap`** (mentions a compiler or "Microsoft Visual C++"):
your Python is newer than the library supports. Install Python 3.13 and
repeat from Step 3.

**Mac, an error mentioning `libomp` when XGBoost starts**: run
`brew install libomp`, then try again.

**"Data file not found"**: `setA.parquet` and `setB.parquet` must be inside
the `data` folder, with exactly those names.

**The run seems stuck**: check the time on the last line. Random Forest
folds take several minutes each. If nothing has appeared for 30 minutes,
press Ctrl+C in the terminal and run `python run_all.py` again; it resumes.

**Anything else**: copy the last 20 lines from the terminal and send them.
