# Chapter Four: every number

Four algorithms (Logistic Regression, Decision Tree, Random Forest, XGBoost), each trained on the restricted and on the full feature set, with SIRS as the rule-based comparator. Dataset A is used for development and internal validation, Dataset B for external validation.

## 1. Cohort (Section 4.1)

|  | Dataset A | Dataset B |
|---|---|---|
| Patients | 20,336 | 19,978 |
| Patient-hours | 790,215 | 761,246 |
| Patients who developed sepsis | 1,790 | 1,140 |
| Patients who developed sepsis (%) | 8.80 | 5.71 |
| Positive patient-hours | 17,136 | 10,763 |
| Positive patient-hours (%) | 2.17 | 1.41 |
| Class imbalance, 1 positive to n negative | 45.1 | 69.7 |
| Age, median | 64.7 | 62.0 |
| Age, interquartile range low | 52.3 | 50.0 |
| Age, interquartile range high | 75.7 | 72.0 |
| Male (%) | 58.2 | 53.7 |
| Record length in hours, median | 39 | 38 |
| Hours of record before first positive hour, median | 28.0 | 27.5 |
| Patients under 18 excluded | 0 | 22 |

## 2. Missingness (Section 4.2)

Eligible predictor variables: 36. Never recorded in Dataset A and therefore dropped: EtCO2. Variables in the full feature set: 35.

Restricted feature set:

| Variable | Carry limit (h) | Missing A (%) | After fill A (%) | Missing B (%) | After fill B (%) |
|---|---|---|---|---|---|
| Age | static | 0.0 | 0.0 | 0.0 | 0.0 |
| Gender | static | 0.0 | 0.0 | 0.0 | 0.0 |
| HR | 4 | 7.7 | 2.4 | 12.1 | 3.3 |
| Resp | 4 | 9.8 | 3.3 | 21.1 | 6.5 |
| O2Sat | 4 | 12.0 | 3.5 | 14.1 | 3.9 |
| SBP | 4 | 15.2 | 7.7 | 13.9 | 3.6 |
| DBP | 4 | 48.1 | 44.0 | 13.9 | 3.6 |
| Temp | 8 | 66.2 | 8.5 | 66.1 | 9.1 |

All other variables:

| Variable | Missing A (%) | After fill A (%) | Missing B (%) | After fill B (%) |
|---|---|---|---|---|
| MAP | 10.2 | 2.8 | 14.8 | 5.0 |
| FiO2 | 85.8 | 45.8 | 97.7 | 77.6 |
| Glucose | 87.8 | 14.4 | 77.8 | 15.8 |
| Hct | 88.2 | 13.6 | 94.2 | 27.9 |
| pH | 88.5 | 42.4 | 97.8 | 78.1 |
| Potassium | 89.1 | 15.3 | 92.3 | 22.9 |
| BaseExcess | 89.6 | 45.1 | 99.8 | 98.3 |
| Hgb | 91.2 | 16.9 | 94.1 | 27.7 |
| PaCO2 | 91.2 | 45.7 | 97.8 | 78.1 |
| Chloride | 91.7 | 17.2 | 99.4 | 94.0 |
| BUN | 91.8 | 16.6 | 94.5 | 26.5 |
| HCO3 | 91.9 | 17.2 | 99.8 | 98.7 |
| Magnesium | 92.2 | 23.8 | 95.2 | 36.1 |
| WBC | 92.5 | 19.0 | 94.7 | 28.8 |
| Creatinine | 93.4 | 18.3 | 94.5 | 26.4 |
| Platelets | 93.5 | 18.8 | 94.7 | 28.6 |
| Phosphate | 95.0 | 35.3 | 97.1 | 56.6 |
| Calcium | 95.0 | 36.3 | 93.2 | 24.0 |
| SaO2 | 95.0 | 69.7 | 98.1 | 81.1 |
| PTT | 95.2 | 40.3 | 99.0 | 85.1 |
| Lactate | 96.6 | 71.6 | 98.1 | 82.2 |
| AST | 98.5 | 80.7 | 98.2 | 71.5 |
| Alkalinephos | 98.5 | 81.2 | 98.2 | 71.4 |
| Bilirubin_total | 98.8 | 81.2 | 98.2 | 71.4 |
| Fibrinogen | 99.2 | 91.6 | 99.4 | 93.0 |
| Bilirubin_direct | 99.9 | 98.2 | 99.8 | 95.7 |
| TroponinI | 99.9 | 98.3 | 98.2 | 77.7 |
| EtCO2 | 100.0 | 100.0 | 92.4 | 85.6 |

Restricted predictors, median (interquartile range) of recorded values:

| Variable | Dataset A | Dataset B |
|---|---|---|
| HR | 84.0 (73.0 to 96.0) | 83.0 (71.0 to 95.0) |
| Resp | 18.0 (15.0 to 22.0) | 18.0 (16.0 to 21.0) |
| Temp | 37.1 (36.6 to 37.5) | 36.9 (36.5 to 37.4) |
| SBP | 118.5 (105.0 to 134.0) | 124.0 (109.0 to 142.0) |
| DBP | 58.5 (51.0 to 67.0) | 65.0 (56.0 to 74.0) |
| O2Sat | 98.0 (96.0 to 99.5) | 98.0 (96.0 to 99.5) |
| Age | 65.2 (52.7 to 75.9) | 63.0 (50.0 to 72.0) |
| Gender | 1.0 (0.0 to 1.0) | 1.0 (0.0 to 1.0) |

## 3. Internal validation: five-fold cross-validation on Dataset A (Section 4.3)

Mean across the five folds, with the standard deviation in brackets.

| Algorithm | Feature set | AUROC | AUPRC | Sensitivity | Specificity | Alert rate |
|---|---|---|---|---|---|---|
| Logistic Regression | restricted | 0.646 (0.007) | 0.042 (0.001) | 0.803 | 0.363 | 0.640 |
| Decision Tree | restricted | 0.645 (0.003) | 0.042 (0.002) | 0.829 | 0.321 | 0.682 |
| Random Forest | restricted | 0.676 (0.009) | 0.050 (0.003) | 0.819 | 0.386 | 0.619 |
| XGBoost | restricted | 0.675 (0.010) | 0.051 (0.003) | 0.760 | 0.467 | 0.538 |
| Logistic Regression | full | 0.703 (0.007) | 0.053 (0.004) | 0.807 | 0.451 | 0.555 |
| Decision Tree | full | 0.653 (0.008) | 0.044 (0.002) | 0.843 | 0.300 | 0.703 |
| Random Forest | full | 0.737 (0.008) | 0.061 (0.004) | 0.820 | 0.513 | 0.494 |
| XGBoost | full | 0.737 (0.011) | 0.064 (0.003) | 0.740 | 0.604 | 0.403 |

Hyperparameters selected in each fold, and those used for the final model (the combination selected most often):

| Algorithm | Feature set | Folds 1 to 5 | Final model |
|---|---|---|---|
| Logistic Regression | restricted | C=0.01; C=1; C=0.01; C=0.01; C=0.01 | C=0.01 |
| Decision Tree | restricted | max_depth=8, min_samples_leaf=50; max_depth=6, min_samples_leaf=200; max_depth=6, min_samples_leaf=200; max_depth=6, min_samples_leaf=50; max_depth=8, min_samples_leaf=50 | max_depth=6, min_samples_leaf=200 |
| Random Forest | restricted | max_depth=10, min_samples_leaf=50; max_depth=10, min_samples_leaf=50; max_depth=10, min_samples_leaf=50; max_depth=24, min_samples_leaf=50; max_depth=10, min_samples_leaf=50 | max_depth=10, min_samples_leaf=50 |
| XGBoost | restricted | max_depth=4, learning_rate=0.05; max_depth=4, learning_rate=0.05; max_depth=4, learning_rate=0.05; max_depth=4, learning_rate=0.05; max_depth=4, learning_rate=0.05 | max_depth=4, learning_rate=0.05 |
| Logistic Regression | full | C=1; C=1; C=0.01; C=0.1; C=0.01 | C=0.01 |
| Decision Tree | full | max_depth=8, min_samples_leaf=200; max_depth=6, min_samples_leaf=50; max_depth=6, min_samples_leaf=200; max_depth=6, min_samples_leaf=200; max_depth=6, min_samples_leaf=200 | max_depth=6, min_samples_leaf=200 |
| Random Forest | full | max_depth=10, min_samples_leaf=50; max_depth=10, min_samples_leaf=50; max_depth=24, min_samples_leaf=50; max_depth=16, min_samples_leaf=50; max_depth=10, min_samples_leaf=50 | max_depth=10, min_samples_leaf=50 |
| XGBoost | full | max_depth=4, learning_rate=0.05; max_depth=4, learning_rate=0.05; max_depth=4, learning_rate=0.05; max_depth=4, learning_rate=0.1; max_depth=6, learning_rate=0.05 | max_depth=4, learning_rate=0.05 |

## 4. Performance on Dataset A (out-of-fold) and Dataset B (external) (Sections 4.4 and 4.5)

Decision thresholds were fixed on Dataset A at 80% sensitivity before Dataset B was scored. Confidence intervals: 2,000 patient-level bootstrap resamples.

### Dataset A

| Model | Feature set | AUROC [95% CI] | AUPRC [95% CI] | Sensitivity | Specificity | PPV | Alert rate | False alerts per true alert |
|---|---|---|---|---|---|---|---|---|
| Logistic Regression | restricted | 0.645 [0.634, 0.656] | 0.042 [0.039, 0.045] | 0.800 | 0.368 | 2.7% | 63.6% | 36 |
| Decision Tree | restricted | 0.645 [0.635, 0.655] | 0.043 [0.040, 0.046] | 0.801 | 0.361 | 2.7% | 64.2% | 36 |
| Random Forest | restricted | 0.672 [0.661, 0.682] | 0.050 [0.046, 0.054] | 0.800 | 0.406 | 2.9% | 59.8% | 33 |
| XGBoost | restricted | 0.675 [0.664, 0.685] | 0.050 [0.047, 0.055] | 0.800 | 0.409 | 2.9% | 59.5% | 33 |
| Logistic Regression | full | 0.703 [0.693, 0.712] | 0.053 [0.050, 0.057] | 0.800 | 0.461 | 3.2% | 54.5% | 30 |
| Decision Tree | full | 0.654 [0.644, 0.664] | 0.044 [0.042, 0.048] | 0.813 | 0.353 | 2.7% | 65.0% | 36 |
| Random Forest | full | 0.718 [0.709, 0.728] | 0.058 [0.054, 0.063] | 0.800 | 0.501 | 3.4% | 50.6% | 28 |
| XGBoost | full | 0.734 [0.724, 0.744] | 0.063 [0.059, 0.069] | 0.800 | 0.530 | 3.6% | 47.7% | 27 |
| SIRS (score of 2 or more) | - | 0.619 [0.609, 0.629] | 0.032 [0.030, 0.033] | 0.524 | 0.664 | 3.3% | 34.0% | 29 |

### Dataset B

| Model | Feature set | AUROC [95% CI] | AUPRC [95% CI] | Sensitivity | Specificity | PPV | Alert rate | False alerts per true alert |
|---|---|---|---|---|---|---|---|---|
| Logistic Regression | restricted | 0.615 [0.600, 0.630] | 0.028 [0.025, 0.031] | 0.796 | 0.307 | 1.6% | 69.4% | 61 |
| Decision Tree | restricted | 0.634 [0.621, 0.646] | 0.030 [0.027, 0.033] | 0.796 | 0.338 | 1.7% | 66.4% | 58 |
| Random Forest | restricted | 0.663 [0.650, 0.676] | 0.035 [0.031, 0.039] | 0.868 | 0.254 | 1.6% | 74.7% | 60 |
| XGBoost | restricted | 0.664 [0.651, 0.677] | 0.034 [0.031, 0.038] | 0.813 | 0.357 | 1.8% | 64.6% | 55 |
| Logistic Regression | full | 0.669 [0.655, 0.683] | 0.035 [0.032, 0.039] | 0.722 | 0.500 | 2.0% | 50.3% | 48 |
| Decision Tree | full | 0.678 [0.665, 0.691] | 0.039 [0.035, 0.043] | 0.896 | 0.193 | 1.6% | 80.8% | 63 |
| Random Forest | full | 0.741 [0.729, 0.753] | 0.048 [0.043, 0.054] | 0.832 | 0.472 | 2.2% | 53.2% | 44 |
| XGBoost | full | 0.669 [0.655, 0.685] | 0.040 [0.036, 0.046] | 0.615 | 0.645 | 2.4% | 35.9% | 40 |
| SIRS (score of 2 or more) | - | 0.660 [0.647, 0.672] | 0.027 [0.025, 0.030] | 0.494 | 0.749 | 2.7% | 25.5% | 35 |

SIRS could be scored on all four of its components on 76.4% of hours in Dataset A and 67.4% in Dataset B. A component with no available value was counted as not met.

Change from Dataset A to Dataset B:

| Algorithm | Feature set | AUROC A | AUROC B | Change | Calibration slope A | Calibration slope B |
|---|---|---|---|---|---|---|
| Logistic Regression | restricted | 0.645 | 0.615 | -0.030 | 0.99 | 0.94 |
| Decision Tree | restricted | 0.645 | 0.634 | -0.012 | 0.59 | 1.08 |
| Random Forest | restricted | 0.672 | 0.663 | -0.009 | 1.15 | 1.54 |
| XGBoost | restricted | 0.675 | 0.664 | -0.011 | 0.92 | 1.08 |
| Logistic Regression | full | 0.703 | 0.669 | -0.034 | 0.87 | 0.70 |
| Decision Tree | full | 0.654 | 0.678 | +0.024 | 0.54 | 0.74 |
| Random Forest | full | 0.718 | 0.741 | +0.023 | 1.50 | 2.25 |
| XGBoost | full | 0.734 | 0.669 | -0.065 | 0.92 | 0.76 |

## 5. Restricted against full feature set: the pre-registered criterion (Section 4.6)

Criterion fixed in Section 3.12: the restriction is acceptable if the lower bound of the 95% confidence interval for the AUROC difference (restricted minus full) is above -0.05.

| Dataset | Algorithm | AUROC difference [95% CI] | Lower bound | Criterion | AUPRC difference [95% CI] |
|---|---|---|---|---|---|
| A | Logistic Regression | -0.0575 [-0.0651, -0.0495] | -0.0651 | **not met** | -0.0110 [-0.0135, -0.0085] |
| A | Decision Tree | -0.0090 [-0.0188, +0.0015] | -0.0188 | met | -0.0019 [-0.0042, +0.0005] |
| A | XGBoost | -0.0598 [-0.0694, -0.0504] | -0.0694 | **not met** | -0.0132 [-0.0169, -0.0098] |
| A | Random Forest | -0.0463 [-0.0556, -0.0368] | -0.0556 | **not met** | -0.0080 [-0.0115, -0.0045] |
| B | Logistic Regression | -0.0539 [-0.0664, -0.0422] | -0.0664 | **not met** | -0.0076 [-0.0105, -0.0049] |
| B | Decision Tree | -0.0448 [-0.0580, -0.0318] | -0.0580 | **not met** | -0.0087 [-0.0121, -0.0057] |
| B | XGBoost | -0.0056 [-0.0221, +0.0105] | -0.0221 | met | -0.0064 [-0.0103, -0.0026] |
| B | Random Forest | -0.0784 [-0.0896, -0.0671] | -0.0896 | **not met** | -0.0134 [-0.0171, -0.0100] |

The criterion was met in 2 of 8 algorithm and dataset combinations.

## 6. Every model against SIRS (Section 4.7)

| Dataset | Model | AUROC difference against SIRS [95% CI] | Interval excludes zero |
|---|---|---|---|
| A | Logistic Regression, restricted | +0.0265 [+0.0162, +0.0362] | yes |
| A | Decision Tree, restricted | +0.0265 [+0.0172, +0.0354] | yes |
| A | XGBoost, restricted | +0.0557 [+0.0451, +0.0658] | yes |
| A | Random Forest, restricted | +0.0531 [+0.0434, +0.0627] | yes |
| A | Logistic Regression, full | +0.0839 [+0.0731, +0.0940] | yes |
| A | Decision Tree, full | +0.0354 [+0.0244, +0.0465] | yes |
| A | XGBoost, full | +0.1155 [+0.1043, +0.1267] | yes |
| A | Random Forest, full | +0.0995 [+0.0889, +0.1098] | yes |
| B | Logistic Regression, restricted | -0.0446 [-0.0577, -0.0314] | yes |
| B | Decision Tree, restricted | -0.0262 [-0.0383, -0.0139] | yes |
| B | XGBoost, restricted | +0.0039 [-0.0082, +0.0157] | **no** |
| B | Random Forest, restricted | +0.0031 [-0.0082, +0.0145] | **no** |
| B | Logistic Regression, full | +0.0093 [-0.0044, +0.0228] | **no** |
| B | Decision Tree, full | +0.0186 [+0.0053, +0.0321] | yes |
| B | XGBoost, full | +0.0095 [-0.0070, +0.0255] | **no** |
| B | Random Forest, full | +0.0815 [+0.0701, +0.0928] | yes |

## 7. Calibration and clinical utility (Section 4.8)

Class weighting inflates the raw scores, so they are not risks. The recalibration is a logistic model fitted on Dataset A only and applied unchanged to Dataset B. It does not change AUROC or AUPRC.

| Model | Dataset | Mean score as fitted | Mean recalibrated risk | Observed rate | Brier as fitted | Brier recalibrated | Slope after recalibration |
|---|---|---|---|---|---|---|---|
| Logistic Regression, restricted | A | 0.4674 | 0.0217 | 0.0217 | 0.2318 | 0.0211 | 0.94 |
| Logistic Regression, restricted | B | 0.4751 | 0.0217 | 0.0141 | 0.2356 | 0.0140 | 1.04 |
| Decision Tree, restricted | A | 0.4489 | 0.0217 | 0.0217 | 0.2215 | 0.0211 | 0.94 |
| Decision Tree, restricted | B | 0.4617 | 0.0216 | 0.0141 | 0.2225 | 0.0139 | 1.84 |
| XGBoost, restricted | A | 0.4277 | 0.0216 | 0.0217 | 0.2045 | 0.0210 | 1.01 |
| XGBoost, restricted | B | 0.4311 | 0.0207 | 0.0141 | 0.2010 | 0.0139 | 1.19 |
| Random Forest, restricted | A | 0.4230 | 0.0217 | 0.0217 | 0.1946 | 0.0210 | 1.00 |
| Random Forest, restricted | B | 0.4494 | 0.0226 | 0.0141 | 0.2098 | 0.0139 | 1.33 |
| Logistic Regression, full | A | 0.4372 | 0.0217 | 0.0217 | 0.2151 | 0.0210 | 1.00 |
| Logistic Regression, full | B | 0.4165 | 0.0194 | 0.0141 | 0.1960 | 0.0139 | 0.81 |
| Decision Tree, full | A | 0.4305 | 0.0217 | 0.0217 | 0.2141 | 0.0210 | 0.94 |
| Decision Tree, full | B | 0.4055 | 0.0200 | 0.0141 | 0.1824 | 0.0138 | 1.41 |
| XGBoost, full | A | 0.3521 | 0.0217 | 0.0217 | 0.1574 | 0.0208 | 1.00 |
| XGBoost, full | B | 0.2886 | 0.0152 | 0.0141 | 0.1110 | 0.0138 | 0.82 |
| Random Forest, full | A | 0.3590 | 0.0217 | 0.0217 | 0.1466 | 0.0209 | 1.00 |
| Random Forest, full | B | 0.3787 | 0.0218 | 0.0141 | 0.1525 | 0.0138 | 1.49 |

Net benefit on recalibrated risks (treat none = 0). A model is useful at a threshold only if its net benefit is above both treat all and zero.

Dataset A:

| Model | Feature set | at 1% | at 2% | at 5% | at 10% |
|---|---|---|---|---|---|
| Logistic Regression | restricted | 0.01184 | 0.00517 | 0.00072 | -0.00006 |
| Decision Tree | restricted | 0.01177 | 0.00501 | 0.00067 | -0.00001 |
| Random Forest | restricted | 0.01212 | 0.00590 | 0.00118 | 0.00009 |
| XGBoost | restricted | 0.01213 | 0.00603 | 0.00125 | 0.00009 |
| Logistic Regression | full | 0.01237 | 0.00715 | 0.00152 | 0.00000 |
| Decision Tree | full | 0.01162 | 0.00557 | 0.00052 | 0.00002 |
| Random Forest | full | 0.01285 | 0.00770 | 0.00147 | 0.00020 |
| XGBoost | full | 0.01301 | 0.00808 | 0.00220 | 0.00027 |
| Treat all | - | 0.01180 | 0.00172 | -0.02981 | -0.08702 |

Dataset B:

| Model | Feature set | at 1% | at 2% | at 5% | at 10% |
|---|---|---|---|---|---|
| Logistic Regression | restricted | 0.00408 | -0.00047 | 0.00012 | -0.00003 |
| Decision Tree | restricted | 0.00414 | -0.00081 | 0.00028 | 0.00000 |
| Random Forest | restricted | 0.00440 | -0.00009 | 0.00027 | 0.00004 |
| XGBoost | restricted | 0.00453 | 0.00093 | 0.00026 | 0.00002 |
| Logistic Regression | full | 0.00461 | 0.00184 | 0.00034 | -0.00007 |
| Decision Tree | full | 0.00406 | 0.00230 | 0.00044 | 0.00000 |
| Random Forest | full | 0.00503 | 0.00232 | 0.00084 | 0.00002 |
| XGBoost | full | 0.00473 | 0.00243 | 0.00067 | 0.00006 |
| Treat all | - | 0.00418 | -0.00598 | -0.03775 | -0.09540 |

## 8. Alert behaviour on Dataset A (Section 4.9)

The first recorded hour is the hour at which the model knows least about a patient. 'Never septic' means hours belonging to patients who did not meet the sepsis definition at any point.

| Model | Threshold | Alert rate, first recorded hour | Alert rate, later hours | Alert rate, never-septic patients | Highest score given | That hour was sepsis |
|---|---|---|---|---|---|---|
| Logistic Regression, restricted | 0.4195 | 60.1% | 63.7% | 61.4% | 0.9614 | no |
| Decision Tree, restricted | 0.3883 | 82.0% | 63.7% | 62.2% | 0.9541 | no |
| XGBoost, restricted | 0.3778 | 75.5% | 59.1% | 56.8% | 0.9324 | no |
| Random Forest, restricted | 0.3897 | 74.6% | 59.4% | 57.2% | 0.8849 | no |
| Logistic Regression, full | 0.3977 | 23.8% | 55.3% | 50.9% | 0.9975 | no |
| Decision Tree, full | 0.3474 | 48.6% | 65.5% | 62.8% | 0.9539 | no |
| XGBoost, full | 0.3383 | 75.5% | 47.0% | 43.4% | 0.9601 | no |
| Random Forest, full | 0.3538 | 45.4% | 50.7% | 46.8% | 0.8119 | yes |

## 9. Sensitivity analyses on the restricted feature set (Section 4.10)

Each row removes one preprocessing decision and keeps everything else the same. Mean cross-validated AUROC on Dataset A.

| Algorithm | As specified | Median imputation only | Change | No availability indicators | Change |
|---|---|---|---|---|---|
| Logistic Regression | 0.6457 | 0.6338 | -0.0119 | 0.6314 | -0.0143 |
| Decision Tree | 0.6451 | 0.6184 | -0.0267 | 0.6340 | -0.0111 |
| Random Forest | 0.6756 | 0.6529 | -0.0227 | 0.6582 | -0.0174 |
| XGBoost | 0.6750 | 0.6486 | -0.0264 | 0.6605 | -0.0145 |

## 10. Explanations (Section 4.11)

Exact Shapley values. Global importance is the mean absolute value over 5,000 randomly sampled Dataset A patient-hours (random state 42). Values are on the log-odds scale for Logistic Regression and XGBoost and on the probability scale for Decision Tree and Random Forest, so compare rankings between algorithms, not magnitudes.

Ten most important features, restricted feature set:

| Rank | Logistic Regression | Decision Tree | Random Forest | XGBoost |
|---|---|---|---|---|
| 1 | DBP_obs (0.2188) | Temp (0.0629) | Temp (0.0483) | Temp (0.2740) |
| 2 | Resp (0.2070) | Resp (0.0441) | Resp (0.0423) | Resp (0.2172) |
| 3 | SBP_obs (0.1747) | HR (0.0362) | HR (0.0372) | HR (0.1667) |
| 4 | HR (0.1493) | DBP_obs (0.0269) | DBP_obs (0.0284) | DBP_obs (0.1662) |
| 5 | Temp (0.1383) | SBP (0.0205) | SBP (0.0192) | SBP (0.1471) |
| 6 | Temp_obs (0.0890) | Temp_obs (0.0085) | Temp_obs (0.0108) | Age (0.1160) |
| 7 | SBP (0.0718) | Age (0.0048) | Age (0.0097) | Temp_obs (0.0839) |
| 8 | Gender (0.0573) | SBP_obs (0.0030) | SBP_obs (0.0094) | O2Sat (0.0784) |
| 9 | O2Sat_obs (0.0495) | O2Sat (0.0013) | O2Sat (0.0071) | SBP_obs (0.0557) |
| 10 | Resp_obs (0.0414) | Gender (0.0006) | DBP (0.0069) | Gender (0.0466) |

Ten most important features, full feature set:

| Rank | Logistic Regression | Decision Tree | Random Forest | XGBoost |
|---|---|---|---|---|
| 1 | SBP_obs (0.2074) | BUN (0.0676) | Temp (0.0334) | Temp (0.2507) |
| 2 | Resp (0.1824) | Temp (0.0582) | BUN (0.0296) | BUN (0.2082) |
| 3 | DBP_obs (0.1809) | Resp (0.0311) | HR (0.0256) | Resp (0.1606) |
| 4 | Temp (0.1715) | HR (0.0246) | Resp (0.0245) | Hgb (0.1287) |
| 5 | BaseExcess_obs (0.1665) | Hgb (0.0208) | FiO2_obs (0.0180) | Phosphate (0.1283) |
| 6 | Hgb (0.1663) | FiO2_obs (0.0199) | Phosphate (0.0133) | HR (0.1235) |
| 7 | BUN (0.1616) | WBC (0.0124) | Creatinine (0.0131) | FiO2_obs (0.1010) |
| 8 | HR (0.1402) | pH (0.0118) | Calcium (0.0124) | Potassium (0.0985) |
| 9 | FiO2_obs (0.1303) | Creatinine (0.0074) | Hgb (0.0115) | WBC (0.0901) |
| 10 | pH_obs (0.1262) | Phosphate (0.0071) | BaseExcess (0.0113) | FiO2 (0.0850) |

Consistency of explanations: Spearman rank correlation of global importance between the five cross-validation fold models (2,000 held-out hours per fold).

| Model | Median | Lowest pair | Highest pair |
|---|---|---|---|
| Logistic Regression, restricted | 0.936 | 0.890 | 0.978 |
| Decision Tree, restricted | 0.969 | 0.956 | 0.996 |
| XGBoost, restricted | 0.976 | 0.965 | 0.991 |
| Random Forest, restricted | 0.960 | 0.877 | 0.987 |
| Logistic Regression, full | 0.925 | 0.911 | 0.955 |
| Decision Tree, full | 0.770 | 0.667 | 0.861 |
| XGBoost, full | 0.957 | 0.946 | 0.964 |
| Random Forest, full | 0.982 | 0.969 | 0.990 |

Individual explanations, Random Forest, restricted. Baseline plus the sum of all contributions equals the score.

Confident true positive: sepsis label 1, score 0.8795, threshold 0.3897, baseline 0.5000, alerted.

| Feature | Value | Contribution |
|---|---|---|
| Temp | 38.78 | +0.1641 |
| Resp | 32.5 | +0.0939 |
| HR | 112 | +0.0617 |
| DBP_obs | 1 | +0.0186 |
| O2Sat | 93.5 | +0.0105 |
| DBP | 77.5 | +0.0095 |
| Gender | 1 | +0.0092 |
| Temp_obs | 0 | +0.0078 |

Confident false positive: sepsis label 0, score 0.8628, threshold 0.3897, baseline 0.5000, alerted.

| Feature | Value | Contribution |
|---|---|---|
| Temp | 38.72 | +0.1327 |
| Resp | 32 | +0.0954 |
| HR | 132 | +0.0652 |
| SBP_obs | 0 | +0.0330 |
| SBP | 119 | +0.0277 |
| Temp_obs | 1 | -0.0061 |
| Age | 74.25 | +0.0051 |
| Gender | 1 | +0.0045 |

Default false positive: sepsis label 0, score 0.5000, threshold 0.3897, baseline 0.5000, alerted.

| Feature | Value | Contribution |
|---|---|---|
| Temp | 36.06 | -0.0436 |
| DBP_obs | 1 | +0.0284 |
| HR | 96 | +0.0141 |
| DBP | 56 | -0.0122 |
| Temp_obs | 0 | +0.0105 |
| O2Sat | 100 | +0.0056 |
| SBP | 105 | -0.0041 |
| Gender | 1 | +0.0025 |

Correct rejection: sepsis label 0, score 0.1157, threshold 0.3897, baseline 0.5000, no alert.

| Feature | Value | Contribution |
|---|---|---|
| HR | 52 | -0.0668 |
| DBP_obs | 0 | -0.0643 |
| Resp | 13 | -0.0535 |
| Temp | 36.22 | -0.0472 |
| Age | 41.82 | -0.0470 |
| SBP | 175 | -0.0465 |
| O2Sat_obs | 0 | -0.0263 |
| DBP | 59 | -0.0157 |

## 11. Where each number comes from

| File in the results folder | Contents |
|---|---|
| cohort.csv, missingness.csv, descriptives_restricted.csv | Sections 1 and 2 |
| cv_*.csv, hyperparameters_*.csv | Section 3, one file per model |
| performance.csv, confidence_intervals.csv, sirs.csv | Section 4 |
| restricted_vs_full.csv | Section 5 |
| model_vs_sirs.csv | Section 6 |
| calibration_summary.csv, calibration_*.csv, dca_*.csv | Section 7 |
| alert_behaviour.csv | Section 8 |
| shap_global_*.csv, shap_consistency.csv, shap_local_cases.csv | Section 10 |
| bootstrap_raw_A.npz, bootstrap_raw_B.npz | every bootstrap resample |
| ../models/artefacts.json | columns, medians, thresholds, hyperparameters |

