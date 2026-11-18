# Controlled v2: baseline findings

Completed main runs: **40/40**; controls: **20/20**. Development data only. Means +/- sample SD across training seeds; no population-sampling uncertainty is estimated.

## What DP changes for different groups

The main comparison is **DP-SGD versus non-private SGD on the same feature policy, split, and initialization seed**. The primary setting is `include_sensitive` (sex and race are available as input features). The secondary setting repeats the comparison without those two predictors. Every reported group is still evaluated under both settings.

Read accuracy and disparity together: two groups can both lose accuracy while their gap shrinks. A smaller fairness gap alone does not mean either group became better off. All results below use the fixed development set and five training seeds.

### Which groups lose accuracy?

Accuracy is percent mean +/- sample SD. Loss is **SGD accuracy minus DP accuracy**, in percentage points (pp), paired by seed before averaging. Positive loss means worse accuracy under DP. The three loss columns cover every tested privacy budget.

#### include_sensitive

| Group | People | SGD accuracy (%) | DP1 accuracy (%) | Loss eps1 (pp) | Loss eps4 (pp) | Loss eps8 (pp) |
| --- | --- | --- | --- | --- | --- | --- |
| Male | 14826 | 80.92 +/- 0.44 | 78.67 +/- 0.59 | +2.26 +/- 0.85 | +2.19 +/- 0.87 | +2.36 +/- 0.86 |
| Female | 15438 | 74.06 +/- 0.68 | 72.49 +/- 0.12 | +1.57 +/- 0.73 | +1.57 +/- 0.70 | +1.59 +/- 0.72 |
| White alone | 18889 | 77.61 +/- 0.59 | 75.98 +/- 0.31 | +1.63 +/- 0.72 | +1.58 +/- 0.72 | +1.69 +/- 0.68 |
| Black or African American alone | 1500 | 79.15 +/- 1.07 | 78.04 +/- 0.47 | +1.11 +/- 1.19 | +1.25 +/- 1.21 | +1.41 +/- 1.28 |
| American Indian alone | 202 | 67.43 +/- 0.89 | 66.04 +/- 0.75 | +1.39 +/- 1.18 | +1.49 +/- 1.53 | +1.29 +/- 1.47 |
| Alaska Native alone | 2 | 100.00 +/- 0.00 | 100.00 +/- 0.00 | +0.00 +/- 0.00 | +0.00 +/- 0.00 | +0.00 +/- 0.00 |
| AIAN specified/unspecified; no other races | 81 | 80.00 +/- 1.03 | 77.28 +/- 1.10 | +2.72 +/- 1.03 | +2.47 +/- 0.87 | +2.96 +/- 0.68 |
| Asian alone | 5017 | 76.80 +/- 0.67 | 74.93 +/- 0.34 | +1.87 +/- 0.86 | +1.84 +/- 0.94 | +1.90 +/- 0.90 |
| Native Hawaiian / Other Pacific Islander alone | 95 | 83.79 +/- 0.94 | 82.53 +/- 0.58 | +1.26 +/- 0.88 | +1.47 +/- 0.94 | +1.26 +/- 0.88 |
| Some Other Race alone | 3299 | 77.14 +/- 0.22 | 73.13 +/- 0.56 | +4.01 +/- 0.74 | +4.01 +/- 0.63 | +4.12 +/- 0.70 |
| Two or More Races | 1179 | 76.61 +/- 1.11 | 74.93 +/- 0.52 | +1.68 +/- 1.17 | +1.58 +/- 1.09 | +1.61 +/- 1.33 |


At eps1, male accuracy loses 2.26 pp and female accuracy loses 1.57 pp. Among race groups with at least 500 people, Some Other Race alone has the largest mean loss (4.01 pp). These are observed training-seed averages, not a claim that a particular group always loses most.

#### exclude_sensitive

| Group | People | SGD accuracy (%) | DP1 accuracy (%) | Loss eps1 (pp) | Loss eps4 (pp) | Loss eps8 (pp) |
| --- | --- | --- | --- | --- | --- | --- |
| Male | 14826 | 80.48 +/- 0.29 | 79.44 +/- 0.13 | +1.04 +/- 0.25 | +1.03 +/- 0.23 | +1.01 +/- 0.21 |
| Female | 15438 | 73.63 +/- 0.14 | 72.18 +/- 0.08 | +1.45 +/- 0.15 | +1.47 +/- 0.20 | +1.44 +/- 0.16 |
| White alone | 18889 | 77.25 +/- 0.10 | 76.04 +/- 0.11 | +1.20 +/- 0.05 | +1.22 +/- 0.08 | +1.20 +/- 0.04 |
| Black or African American alone | 1500 | 79.53 +/- 0.26 | 78.51 +/- 0.20 | +1.03 +/- 0.42 | +0.85 +/- 0.34 | +0.96 +/- 0.46 |
| American Indian alone | 202 | 69.90 +/- 1.01 | 66.73 +/- 0.65 | +3.17 +/- 1.47 | +2.97 +/- 1.53 | +2.87 +/- 1.23 |
| Alaska Native alone | 2 | 100.00 +/- 0.00 | 100.00 +/- 0.00 | +0.00 +/- 0.00 | +0.00 +/- 0.00 | +0.00 +/- 0.00 |
| AIAN specified/unspecified; no other races | 81 | 80.49 +/- 0.55 | 79.75 +/- 1.10 | +0.74 +/- 1.41 | -0.25 +/- 1.03 | +0.25 +/- 0.55 |
| Asian alone | 5017 | 76.39 +/- 0.25 | 75.08 +/- 0.07 | +1.31 +/- 0.25 | +1.32 +/- 0.24 | +1.32 +/- 0.25 |
| Native Hawaiian / Other Pacific Islander alone | 95 | 82.95 +/- 2.02 | 82.53 +/- 0.94 | +0.42 +/- 2.18 | +0.42 +/- 2.18 | +0.84 +/- 2.40 |
| Some Other Race alone | 3299 | 75.28 +/- 0.43 | 73.92 +/- 0.27 | +1.36 +/- 0.55 | +1.41 +/- 0.28 | +1.29 +/- 0.49 |
| Two or More Races | 1179 | 77.29 +/- 0.56 | 75.76 +/- 0.22 | +1.53 +/- 0.51 | +1.46 +/- 0.40 | +1.41 +/- 0.56 |


At eps1, male accuracy loses 1.04 pp and female accuracy loses 1.45 pp. Among race groups with at least 500 people, Two or More Races has the largest mean loss (1.53 pp). These are observed training-seed averages, not a claim that a particular group always loses most.

Race categories are original ACS categories; none were merged. Small groups remain visible. In particular, the two-person Alaska Native-alone group has no actual positives, so its 100% accuracy does not establish good predictive performance.

### What each fairness gap asks

| Metric | Rates that should match | Reported gap |
| --- | --- | --- |
| Accuracy parity | Accuracy across groups | max accuracy - min accuracy |
| Independence | Predicted-positive / selection rate | max selection rate - min selection rate |
| Separation (equalized odds) | TPR and FPR separately | larger of TPR range and FPR range |
| Binary sufficiency | PPV and NPV separately | larger of PPV range and NPV range |


For every gap, **0 means parity and smaller means closer to that fairness criterion**. This binary sufficiency diagnostic concerns thresholded predictions, not full score calibration. Gaps are computed within each seed before averaging. They need not equal the gap between averaged rates.

### Did DP improve or worsen these gaps?

Each level is mean +/- sample SD in pp. The signed change is DP1 minus matched SGD: **positive means a larger gap; negative means a smaller gap**. Sex uses both groups. Headline race uses only the fixed five groups with development n>=500.

#### include_sensitive

| Comparison | Metric | SGD | DP eps1 | DP eps4 | DP eps8 | DP1 change (pp) | Smaller / larger gap seeds |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Sex | Accuracy parity | 6.86 +/- 0.31 | 6.17 +/- 0.48 | 6.25 +/- 0.45 | 6.09 +/- 0.52 | -0.69 +/- 0.68 | 4 / 1 of 5 |
| Sex | Independence | 7.82 +/- 2.34 | 6.45 +/- 1.26 | 6.41 +/- 1.12 | 6.57 +/- 1.11 | -1.37 +/- 2.33 | 3 / 2 of 5 |
| Sex | Separation | 7.75 +/- 3.00 | 5.03 +/- 0.71 | 5.04 +/- 0.64 | 5.01 +/- 0.58 | -2.72 +/- 2.98 | 5 / 0 of 5 |
| Sex | Binary sufficiency | 9.29 +/- 0.46 | 8.72 +/- 0.76 | 8.80 +/- 0.69 | 8.61 +/- 0.75 | -0.57 +/- 0.60 | 4 / 1 of 5 |
| Race n>=500 | Accuracy parity | 2.85 +/- 0.54 | 4.91 +/- 0.48 | 4.76 +/- 0.29 | 4.71 +/- 0.43 | +2.06 +/- 0.53 | 0 / 5 of 5 |
| Race n>=500 | Independence | 17.83 +/- 0.99 | 25.26 +/- 1.07 | 25.00 +/- 0.92 | 25.09 +/- 1.11 | +7.42 +/- 1.36 | 0 / 5 of 5 |
| Race n>=500 | Separation | 16.72 +/- 2.77 | 31.24 +/- 1.44 | 30.79 +/- 1.16 | 30.96 +/- 1.44 | +14.52 +/- 3.18 | 0 / 5 of 5 |
| Race n>=500 | Binary sufficiency | 4.64 +/- 0.95 | 4.28 +/- 0.43 | 4.14 +/- 0.25 | 4.21 +/- 0.21 | -0.36 +/- 0.59 | 4 / 1 of 5 |


#### exclude_sensitive

| Comparison | Metric | SGD | DP eps1 | DP eps4 | DP eps8 | DP1 change (pp) | Smaller / larger gap seeds |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Sex | Accuracy parity | 6.85 +/- 0.39 | 7.27 +/- 0.09 | 7.30 +/- 0.06 | 7.29 +/- 0.07 | +0.41 +/- 0.38 | 1 / 4 of 5 |
| Sex | Independence | 0.71 +/- 0.15 | 0.82 +/- 0.16 | 0.80 +/- 0.20 | 0.82 +/- 0.16 | +0.11 +/- 0.28 | 2 / 3 of 5 |
| Sex | Separation | 12.09 +/- 0.21 | 11.17 +/- 0.35 | 11.18 +/- 0.30 | 11.22 +/- 0.31 | -0.91 +/- 0.53 | 5 / 0 of 5 |
| Sex | Binary sufficiency | 12.60 +/- 0.20 | 11.98 +/- 0.16 | 11.99 +/- 0.10 | 12.01 +/- 0.13 | -0.61 +/- 0.32 | 5 / 0 of 5 |
| Race n>=500 | Accuracy parity | 4.25 +/- 0.59 | 4.59 +/- 0.38 | 4.80 +/- 0.20 | 4.58 +/- 0.38 | +0.34 +/- 0.87 | 2 / 3 of 5 |
| Race n>=500 | Independence | 17.35 +/- 1.17 | 19.74 +/- 0.44 | 19.77 +/- 0.27 | 19.65 +/- 0.34 | +2.39 +/- 1.01 | 0 / 5 of 5 |
| Race n>=500 | Separation | 19.86 +/- 1.65 | 24.81 +/- 0.83 | 25.07 +/- 0.31 | 24.68 +/- 0.71 | +4.95 +/- 1.56 | 0 / 5 of 5 |
| Race n>=500 | Binary sufficiency | 6.93 +/- 1.50 | 5.98 +/- 0.25 | 6.24 +/- 0.19 | 6.15 +/- 0.23 | -0.95 +/- 1.44 | 4 / 1 of 5 |


- include_sensitive, sex, DP1 versus SGD: smaller mean gaps: accuracy parity, independence, separation, binary sufficiency.

- include_sensitive, headline race, DP1 versus SGD: smaller mean gaps: binary sufficiency; larger mean gaps: accuracy parity, independence, separation.

- exclude_sensitive, sex, DP1 versus SGD: smaller mean gaps: separation, binary sufficiency; larger mean gaps: accuracy parity, independence.

- exclude_sensitive, headline race, DP1 versus SGD: smaller mean gaps: binary sufficiency; larger mean gaps: accuracy parity, independence, separation.

The answer therefore depends on the fairness definition and group comparison.

### Which actual rates drive the change? Primary setting, SGD -> DP1

Entries show mean rates in percent. Selection is the fraction predicted positive. TPR is the fraction of actual positives correctly predicted positive. FPR is the fraction of actual negatives incorrectly predicted positive. PPV is the fraction of predicted positives that are actually positive. NPV is the fraction of predicted negatives that are actually negative. For predictive performance, higher TPR/PPV/NPV and lower FPR are desirable; for the corresponding fairness criterion, groups should have matching rates.

| Group | Selection | TPR | FPR | PPV | NPV |
| --- | --- | --- | --- | --- | --- |
| Male | 64.24 -> 72.25 | 86.69 -> 91.37 | 28.31 -> 41.66 | 83.14 -> 77.83 | 77.41 -> 80.85 |
| Female | 56.42 -> 65.80 | 78.94 -> 86.36 | 31.37 -> 42.93 | 73.84 -> 69.11 | 75.02 -> 79.02 |
| White alone | 59.02 -> 66.83 | 82.31 -> 87.76 | 28.54 -> 39.44 | 79.20 -> 74.45 | 75.87 -> 79.07 |
| Black or African American alone | 48.39 -> 54.19 | 78.10 -> 82.88 | 19.84 -> 26.61 | 79.22 -> 74.95 | 79.46 -> 81.71 |
| Asian alone | 64.01 -> 73.63 | 84.71 -> 91.29 | 34.48 -> 48.43 | 77.92 -> 72.90 | 75.50 -> 80.59 |
| Some Other Race alone | 66.17 -> 79.44 | 86.19 -> 93.93 | 36.33 -> 57.86 | 78.09 -> 70.75 | 76.04 -> 82.33 |
| Two or More Races | 64.99 -> 73.99 | 85.64 -> 91.92 | 36.05 -> 48.88 | 77.03 -> 72.49 | 76.91 -> 81.87 |


- Male versus female: the absolute difference between mean TPR rates changes from 7.75 to 5.00 pp.

- Male versus female: the absolute difference between mean FPR rates changes from 3.06 to 1.27 pp.

- White-alone versus Black-alone: the absolute difference between mean TPR rates changes from 4.21 to 4.87 pp.

- White-alone versus Black-alone: the absolute difference between mean FPR rates changes from 8.69 to 12.82 pp.

Both sex groups have higher TPR **and higher FPR** under DP1. The smaller sex separation gap therefore coexists with more false positives and lower accuracy. White-alone and Black-alone are one concrete pair, not a reference population or the entire multi-race comparison. The headline race separation range covers all five supported groups.

**Full group tables:** [GROUP_COMPARISONS.md](GROUP_COMPARISONS.md) includes both policies, all three budgets, all nine race groups and all 18 race-by-sex intersections, with denominator support, rate levels, and paired changes.

## What persists across the 5 seeds

- **include_sensitive:** adding clipping to matched non-private Poisson SGD changes mean accuracy by -2.055 pp; DP1 minus the clipped/no-noise control is +0.005 pp. These contrasts identify where the measured utility changes occur.

- **include_sensitive, DP1 versus SGD:** sex separation improves in 5/5 seeds; headline race separation worsens in 5/5. A single statement that DP improved or worsened every notion/group would be inaccurate.

- **exclude_sensitive:** adding clipping to matched non-private Poisson SGD changes mean accuracy by -1.220 pp; DP1 minus the clipped/no-noise control is +0.021 pp. These contrasts identify where the measured utility changes occur.

- **exclude_sensitive, DP1 versus SGD:** sex separation improves in 5/5 seeds; headline race separation worsens in 5/5. A single statement that DP improved or worsened every notion/group would be inaccurate.

- **include_sensitive, FPR and budget persistence:** female FPR is higher in 3/5 DP1 seeds. Epsilon8 accuracy is higher than epsilon1 in 0/5 seeds and lower in 5/5. Mean change is -0.066 pp. Use these observed signs rather than dismissing every budget difference as random noise.

- **exclude_sensitive, FPR and budget persistence:** female FPR is higher in 5/5 DP1 seeds. Epsilon8 accuracy is higher than epsilon1 in 3/5 seeds and lower in 2/5. Mean change is +0.020 pp. Use these observed signs rather than dismissing every budget difference as random noise.

Ten-seed follow-up is recommended before final generalizations. Headline race gaps use the fixed n>=500 development groups (White-alone, Black-alone, Asian-alone, Some Other Race-alone, Two or More Races). All nine races/intersections remain in detailed CSVs.

## Overall accuracy and selected fairness outcomes

### include_sensitive

| Method | Accuracy (%) | Balanced accuracy (%) | ROC-AUC | BCE | Sex separation (pp) | Race n>=500 separation (pp) |
| --- | --- | --- | --- | --- | --- | --- |
| sgd | 77.423 +/- 0.551 (n=5) | 76.507 +/- 0.258 (n=5) | 0.8463 +/- 0.0005 (n=5) | 0.4796 +/- 0.0078 (n=5) | 7.753 +/- 3.001 (n=5) | 16.723 +/- 2.769 (n=5) |
| dp_epsilon_1 | 75.517 +/- 0.343 (n=5) | 73.319 +/- 0.358 (n=5) | 0.8178 +/- 0.0060 (n=5) | 0.7549 +/- 0.0198 (n=5) | 5.032 +/- 0.712 (n=5) | 31.242 +/- 1.445 (n=5) |
| dp_epsilon_4 | 75.550 +/- 0.334 (n=5) | 73.367 +/- 0.362 (n=5) | 0.8176 +/- 0.0058 (n=5) | 0.7529 +/- 0.0231 (n=5) | 5.044 +/- 0.642 (n=5) | 30.792 +/- 1.163 (n=5) |
| dp_epsilon_8 | 75.451 +/- 0.351 (n=5) | 73.217 +/- 0.406 (n=5) | 0.8177 +/- 0.0060 (n=5) | 0.7598 +/- 0.0272 (n=5) | 5.015 +/- 0.581 (n=5) | 30.962 +/- 1.440 (n=5) |
| poisson_unclipped | 77.568 +/- 0.415 (n=5) | 76.440 +/- 0.270 (n=5) | 0.8464 +/- 0.0005 (n=5) | 0.4778 +/- 0.0072 (n=5) | 8.003 +/- 2.260 (n=5) | 18.040 +/- 2.442 (n=5) |
| poisson_clipped | 75.513 +/- 0.345 (n=5) | 73.281 +/- 0.370 (n=5) | 0.8177 +/- 0.0061 (n=5) | 0.7570 +/- 0.0242 (n=5) | 4.883 +/- 0.648 (n=5) | 30.799 +/- 1.433 (n=5) |


### exclude_sensitive

| Method | Accuracy (%) | Balanced accuracy (%) | ROC-AUC | BCE | Sex separation (pp) | Race n>=500 separation (pp) |
| --- | --- | --- | --- | --- | --- | --- |
| sgd | 76.985 +/- 0.109 (n=5) | 75.517 +/- 0.254 (n=5) | 0.8300 +/- 0.0007 (n=5) | 0.4927 +/- 0.0017 (n=5) | 12.087 +/- 0.214 (n=5) | 19.860 +/- 1.648 (n=5) |
| dp_epsilon_1 | 75.734 +/- 0.095 (n=5) | 73.524 +/- 0.154 (n=5) | 0.8111 +/- 0.0008 (n=5) | 0.7580 +/- 0.0131 (n=5) | 11.173 +/- 0.345 (n=5) | 24.810 +/- 0.833 (n=5) |
| dp_epsilon_4 | 75.732 +/- 0.071 (n=5) | 73.520 +/- 0.104 (n=5) | 0.8110 +/- 0.0008 (n=5) | 0.7579 +/- 0.0111 (n=5) | 11.177 +/- 0.301 (n=5) | 25.073 +/- 0.314 (n=5) |
| dp_epsilon_8 | 75.755 +/- 0.096 (n=5) | 73.557 +/- 0.156 (n=5) | 0.8109 +/- 0.0009 (n=5) | 0.7560 +/- 0.0140 (n=5) | 11.219 +/- 0.305 (n=5) | 24.677 +/- 0.708 (n=5) |
| poisson_unclipped | 76.933 +/- 0.242 (n=5) | 75.363 +/- 0.213 (n=5) | 0.8299 +/- 0.0008 (n=5) | 0.4931 +/- 0.0047 (n=5) | 12.069 +/- 0.405 (n=5) | 20.484 +/- 1.364 (n=5) |
| poisson_clipped | 75.713 +/- 0.051 (n=5) | 73.489 +/- 0.077 (n=5) | 0.8110 +/- 0.0007 (n=5) | 0.7596 +/- 0.0189 (n=5) | 11.055 +/- 0.272 (n=5) | 25.077 +/- 0.647 (n=5) |


## Paired accuracy losses versus same-seed SGD (pp)

| feature_policy | method | mean | std | valid_seeds |
| --- | --- | --- | --- | --- |
| exclude_sensitive | dp_epsilon_1 | 1.25099 | 0.06281 | 5 |
| exclude_sensitive | dp_epsilon_4 | 1.25297 | 0.08513 | 5 |
| exclude_sensitive | dp_epsilon_8 | 1.2305 | 0.04331 | 5 |
| include_sensitive | dp_epsilon_1 | 1.90523 | 0.71294 | 5 |
| include_sensitive | dp_epsilon_4 | 1.87285 | 0.72695 | 5 |
| include_sensitive | dp_epsilon_8 | 1.97132 | 0.70116 | 5 |


Positive loss means DP was less accurate. Each difference is calculated within policy/seed first, then averaged.

## Budget differences versus training variation

| feature_policy | comparison | valid_seeds | later_budget_higher_seeds | later_budget_lower_seeds | tied_seeds | accuracy_change_pp | accuracy_change_sample_sd_pp |
| --- | --- | --- | --- | --- | --- | --- | --- |
| exclude_sensitive | dp_epsilon_4 minus dp_epsilon_1 | 5 | 2 | 3 | 0 | -0.00198 | 0.11357 |
| exclude_sensitive | dp_epsilon_8 minus dp_epsilon_1 | 5 | 3 | 2 | 0 | 0.02049 | 0.07173 |
| exclude_sensitive | dp_epsilon_8 minus dp_epsilon_4 | 5 | 4 | 1 | 0 | 0.02247 | 0.06869 |
| include_sensitive | dp_epsilon_4 minus dp_epsilon_1 | 5 | 3 | 2 | 0 | 0.03238 | 0.05927 |
| include_sensitive | dp_epsilon_8 minus dp_epsilon_1 | 5 | 0 | 5 | 0 | -0.06609 | 0.04225 |
| include_sensitive | dp_epsilon_8 minus dp_epsilon_4 | 5 | 0 | 5 | 0 | -0.09847 | 0.08775 |


Differences above are later method minus earlier method in percentage points. Compare each mean with its paired SD and seed-level signs; neither monotonicity nor significance is assumed. All budget differences for fairness components remain in the full aggregate CSV. Small budget effects motivate extending prespecified important settings to ten seeds for final analysis.

## Female versus male false-positive rates

| feature_policy | method | valid_seeds | female_higher_seeds | mean | std |
| --- | --- | --- | --- | --- | --- |
| exclude_sensitive | dp_epsilon_1 | 5 | 5 | 11.17303 | 0.34547 |
| exclude_sensitive | dp_epsilon_4 | 5 | 5 | 11.17661 | 0.30054 |
| exclude_sensitive | dp_epsilon_8 | 5 | 5 | 11.21912 | 0.30521 |
| exclude_sensitive | poisson_clipped | 5 | 5 | 11.05518 | 0.2717 |
| exclude_sensitive | poisson_unclipped | 5 | 5 | 12.06921 | 0.4051 |
| exclude_sensitive | sgd | 5 | 5 | 12.08735 | 0.21402 |
| include_sensitive | dp_epsilon_1 | 5 | 3 | 1.27467 | 2.1355 |
| include_sensitive | dp_epsilon_4 | 5 | 4 | 1.44163 | 1.94708 |
| include_sensitive | dp_epsilon_8 | 5 | 3 | 0.96041 | 2.07029 |
| include_sensitive | poisson_clipped | 5 | 4 | 1.29461 | 2.03873 |
| include_sensitive | poisson_unclipped | 5 | 5 | 2.03615 | 0.93576 |
| include_sensitive | sgd | 5 | 5 | 3.05862 | 1.49133 |


Positive mean means female FPR is higher. Counts show how many valid seeds have this ordering. ESR-specific comparisons are in `aggregate/within_esr_sex_fpr_difference_mean_std.csv`; ESR1 is not an actual-negative category, so its FPR stays undefined. Small conditional denominators remain flagged in the seed-level ESR tables.

## Audit findings and interpretation limits

- Feature inclusion uses 16 input variables/87 encoded columns; exclusion uses 14/77. Race code4 is absent from reference data: its one-hot block uses the existing unknown-category rule, while all race labels/records remain in evaluation.
- Household overlap exists under the fixed original person split. No split was changed; these are individual-record DP claims, not household-level guarantees.
- Ordinary SGD uses actual-batch mean gradients. Both Poisson controls match DP expected-batch normalization. Compare shuffled SGD -> Poisson-unclipped -> Poisson-clipped -> DP to diagnose the sequence of changes. Clipped/unclipped controls share a deterministic sampling stream; DP sampling/noise are fresh.
- Full nine-race and intersection metrics remain in detailed tables, with missing values preserved. The n>=500 headline race comparison is separately labeled and is not the full-race result.
- Protected-feature inclusion versus exclusion uses the same initialization seed, but input dimensions differ: weights cannot be identical across those policies. Exact initialization hashes match within each policy/seed.
- No interventions, target changes, final-test evaluation, or publication figures were performed. The repository includes non-private baselines/controls/audits; per-run DP estimates are not a whole-repository guarantee.

## Tables and next review

All seed-level, component/group, paired, feature-policy, ESR, and composition tables are under `aggregate/`. `plot_plan.json` specifies at most three candidate main figures; the reusable plotting command waits for explicit selection. Review whether clipping/sampling explains the main loss, whether budget changes exceed variability, whether female FPR persists within ESR categories, and which settings need ten seeds.

## Resume/reproduce

From the repository root: ` .\.venv\Scripts\python.exe scripts\run_controlled.py --stage main `, then `--stage controls`, then `--stage summarize`. Completed matching runs are skipped. See the root README for preparation, tests, and interrupted-lock handling.

## Concrete training contrasts

- include_sensitive: SGD -> Poisson-unclipped accuracy change +0.145 pp; Poisson-unclipped -> clipped -2.055 pp; clipped -> DP1 +0.005 pp.

- include_sensitive: epsilon8 minus epsilon1 accuracy is -0.066 pp (paired sample SD 0.042 pp; 5 valid seeds). This is a variability comparison, not a significance claim.

- exclude_sensitive: SGD -> Poisson-unclipped accuracy change -0.052 pp; Poisson-unclipped -> clipped -1.220 pp; clipped -> DP1 +0.021 pp.

- exclude_sensitive: epsilon8 minus epsilon1 accuracy is +0.020 pp (paired sample SD 0.072 pp; 5 valid seeds). This is a variability comparison, not a significance claim.

The two non-private Poisson controls share sample streams; DP draws fresh samples/noise. Use these contrasts to distinguish sampling/normalization and clipping from additional noise, while retaining seed variability.

## Effect of including sensitive predictors

| method | metric | mean | std | valid_seeds |
| --- | --- | --- | --- | --- |
| dp_epsilon_1 | RAC1P_n500_separation | 6.43187 | 1.51625 | 5 |
| dp_epsilon_1 | SEX_separation | -6.1411 | 0.71529 | 5 |
| dp_epsilon_1 | accuracy | -0.21676 | 0.36576 | 5 |
| sgd | RAC1P_n500_separation | -3.13694 | 1.73218 | 5 |
| sgd | SEX_separation | -4.33484 | 2.83197 | 5 |
| sgd | accuracy | 0.43748 | 0.44992 | 5 |


Differences are include_sensitive minus exclude_sensitive, in percentage points. Positive accuracy difference favors inclusion; positive gap difference indicates more disparity on that criterion. All budgets and controls are retained in the full feature-policy CSV.

## ESR diagnostic example: include_sensitive, DP epsilon1

| ESR | Meaning | Male_n | Female_n | Male_FPR | Female_FPR | Female_higher_seeds |
| --- | --- | --- | --- | --- | --- | --- |
| 2 | Civilian employed, temporarily absent | 170 | 262 | 88.706 +/- 0.767 (n=5) | 81.221 +/- 0.827 (n=5) | 0/5 |
| 3 | Unemployed | 514 | 480 | 84.202 +/- 1.154 (n=5) | 80.375 +/- 0.713 (n=5) | 0/5 |
| 4 | Armed forces, at work | 137 | 20 | 46.277 +/- 1.107 (n=5) | 51.000 +/- 2.236 (n=5) | 5/5 |
| 5 | Armed forces, temporarily absent | 1 | 0 | 100.000 +/- 0.000 (n=5) | undefined | 0/0 |
| 6 | Not in labor force | 4880 | 6548 | 35.393 +/- 1.839 (n=5) | 38.629 +/- 0.647 (n=5) | 5/5 |


FPR entries are percent mean +/- sample SD. Counts are development people in each actual-negative ESR category, not five times that population. Categories with fewer than30 people have weak support even if the training-seed ordering is consistent. ESR1 is actual positive and has no FPR denominator.

Share of each sex group's false positives contributed by each ESR category (percent):

| ESR | Male_FP_share | Female_FP_share |
| --- | --- | --- |
| 2 | 6.357 +/- 0.253 (n=5) | 6.781 +/- 0.059 (n=5) |
| 3 | 18.239 +/- 0.529 (n=5) | 12.295 +/- 0.110 (n=5) |
| 4 | 2.671 +/- 0.081 (n=5) | 0.325 +/- 0.015 (n=5) |
| 5 | 0.042 +/- 0.002 (n=5) | 0.000 +/- 0.000 (n=5) |
| 6 | 72.691 +/- 0.854 (n=5) | 80.599 +/- 0.151 (n=5) |


These tables distinguish within-category error rates from differences in the composition of the negative class. They do not justify relabeling ESR2/4/5 without a separate target-definition decision.

## Recommended figures after review (at most three)

1. Accuracy and sex/headline-race separation versus epsilon, with SGD and Poisson controls: show the small budget differences alongside the much larger clipping-related utility change.
2. Paired accuracy losses by sex and supported race group under both feature policies: show the observed change in which groups bear larger costs.
3. Sex-specific TPR/FPR across methods, with an ESR diagnostic inset if selected: show error levels, feature-policy dependence, and label-composition effects.

These are recommendations only; no publication figures have been rendered. Review the five-seed findings, prespecify ten-seed follow-up settings, and decide whether an alternative target deserves a separate experiment before implementing interventions.
