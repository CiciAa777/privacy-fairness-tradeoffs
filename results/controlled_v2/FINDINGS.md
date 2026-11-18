# Controlled v2: baseline findings

Completed main runs: **40/40**; controls: **6/20**. Development data only. Means +/- sample SD across training seeds; no population-sampling uncertainty is estimated.

## Overall accuracy and selected fairness outcomes

### include_sensitive

| Method | Accuracy (%) | Balanced accuracy (%) | ROC-AUC | BCE | Sex separation (pp) | Race n>=500 separation (pp) |
| --- | --- | --- | --- | --- | --- | --- |
| sgd | 77.423 +/- 0.551 (n=5) | 76.507 +/- 0.258 (n=5) | 0.846 +/- 0.000 (n=5) | 0.480 +/- 0.008 (n=5) | 7.753 +/- 3.001 (n=5) | 16.723 +/- 2.769 (n=5) |
| dp_epsilon_1 | 75.517 +/- 0.343 (n=5) | 73.319 +/- 0.358 (n=5) | 0.818 +/- 0.006 (n=5) | 0.755 +/- 0.020 (n=5) | 5.032 +/- 0.712 (n=5) | 31.242 +/- 1.445 (n=5) |
| dp_epsilon_4 | 75.550 +/- 0.334 (n=5) | 73.367 +/- 0.362 (n=5) | 0.818 +/- 0.006 (n=5) | 0.753 +/- 0.023 (n=5) | 5.044 +/- 0.642 (n=5) | 30.792 +/- 1.163 (n=5) |
| dp_epsilon_8 | 75.451 +/- 0.351 (n=5) | 73.217 +/- 0.406 (n=5) | 0.818 +/- 0.006 (n=5) | 0.760 +/- 0.027 (n=5) | 5.015 +/- 0.581 (n=5) | 30.962 +/- 1.440 (n=5) |
| poisson_unclipped | 77.680 +/- 0.168 (n=2) | 76.411 +/- 0.407 (n=2) | 0.847 +/- 0.000 (n=2) | 0.476 +/- 0.003 (n=2) | 7.115 +/- 2.550 (n=2) | 18.272 +/- 2.177 (n=2) |
| poisson_clipped | 75.560 +/- 0.236 (n=2) | 73.306 +/- 0.245 (n=2) | 0.820 +/- 0.002 (n=2) | 0.744 +/- 0.002 (n=2) | 4.660 +/- 0.235 (n=2) | 30.418 +/- 2.095 (n=2) |


### exclude_sensitive

| Method | Accuracy (%) | Balanced accuracy (%) | ROC-AUC | BCE | Sex separation (pp) | Race n>=500 separation (pp) |
| --- | --- | --- | --- | --- | --- | --- |
| sgd | 76.985 +/- 0.109 (n=5) | 75.517 +/- 0.254 (n=5) | 0.830 +/- 0.001 (n=5) | 0.493 +/- 0.002 (n=5) | 12.087 +/- 0.214 (n=5) | 19.860 +/- 1.648 (n=5) |
| dp_epsilon_1 | 75.734 +/- 0.095 (n=5) | 73.524 +/- 0.154 (n=5) | 0.811 +/- 0.001 (n=5) | 0.758 +/- 0.013 (n=5) | 11.173 +/- 0.345 (n=5) | 24.810 +/- 0.833 (n=5) |
| dp_epsilon_4 | 75.732 +/- 0.071 (n=5) | 73.520 +/- 0.104 (n=5) | 0.811 +/- 0.001 (n=5) | 0.758 +/- 0.011 (n=5) | 11.177 +/- 0.301 (n=5) | 25.073 +/- 0.314 (n=5) |
| dp_epsilon_8 | 75.755 +/- 0.096 (n=5) | 73.557 +/- 0.156 (n=5) | 0.811 +/- 0.001 (n=5) | 0.756 +/- 0.014 (n=5) | 11.219 +/- 0.305 (n=5) | 24.677 +/- 0.708 (n=5) |
| poisson_unclipped | 76.794 +/- undefined (n=1) | 75.560 +/- undefined (n=1) | 0.830 +/- undefined (n=1) | 0.494 +/- undefined (n=1) | 12.134 +/- undefined (n=1) | 19.545 +/- undefined (n=1) |
| poisson_clipped | 75.734 +/- undefined (n=1) | 73.525 +/- undefined (n=1) | 0.810 +/- undefined (n=1) | 0.749 +/- undefined (n=1) | 10.908 +/- undefined (n=1) | 24.420 +/- undefined (n=1) |


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

| feature_policy | comparison | metric | mean | std | valid_seeds |
| --- | --- | --- | --- | --- | --- |
| exclude_sensitive | dp_epsilon_4 minus dp_epsilon_1 | RAC1P_n500_separation | 0.26252 | 0.84448 | 5 |
| exclude_sensitive | dp_epsilon_4 minus dp_epsilon_1 | SEX_separation | 0.00358 | 0.22704 | 5 |
| exclude_sensitive | dp_epsilon_4 minus dp_epsilon_1 | accuracy | -0.00198 | 0.11357 | 5 |
| exclude_sensitive | dp_epsilon_8 minus dp_epsilon_1 | RAC1P_n500_separation | -0.13289 | 0.3167 | 5 |
| exclude_sensitive | dp_epsilon_8 minus dp_epsilon_1 | SEX_separation | 0.04609 | 0.06933 | 5 |
| exclude_sensitive | dp_epsilon_8 minus dp_epsilon_1 | accuracy | 0.02049 | 0.07173 | 5 |
| exclude_sensitive | dp_epsilon_8 minus dp_epsilon_4 | RAC1P_n500_separation | -0.39541 | 0.62949 | 5 |
| exclude_sensitive | dp_epsilon_8 minus dp_epsilon_4 | SEX_separation | 0.04251 | 0.19 | 5 |
| exclude_sensitive | dp_epsilon_8 minus dp_epsilon_4 | accuracy | 0.02247 | 0.06869 | 5 |
| include_sensitive | dp_epsilon_4 minus dp_epsilon_1 | RAC1P_n500_separation | -0.45066 | 0.6058 | 5 |
| include_sensitive | dp_epsilon_4 minus dp_epsilon_1 | SEX_separation | 0.01225 | 0.15683 | 5 |
| include_sensitive | dp_epsilon_4 minus dp_epsilon_1 | accuracy | 0.03238 | 0.05927 | 5 |
| include_sensitive | dp_epsilon_8 minus dp_epsilon_1 | RAC1P_n500_separation | -0.28028 | 0.21898 | 5 |
| include_sensitive | dp_epsilon_8 minus dp_epsilon_1 | SEX_separation | -0.01711 | 0.1686 | 5 |
| include_sensitive | dp_epsilon_8 minus dp_epsilon_1 | accuracy | -0.06609 | 0.04225 | 5 |
| include_sensitive | dp_epsilon_8 minus dp_epsilon_4 | RAC1P_n500_separation | 0.17038 | 0.66573 | 5 |
| include_sensitive | dp_epsilon_8 minus dp_epsilon_4 | SEX_separation | -0.02936 | 0.12221 | 5 |
| include_sensitive | dp_epsilon_8 minus dp_epsilon_4 | accuracy | -0.09847 | 0.08775 | 5 |


Differences above are later method minus earlier method in percentage points. Compare each mean with its paired SD and seed-level signs; neither monotonicity nor significance is assumed. Small budget effects motivate extending prespecified important settings to ten seeds for final analysis.

## Female versus male false-positive rates

| feature_policy | method | valid_seeds | female_higher_seeds | mean | std |
| --- | --- | --- | --- | --- | --- |
| exclude_sensitive | dp_epsilon_1 | 5 | 5 | 11.17303 | 0.34547 |
| exclude_sensitive | dp_epsilon_4 | 5 | 5 | 11.17661 | 0.30054 |
| exclude_sensitive | dp_epsilon_8 | 5 | 5 | 11.21912 | 0.30521 |
| exclude_sensitive | poisson_clipped | 1 | 1 | 10.90765 | undefined |
| exclude_sensitive | poisson_unclipped | 1 | 1 | 12.13429 | undefined |
| exclude_sensitive | sgd | 5 | 5 | 12.08735 | 0.21402 |
| include_sensitive | dp_epsilon_1 | 5 | 3 | 1.27467 | 2.1355 |
| include_sensitive | dp_epsilon_4 | 5 | 4 | 1.44163 | 1.94708 |
| include_sensitive | dp_epsilon_8 | 5 | 3 | 0.96041 | 2.07029 |
| include_sensitive | poisson_clipped | 2 | 2 | 1.41418 | 1.03797 |
| include_sensitive | poisson_unclipped | 2 | 2 | 2.68533 | 1.25194 |
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

- include_sensitive: SGD -> Poisson-unclipped accuracy change +0.257 pp; Poisson-unclipped -> clipped -2.120 pp; clipped -> DP1 -0.043 pp.

- include_sensitive: epsilon8 minus epsilon1 accuracy is -0.066 pp (paired sample SD 0.042 pp; 5 valid seeds). This is a variability comparison, not a significance claim.

- exclude_sensitive: SGD -> Poisson-unclipped accuracy change -0.191 pp; Poisson-unclipped -> clipped -1.061 pp; clipped -> DP1 +0.001 pp.

- exclude_sensitive: epsilon8 minus epsilon1 accuracy is +0.020 pp (paired sample SD 0.072 pp; 5 valid seeds). This is a variability comparison, not a significance claim.

The two non-private Poisson controls share sample streams; DP draws fresh samples/noise. Use these contrasts to distinguish sampling/normalization and clipping from additional noise, while retaining seed variability.

## Effect of including sensitive predictors

| method | metric | mean | std | valid_seeds |
| --- | --- | --- | --- | --- |
| dp_epsilon_1 | RAC1P_n500_separation | 6.43187 | 1.51625 | 5 |
| dp_epsilon_1 | SEX_separation | -6.1411 | 0.71529 | 5 |
| dp_epsilon_1 | accuracy | -0.21676 | 0.36576 | 5 |
| dp_epsilon_4 | RAC1P_n500_separation | 5.71868 | 1.18222 | 5 |
| dp_epsilon_4 | SEX_separation | -6.13243 | 0.69898 | 5 |
| dp_epsilon_4 | accuracy | -0.18239 | 0.34685 | 5 |
| dp_epsilon_8 | RAC1P_n500_separation | 6.28447 | 1.42802 | 5 |
| dp_epsilon_8 | SEX_separation | -6.2043 | 0.53721 | 5 |
| dp_epsilon_8 | accuracy | -0.30333 | 0.38054 | 5 |
| poisson_clipped | RAC1P_n500_separation | 7.47959 | undefined | 1 |
| poisson_clipped | SEX_separation | -6.08145 | undefined | 1 |
| poisson_clipped | accuracy | -0.34034 | undefined | 1 |
| poisson_unclipped | RAC1P_n500_separation | -2.81188 | undefined | 1 |
| poisson_unclipped | SEX_separation | -3.21548 | undefined | 1 |
| poisson_unclipped | accuracy | 0.76659 | undefined | 1 |
| sgd | RAC1P_n500_separation | -3.13694 | 1.73218 | 5 |
| sgd | SEX_separation | -4.33484 | 2.83197 | 5 |
| sgd | accuracy | 0.43748 | 0.44992 | 5 |


Differences are include_sensitive minus exclude_sensitive, in percentage points. Positive accuracy difference favors inclusion; positive gap difference indicates more disparity on that criterion.

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
