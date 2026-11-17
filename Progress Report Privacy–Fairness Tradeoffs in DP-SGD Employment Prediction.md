# Progress Report: Privacy–Fairness Tradeoffs in DP-SGD Employment Prediction

*Draft for November 18, 2026. All results are single-seed development-set results; the test set has not been loaded.*

## 1. Motivation

Differentially private SGD (DP-SGD) protects individual training records by clipping per-person gradients and adding noise. Bagdasaryan et al. (2019), *Differential Privacy Has Disparate Impact on Model Accuracy*, showed that this can cost underrepresented groups more accuracy than the majority. They measured accuracy parity and left other fairness notions to future work.

We extend that question to tabular employment prediction. We ask: (1) Does DP-SGD change accuracy disproportionately across sex and race groups? (2) Do Learning Fair Representations (LFR) and equalized-odds (EO) post-processing reduce observed disparities, and at what accuracy cost? (3) How do effects vary with the privacy budget? Beyond accuracy parity, we also report independence, separation, and binary sufficiency. This is an experimental extension, not a reproduction of the original datasets or architectures. A finding of no consistent effect is a valid result.

## 2. Task and data

**Data.** California 2018 ACS 1-Year person records, via Folktables. We use the filtered employment task: ages 17–89 (AGEP > 16 and < 90) and person weight PWGTP ≥ 1.

**Target.** Y = 1 exactly when ESR = 1 (civilian employed, at work), following the Folktables benchmark. Y = 0 therefore includes unemployed people, people outside the labor force, employed civilians temporarily away from work (ESR 2), and Armed Forces codes (ESR 4–5). It does not mean "unemployed." The positive rate is about 57%.

**Inputs.** 14 source variables become 77 encoded columns (scaled age plus one-hot categories). Sex (SEX) and race (RAC1P) are held out as audit metadata and are not model inputs; correlated features can still carry group information. ESP (parents' employment status) was dropped by decision on Nov 16 because it is 98.58% structurally missing; all eligible people were kept. Preprocessing is fitted only on the reference partition.

| Step | Count |
| --- | --- |
| Raw people | 378,817 |
| Excluded (age or weight rules) | 76,177 |
| Eligible | 302,640 |

| Partition | People | Purpose |
| --- | --- | --- |
| Reference | 30,264 | Fit preprocessing; later LFR (public) |
| Train | 151,320 | Fit ordinary or private MLP (private) |
| Development | 30,264 | Evaluate pilots (public) |
| Calibration | 30,264 | Later EO rules (public) |
| Test | 60,528 | Final evaluation; untouched |

Splits are random at the person level (seed 285), not stratified, and disjoint across partitions. Household-level separation has not been enforced and is a decision to review before freezing the protocol.

## 3. Model, training, and privacy

The model is an MLP, 77 → 64 → 32 → 1 with ReLU, trained with binary cross-entropy, SGD with learning rate 0.05, no momentum, batch size 512, 20 epochs, threshold 0.5, and the final epoch kept (no checkpoint selection). The ordinary baseline and DP runs share the architecture, initialization seed (285), data fingerprint, and hyperparameters.

DP runs use Opacus 1.6.0 with Poisson sampling (q ≈ 0.00338, 296 steps per "epoch"), per-person gradient clipping at C = 1, an RDP accountant, and δ = 1/(10·n\_train) ≈ 6.6×10⁻⁷. Each run takes 5,920 steps.

| Target ε | Achieved ε | Noise multiplier | Runtime (s) |
| --- | --- | --- | --- |
| 1 | 0.999 | 1.411 | 48 |
| 4 | 3.999 | 0.741 | 79 |
| 8 | 8.000 | 0.597 | 68 |

**Privacy scope.** The guarantee applies to records in the training partition only, conditional on fixed public reference, development, and calibration partitions and on preprocessing fitted there. Each ε describes a single run. Releasing all three models together costs at most ε ≈ 13.0, δ ≈ 1.98×10⁻⁶ under basic composition. The ordinary baseline is not a private release. Runs use research-grade random number generators (secure\_mode = False); sampling and noise seeds are not exported.

## 4. Metrics

For each group g we compute accuracy, selection rate, TPR, FPR, PPV, and NPV from its confusion matrix. Gaps are max − min across the groups being compared.

| Criterion | Gap reported |
| --- | --- |
| Accuracy parity | accuracy gap |
| Independence | selection-rate gap |
| Separation (equalized odds) | max(TPR gap, FPR gap) |
| Binary sufficiency | max(PPV gap, NPV gap) |

Disproportionate accuracy loss is loss\_g(ε) = accuracy\_SGD,g − accuracy\_DP,g(ε) on the same development records. Positive means privacy reduced accuracy. Rates with fewer than 30 observations in a denominator are flagged as unstable. If any required group rate is undefined, the full-group gap is reported as undefined rather than as zero, and no group is silently dropped. These are predictive-disparity measurements on observational data and do not support causal claims about why employment differs.

## 5. Preliminary results

### 5.1 The baseline learns

The non-private MLP reaches 77.08% development accuracy (balanced accuracy 75.73%), versus 57.01% for always predicting the majority class. Training loss falls from 0.670 to 0.495 and development loss from 0.636 to 0.492, with no sign of overfitting.

### 5.2 Overall cost of privacy

| Method | Achieved ε | Accuracy | Balanced accuracy | Accuracy loss (pp) |
| --- | --- | --- | --- | --- |
| SGD | — | 77.08% | 75.73% | 0.00 |
| DP-SGD | 1 | 75.82% | 73.67% | 1.26 |
| DP-SGD | 4 | 75.81% | 73.69% | 1.27 |
| DP-SGD | 8 | 75.78% | 73.63% | 1.30 |

Privacy costs about 1.3 percentage points of accuracy and about 2.1 points of balanced accuracy. The cost is essentially flat across ε = 1 to 8; the 0.04 pp spread is far smaller than the sampling noise of a 30,264-record evaluation (roughly 0.24 pp standard error), so we do not claim any trend with ε. Balanced accuracy falls more than accuracy because DP models predict the positive class more often: the overall selection rate rises from 63.2% to about 68.5%, TPR from 85.3% to about 88.9%, and FPR from 33.9% to about 41.5%. (Figure 1.)

### 5.3 Accuracy loss by group

| Group | n | SGD acc. | Loss at ε=1 (pp) | Loss at ε=4 (pp) | Loss at ε=8 (pp) |
| --- | --- | --- | --- | --- | --- |
| Male | 14,826 | 80.48% | 0.90 | 0.93 | 0.91 |
| Female | 15,438 | 73.81% | 1.61 | 1.59 | 1.67 |
| White alone | 18,889 | 77.33% | 1.24 | 1.18 | 1.23 |
| Black alone | 1,500 | 79.73% | 1.07 | 1.27 | 1.40 |
| Asian alone | 5,017 | 76.78% | 1.63 | 1.69 | 1.65 |
| Some other race alone | 3,299 | 75.02% | 0.67 | 0.97 | 0.88 |
| Two or more races | 1,179 | 77.27% | 1.70 | 1.61 | 1.78 |

Women lose roughly 1.6–1.7 pp versus roughly 0.9 pp for men, and the pattern holds at all three budgets. This is consistent with a disproportionate impact on the lower-accuracy sex group, but it is one run per budget, so we treat it as suggestive. Among the larger race groups, losses range from about 0.7 to 1.8 pp with no consistent ordering across ε. The very small groups (American Indian n = 202, Other AIAN n = 81, Pacific Islander n = 95, Alaska Native n = 2) are omitted above because a handful of records can swing their losses; for example, the Pacific Islander group shows a 3.16 pp loss at every ε, which is 3 records. (Figure 2.)

### 5.4 Fairness gaps

Sex (gaps in percentage points):

| Method | Accuracy gap | Independence | Separation | Sufficiency |
| --- | --- | --- | --- | --- |
| SGD | 6.67 | 0.64 | 12.01 | 12.58 |
| DP-SGD ε=1 | 7.38 | 0.78 | 11.38 | 12.12 |
| DP-SGD ε=4 | 7.33 | 0.91 | 11.51 | 12.16 |
| DP-SGD ε=8 | 7.43 | 0.87 | 11.54 | 12.19 |

The sex accuracy gap widens by about 0.7 pp under DP, while separation and sufficiency gaps shrink slightly (about 0.5 pp). Different criteria move in different directions, and none of these changes is large relative to run-to-run noise. The separation gap is driven almost entirely by FPR: in the baseline, men have FPR 27.1% and women 39.1%, while the TPR gap is only 0.2 pp. The selection rates are nearly equal (63.0% vs 63.5%) even though the label base rates differ (61.5% vs 52.6%), which is why independence looks small while accuracy and FPR differ.

Race: the accuracy gap is 31.2 pp for SGD and about 32.2 pp for DP, and the independence gap rises from 69.5 to about 77 pp. Both are driven by tiny groups: the Alaska Native group has n = 2 in development (no positives), so its accuracy and selection rate are extreme. Full-race **separation and sufficiency are undefined** because that group has no positives, and we report them as undefined rather than excluding categories to obtain a number. (Figure 3.) Race-level gap numbers should therefore not be read as findings about a typical race group. We propose a prespecified supported comparison (White alone vs. Black alone, as in the proposal), labeled separately from full-population results.

## 6. Limitations

- **Single seed, single run.** Each budget has one run with fixed initialization and independent sampling/noise. We cannot separate real ε effects from run-to-run variation.
- **Mechanism confounded.** Ordinary SGD uses shuffled fixed batches; DP-SGD uses Poisson sampling plus clipping plus noise, so the accuracy loss cannot be attributed to noise alone. A clipping-only or sampling-matched diagnostic is planned.
- **Development data only.** No test outcomes were used. Development data is also treated as public auxiliary data in our privacy design.
- **Small groups.** Race categories 4, 5, and 7 and many race-by-sex cells have fewer than 30 observations in some conditional denominators. In calibration, several race-by-sex cells are empty or single-label, which blocks a full intersectional EO fit.
- **Target definition.** ESR = 1 excludes employed civilians temporarily away and military employment; a broader target remains a pending decision.
- **Unweighted metrics.** Results describe this sample, not survey-weighted California population estimates.

## 7. Remaining work

| Dates | Work |
| --- | --- |
| Nov 19–21 | EO post-processing for sex and race, with non-private controls; audit both attributes for every output |
| Nov 22–24 | LFR on reference data (sex first, then the supported binary race cohort); downstream ordinary and private MLPs with matched controls |
| Nov 25 | Freeze cohorts, splits, preprocessing, tuning rules, seeds, and metrics |
| Nov 25–30 | Main grid: SGD, DP-SGD, LFR+SGD, LFR+DP-SGD, SGD+EO, DP-SGD+EO; ε in {0.5, 1, 2, 4, 8} (fallback {1, 4, 8}); at least three seeds |
| Dec 1–3 | Final test evaluation, uncertainty intervals, and figures |
| Dec 4–7 | Final report; submit by Dec 7 |

## 8. Reproducibility

Settings are in configs/baseline.json and configs/dp\_pilot.json. Commands: `scripts/prepare_data.py`, `scripts/train_baseline.py`, `scripts/train_dp.py`, `scripts/report_dp.py`, and `python -m unittest discover -s tests`. Completed DP runs refuse to be overwritten; a replicate needs a copied config with a new output directory, and each additional released run composes privacy cost. Raw data, checkpoints, and row-level predictions are excluded from Git.

## References

- Bagdasaryan, Poursaeed, Shmatikov (2019). Differential Privacy Has Disparate Impact on Model Accuracy. NeurIPS.
- Ding, Hardt, Miller, Schmidt (2021). Retiring Adult: New Datasets for Fair Machine Learning. NeurIPS (Folktables).
- Hardt, Price, Srebro (2016). Equality of Opportunity in Supervised Learning.
- Zemel, Wu, Swersky, Pitassi, Dwork (2013). Learning Fair Representations. ICML.
- Barocas, Hardt, Narayanan. Fairness and Machine Learning, classification criteria chapter.
- Dwork and Roth. The Algorithmic Foundations of Differential Privacy.
- Opacus (PrivacyEngine) documentation, opacus.ai.
- U.S. Census Bureau, 2018 ACS PUMS Data Dictionary.

## Figures to insert

- **Figure 1:** figures/dp\_pilot/accuracy\_vs\_epsilon.png
- **Figure 2:** figures/dp\_pilot/group\_loss\_vs\_epsilon.png (new; generated by scripts/plot\_group\_losses.py)
- **Figure 3:** figures/dp\_pilot/fairness\_gaps.png
