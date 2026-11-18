# Privacy–Fairness Tradeoffs: Research Proposal

**Working design, updated November 17, 2026.** The controlled rerun now takes priority; [plan.md](plan.md) records the changes and execution stages. The original design, single-run results, and report are preserved under `V0/`.

## 1. The project in plain language

We will predict the Folktables employment label using a small neural network. We will compare ordinary training with differentially private training, measure how performance changes across sex and race groups, and test whether two fairness interventions help.

| Question | Starting specification |
| --- | --- |
| Which data? | California, 2018 ACS 1-Year person data |
| Which population? | Ages 17–89, with valid person weights |
| What do we predict? | Whether `ESR == 1`, the benchmark's positive employment label |
| Which groups do we audit? | Sex and race separately; race-by-sex intersections where supported |
| Main outcomes? | Group accuracy losses under DP and equalized-odds gaps |
| Additional fairness outcomes? | Independence and binary-decision sufficiency |
| Which interventions? | Learning Fair Representations (LFR) and equalized-odds post-processing |

Research questions:

1. Does DP-SGD change accuracy disproportionately across groups?
2. Do LFR and equalized-odds post-processing reduce the observed disparities, and at what accuracy cost?
3. How do these effects vary across privacy budgets?

Worsening fairness is a hypothesis. Mixed effects or no consistent effect are valid findings.

## 2. What we extend from Bagdasaryan et al. (2019)

[Differential Privacy Has Disparate Impact on Model Accuracy](https://papers.nips.cc/paper_files/paper/2019/file/fc0de4e0396fff257ea362983c2dda5a-Paper.pdf) studies unequal accuracy costs of private training. Section 3.4 calls its fairness criterion **accuracy parity**. It examines underrepresented classes and subgroups, and leaves other fairness notions for future work.

Examples from the paper:

| Prediction task | Target being predicted | Groups used for accuracy comparisons |
| --- | --- | --- |
| Facial gender classification | Recorded gender | Skin-color groups |
| Facial age classification | Age category | Age × gender × skin-color intersections |
| Tweet sentiment classification | Positive/negative sentiment | Standard American/African-American English |

The models learn their prediction tasks through a task loss. Private training clips gradients and adds noise; these experiments do not impose a fairness penalty. Gender and age can therefore be prediction targets in those experiments even though they can serve as protected attributes in other applications.

Our extension changes the dataset to ACS Employment, keeps employment as the target, adds independence/separation/sufficiency assessments, and compares pre-processing with post-processing interventions. This is an experimental extension rather than a reproduction of the original datasets or architectures.

## 3. Dataset, population, and target

### Population and task variant

Use the **filtered employment task**: `16 < AGEP < 90` and `PWGTP >= 1`. This follows the stated population in Appendix B.4 of the [Folktables paper, Retiring Adult: New Datasets for Fair Machine Learning](https://arxiv.org/pdf/2108.04884). The paper specifies the cutoffs without explaining the exact upper-age choice. Older adults remain in the main population; an ages-18–64 sensitivity check is optional.

The current [Folktables implementation](https://github.com/socialfoundations/folktables/blob/main/folktables/acs.py) distinguishes:

- `ACSEmployment`: 16 source features and no automatic age filter.
- `ACSEmploymentFiltered`: 17 source features, including `GCL`, and the filters above.

**Starting choice: `ACSEmploymentFiltered`.** Record the installed version and assert the exact filters/features. If that variant is unavailable, construct the same definition explicitly. Calling both variants simply “ACSEmployment” without specifying the filters would make our experiment ambiguous.

### Target and prediction

**Review pending:** whether to retain `ESR == 1` or broaden employment to include military and/or temporary-absence codes. The implemented target remains unchanged. See [the processing log](data_processing_log.md#3-target-definition-why-esr4-is-currently-negative) for code meanings and alternative-label counts.

- **True target `Y`:** `1` when `ESR == 1`; `0` for the other valid employment-status codes.
- **Model score `p`:** an estimated probability of the benchmark's positive label.
- **Binary prediction `Y_hat`:** initially `1` when `p >= 0.5`, otherwise `0`. EO later uses its calibrated decision rules.
- **Training objective:** minimize binary cross-entropy against the original `Y`. Both ordinary SGD and DP-SGD learn the same target.

In the [2018 Census data dictionary](https://www2.census.gov/programs-surveys/acs/tech_docs/pums/data_dict/PUMS_Data_Dictionary_2018.txt), `ESR == 1` means **civilian employed and at work**. Code `2` describes employed civilians temporarily away from work, yet maps to `Y=0` under Folktables. Consequently, the negative class includes more than unemployed people. Preserve the benchmark rule and state it precisely in reports. Audit missing/invalid target codes before constructing `Y`.

### Features and protected attributes

The filtered task has these 17 source features. Meanings follow the [Folktables paper, Appendix B.4](https://arxiv.org/pdf/2108.04884), with coding checked against the [2018 Census dictionary](https://www2.census.gov/programs-surveys/acs/tech_docs/pums/data_dict/PUMS_Data_Dictionary_2018.txt).

| Code | Meaning | Primary role |
| --- | --- | --- |
| `AGEP` | Age | Model input |
| `SCHL` | Educational attainment | Model input |
| `MAR` | Marital status | Model input |
| `SEX` | Recorded sex | One-hot predictor in primary policy; protected metadata in both policies |
| `DIS` | Disability status | Model input |
| `ESP` | Parents' employment status | Dropped by user decision (Nov 16): 98.58% missing |
| `MIG` | Mobility status | Model input |
| `CIT` | Citizenship | Model input |
| `MIL` | Military service | Model input |
| `ANC` | Ancestry response recode | Model input |
| `NATIVITY` | Nativity | Model input |
| `RELP` | Relationship to reference person | Model input |
| `DEAR` | Hearing difficulty | Model input |
| `DEYE` | Vision difficulty | Model input |
| `DREM` | Cognitive difficulty | Model input |
| `RAC1P` | Race category | One-hot predictor in primary policy; protected metadata in both policies |
| `GCL` | Grandparents living with grandchildren | Model input |

The primary `include_sensitive` policy has **16 input columns before encoding**, including SEX and RAC1P; the secondary `exclude_sensitive` policy retains the original **14 inputs**. Both preserve sensitive labels separately for evaluation. ESP remains excluded. `ESR`, identifiers, survey weights, and Y are never predictor inputs. `ANC` describes the ancestry response pattern rather than a detailed ancestry identity.

Sex and race are both protected attributes in our evaluation and are included as categorical predictors in the primary controlled comparison. Keep all nine race categories in detailed audits; separately label headline race gaps for groups with at least 500 development observations. Use unweighted benchmark metrics and distinguish them from survey-weighted population estimates. Reference-only encoding yields 87 primary and 77 secondary columns: race code4 is absent from the reference subset and is handled by the existing unknown-category rule, without dropping those records or their audit labels.

## 4. Fairness definitions and metrics

We assess **binary decisions**. Let `A` be the audited group and `Y` the true benchmark label. For a rate `r`, define `gap(r) = max_g r_g - min_g r_g` across the declared comparison groups. Report gaps in percentage points.

### Accuracy parity and disproportionate accuracy loss

Following the motivating paper's accuracy comparisons, report:

- `accuracy_g = number of correct predictions in group g / number in group g`.
- `accuracy_gap = max_g accuracy_g - min_g accuracy_g`.
- `loss_g(epsilon) = accuracy_SGD,g - accuracy_DP,g(epsilon)` on the same evaluation records.

Positive loss means privacy reduced accuracy; negative loss means accuracy improved. Compare the losses across groups, alongside overall accuracy, balanced accuracy, and worst-group accuracy. Equal absolute accuracies and equal losses from adding privacy are different questions.

### Independence, separation, and sufficiency

| Notion | Formal condition | Operational gap |
| --- | --- | --- |
| Independence / demographic parity | `Y_hat` is independent of `A` | `gap(selection rate)` |
| Separation / equalized odds | `Y_hat` is independent of `A` conditional on `Y` | `max(gap(TPR), gap(FPR))` |
| Sufficiency for binary decisions | `Y` is independent of `A` conditional on `Y_hat` | `max(gap(PPV), gap(NPV))` |

These are established statistical criteria; binary sufficiency requires both predictive values. Score-level calibration is a separate analysis. [Fairness and Machine Learning: classification criteria](https://fairmlbook.org/classification.html)

For each group, compute these rates from its confusion matrix:

| Rate | Formula | Meaning |
| --- | --- | --- |
| Selection rate | `(TP + FP) / n` | Fraction predicted positive |
| True-positive rate, TPR | `TP / (TP + FN)` | Fraction of actual positives identified |
| False-positive rate, FPR | `FP / (FP + TN)` | Fraction of actual negatives labeled positive |
| Positive predictive value, PPV | `TP / (TP + FP)` | Reliability of positive predictions |
| Negative predictive value, NPV | `TN / (TN + FN)` | Reliability of negative predictions |

`TP`/`TN` are correct positive/negative predictions; `FP`/`FN` are incorrect positive/negative predictions. Lower gaps indicate closer agreement with the corresponding criterion. The criteria can conflict; report components separately rather than combining all notions into one fairness score. [Definitions and relationships](https://fairmlbook.org/classification.html)

**Primary analyses:** group accuracy losses and separation gaps. **Additional analyses:** independence and binary sufficiency, retaining all three notions from the project proposal. Also report the TPR gap as the equal-opportunity diagnostic. Equal opportunity constrains TPR alone; equalized odds constrains both TPR and FPR. [Hardt et al., Definitions 2.1–2.2](https://arxiv.org/pdf/1610.02413)

Report group sizes, actual-positive rates, and conditional denominators. Flag rates with fewer than 30 denominator observations as unstable; this is a reporting convention, not a guarantee of reliability. Zero-denominator rates are undefined. If a required group rate is undefined, mark the corresponding full-group gap undefined; any supported-subset gap must be labeled separately.

### Which groups and which comparisons?

1. Audit sex and race separately for every model.
2. Add race-by-sex intersections where counts support interpretation.
3. Fit sex-targeted and race-targeted interventions separately in the core experiments; audit both attributes afterward.
4. For standard binary LFR, start with a White-alone/Black-alone race comparison, subject to the initial support audit. Train matched SGD, DP-SGD, and EO controls on that exact cohort. Keep its results separate from full-population results.

Improving a sex-level metric does not establish fairness within every race-by-sex group. These are predictive disparity measurements on observational data; they do not establish a causal explanation for employment differences.

## 5. Interventions and experimental controls

**LFR:** learn a representation balancing reconstruction, prediction, and statistical parity. It targets independence in the representation and does not directly enforce equalized odds. Use original employment labels for downstream training/evaluation; AIF360's returned replacement labels must not become ground truth. Its standard implementation supports one protected attribute and two designated groups. [Zemel et al.](https://proceedings.mlr.press/v28/zemel13.pdf), [AIF360 implementation](https://github.com/Trusted-AI/AIF360/blob/main/aif360/algorithms/preprocessing/lfr.py)

**EO post-processing:** learn group-specific, possibly randomized decision rules from a separate calibration set while keeping the trained MLP fixed. Group membership is needed at prediction time; test-set EO need not exactly match calibration-set EO. [Hardt et al.](https://arxiv.org/pdf/1610.02413)

Compare six families: SGD, DP-SGD, LFR+SGD, LFR+DP-SGD, SGD+EO, and DP-SGD+EO. The non-private intervention controls reveal each intervention's own utility cost.

Controlled settings: MLP hidden widths `[64, 32]`, ReLU, SGD lr0.05/momentum0, 20 epochs, nominal batch512, fixed threshold0.5 and final epoch. Run SGD plus DP epsilon `{1,4,8}` with seeds `[42,123,285,2026,2027]` under both feature policies (40 main runs). Add Poisson-unclipped and Poisson-clipped C=1 noise-free non-private controls for both policies/seeds (20 runs), using DP's expected-batch gradient normalization. Extend important settings to ten seeds after reviewing variability. Interventions and publication figures wait until the controlled baselines are reviewed; EO will reuse suitable checkpoints later.

## 6. Data preparation and privacy scope

Inspect raw missingness and Census not-applicable codes before the library fills values. Keep categorical missingness explicit; start with one-hot categorical encoding and scaled age. Fit learned preprocessing on the reference subset below. Preserve raw data, record exclusion counts, check person identifiers for duplicates, and retain natural group/class imbalance. Identical features alone do not identify duplicate people.

| Partition | Starting share | Purpose |
| --- | --- | --- |
| Public reference | 10% | Fit preprocessing and LFR |
| Model training | 50% | Fit ordinary or private MLP |
| Development | 10% | Choose settings and evaluate pilots |
| EO calibration | 10% | Learn post-processing rules |
| Test | 20% | Final evaluation after settings are frozen |

Save disjoint row assignments and reuse them across methods. Audit group/label support before fixing the splits. Freeze all decisions before accessing final test outcomes.

**Proposed privacy assumption:** auxiliary reference/development/calibration data are treated as public. A DP guarantee applies to individual records in the separate model-training partition, conditional on fixed auxiliary data and partition assignments. Ordinary preprocessing/LFR fitted on protected training records would require separate privacy treatment; private calibration records would too. Multiple released runs require composition accounting. [Dwork and Roth: post-processing and composition](https://www.cis.upenn.edu/~aaroth/Papers/privacybook.pdf)

Use a fixed delta within each cohort, initially `1 / (10 * n_train)`. Log achieved epsilon/delta, sampling, accountant, clipping norm, noise multiplier, steps, batch size, and seeds for each model. A cohort-specific guarantee does not describe the entire experiment sweep.

## 7. Workflow and decisions to check next

Use this proposal as the research-design reference, the [plan](plan.md) as the execution checklist, a data-audit notebook for visible exploration, and reusable Python modules for transformations and metrics. Store chosen settings in configuration files and results in tables. Build the progress/final reports from that recorded evidence.

Before training, check the following in the first notebook:

- Confirm availability of the filtered task, its 17 source columns, and exact target/filter behavior.
- Show retained ages, raw `ESR` counts, missingness, and group/label counts.
- Confirm race-comparison support, the feature-encoding rules, and the proposed public-auxiliary assumption.
- Record any changes here with their reason before running comparisons; reflect them in configurations and tasks.

Milestones: data audit and baseline on Nov 16; DP pilots on Nov 17; progress report on Nov 18; final report on Dec 7. Detailed dates and fallbacks are in [plan.md](plan.md).
