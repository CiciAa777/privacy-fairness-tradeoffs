# Project log

Updated November 18, 2026. This is the single record of data processing, experimental settings, decisions and completed work. Read [findings.md](findings.md) for results and [plan.md](plan.md) for next steps.

## 1. Data and processing rules

| Item | Implemented rule |
| --- | --- |
| Source | California 2018 ACS 1-Year person data, Folktables filtered employment task |
| Cohort | `16 < AGEP < 90`, `PWGTP >= 1`: 302,640 of 378,817 records; 76,177 excluded |
| Identity | Check `SERIALNO` + `SPORDER` uniqueness; use original row position as `row_id` for alignment |
| Target | `Y=1` exactly for `ESR=1`; validate ESR in 1..6 |
| Predictors excluded | ESR, ESP, Y, identifiers, row_id, split, PWGTP |
| Weighting | PWGTP is an eligibility check; training/evaluation are unweighted |
| Missingness | Drop ESP, retain people; encode categorical blanks as `NOT_APPLICABLE`; no row dropping for GCL |
| Encoding | Standardize age; one-hot all remaining inputs as categories; fit on reference only |
| Unseen categories | `OneHotEncoder(handle_unknown='ignore')`: all-zero block for a reference-unseen level |
| Splitting | Person-level permutation, NumPy `default_rng(285)`, fixed ratio assignment; not stratified or household-disjoint |

**Target meaning matters:** ESR1=civilian employed and at work; ESR2=civilian employed but temporarily absent; ESR3=unemployed; ESR4=armed forces at work; ESR5=armed forces temporarily absent; ESR6=not in labor force. Thus Y=0 does not mean only unemployed. Broadening the label would be a separate experiment; the current target is unchanged.

### Features and observed codes

| Field | Meaning | Codes / treatment |
| --- | --- | --- |
| AGEP | Age | 17-89; continuous |
| SCHL | Education | 1-24; categorical |
| MAR | Marital status | 1-5; categorical |
| SEX | Recorded sex | 1=male, 2=female; categorical/protected |
| DIS | Disability | 1-2 |
| ESP | Parents' employment | Dropped: 98.58% cohort missing |
| MIG | Mobility | 1-3 |
| CIT | Citizenship | 1-5 |
| MIL | Military service | 1-4 |
| ANC | Ancestry response recode | 1-4; not detailed ancestry identity |
| NATIVITY | Nativity | 1-2 |
| RELP | Relationship | 0-17 |
| DEAR / DEYE / DREM | Hearing / vision / cognitive difficulty | 1-2 each |
| RAC1P | Race | Original nine categories retained |
| GCL | Grandchildren living with grandparent | 1-2 plus missing category |

Race codes: 1 White alone; 2 Black or African American alone; 3 American Indian alone; 4 Alaska Native alone; 5 American Indian/Alaska Native specified or unspecified, no other races; 6 Asian alone; 7 Native Hawaiian/Other Pacific Islander alone; 8 Some Other Race alone; 9 Two or More Races. No categories were merged.

The two feature policies use the same people, labels and splits:

- **Primary `include_sensitive`:** 16 raw predictors including SEX/RAC1P, 87 encoded columns.
- **Secondary `exclude_sensitive`:** 14 predictors without SEX/RAC1P, 77 encoded columns. Protected labels remain available for auditing.

Reference data contains no RAC1P4 level. Its encoder block follows the unknown-category rule; those people remain in evaluation. Encoded dimension counts input indicators, not people or independent variables.

### Missingness audit

| Feature | Raw missing | Cohort missing | Cohort % |
| --- | ---: | ---: | ---: |
| AGEP | 0 | 0 | 0 |
| SCHL | 10,952 | 0 | 0 |
| MAR / SEX / DIS | 0 | 0 | 0 |
| ESP | 305,431 | 298,334 | 98.58 |
| MIG | 3,444 | 0 | 0 |
| CIT | 0 | 0 | 0 |
| MIL | 72,558 | 0 | 0 |
| ANC / NATIVITY / RELP / DEAR / DEYE | 0 | 0 | 0 |
| DREM | 18,886 | 0 | 0 |
| RAC1P | 0 | 0 | 0 |
| GCL | 137,905 | 65,347 | 21.59 |

Only ESP and GCL have blanks after cohort filtering. Other raw missingness occurs outside the selected cohort; zeros above were not produced by imputation. `NOT_APPLICABLE` is our encoding convention, not a verified explanation of every blank. The original per-column counts are retained in [input_missingness.csv](results/controlled_v2/audit/input_missingness.csv).

### Fixed partitions

| Partition | People | Purpose |
| --- | ---: | --- |
| Reference, 10% | 30,264 | Fit preprocessing; later public-reference interventions |
| Training, 50% | 151,320 | Train models |
| Development, 10% | 30,264 | Current results and model/settings inspection |
| Calibration, 10% | 30,264 | Later EO decision rules |
| Test, 20% | 60,528 | Final evaluation after choices are frozen |

Do not regenerate prepared splits over existing study data. Household overlap exists under these fixed person-level splits; audit it without silently changing the cohort. Test outcomes remain unevaluated.

## 2. Completed experiments and exact settings

| Setting | Controlled v2 |
| --- | --- |
| Model / loss | Input -> 64 -> 32 -> 1, ReLU; BCEWithLogitsLoss; sigmoid for evaluation |
| Optimizer | SGD, learning rate 0.05, momentum 0 |
| Training / selection | 20 epochs, nominal batch 512, fixed last epoch; threshold 0.5 |
| Initialization seeds | 42, 123, 285, 2026, 2027; matched across methods within each policy |
| Main grid | 2 policies x 5 seeds x (SGD, DP1, DP4, DP8) = **40 runs completed** |
| Diagnostic grid | 2 policies x 5 seeds x (Poisson unclipped, Poisson clipped/no noise) = **20 runs completed** |
| Ordinary SGD | Shuffled batches; actual-batch mean gradient |
| DP / Poisson controls | Poisson sampling q=1/296; 5,920 updates; expected-batch normalizer 511 |
| Clipping | Per-example full-model L2 clipping, C=1 |
| Noise / accounting | Opacus 1.6.0, RDP; Gaussian noise on clipped sums, followed by expected-batch normalization |
| Delta | 1/(10*151320) = 6.608511763150938e-7 |
| Epsilon target -> achieved; sigma | 1 -> 0.999160; 1.411133 / 4 -> 3.999411; 0.740509 / 8 -> 7.999651; 0.597458 |
| Randomness | Public initialization seed; fresh private sampling/noise entropy; research `secure_mode=False` |
| Controls | Non-private, no privacy guarantee; clipped/unclipped controls share minibatch streams |

Completed evidence includes 12 public-reference smoke configurations, 13 unit tests, 30 accountant checks and 12 representative checkpoint/prediction reconstructions. Manual clipping-only updates match the zero-noise Opacus update. No private RNG states or raw per-example gradients are exported.

**Main outcome:** unequal positive-prediction shifts and group accuracy losses; contrasting sex/race fairness effects; the large accuracy reduction is already present in the clipping-only control. Budget effects at fixed C=1 are much smaller, but not absent. The next priority is clipping-norm sensitivity and cross-task robustness, not a wider epsilon sweep. See [findings.md](findings.md).

## 3. Metrics, units and data locations

- Accuracy loss = matched SGD accuracy minus method accuracy: positive means harm. Pair within policy/group/initialization before averaging.
- Accuracy parity = max-min group accuracy; independence = max-min selection rate; separation = max(TPR range,FPR range); binary sufficiency = max(PPV range,NPV range). Smaller gaps mean closer parity, not necessarily better group outcomes.
- TPR=TP/(TP+FN), FPR=FP/(FP+TN), PPV=TP/(TP+FP), NPV=TN/(TN+FN). Binary sufficiency concerns thresholded predictions, not full score calibration.
- Gaps are calculated per seed before averaging. Mean +/- sample SD uses ddof=1; it is training variation on a fixed split, not a confidence interval.
- All nine races and 18 race-by-sex intersections remain in CSVs. Headline race gaps use the separately labeled fixed five groups with development n>=500. Flag conditional denominators below30; preserve undefined rates, never replace them with zero.

Machine-readable records remain under `results/controlled_v2/`: `runs/` contains per-run summaries/counts/history, `aggregate/` contains per-seed/mean-SD/paired/group/ESR tables, `audit/` contains data/accounting checks, and `report_assets/` contains publication figures and their numeric source tables. Raw aggregate rates are fractions (multiply by100); columns ending `_pp` are already percentage points. Report-assets tables explicitly use % or pp. `group_rate_changes` is method minus SGD; `group_accuracy_losses` reverses that sign.

## 4. Privacy assumptions and limits

DP claims apply per model to individual training records conditional on fixed public reference/development/calibration data. Multiple private releases compose. Baselines, diagnostic controls, audits and this repository are not an end-to-end DP release; the guarantees are not household-level. Smaller epsilon is stronger privacy at fixed delta. At fixed sampling/steps, sigma is calibrated by the accountant; absolute Gaussian SD on the clipped sum is sigma*C.

The old three private pilot releases had basic-composition epsilon 12.99822154032934 and delta 1.9825535289452814e-6. Their cost does not disappear when old files are deleted. [historical_pilot_reference.json](configs/historical_pilot_reference.json) preserves the bounds and the seed285 exclusion-baseline accuracy needed by active checks. Current aggregate composition includes separate v2-only and joint historical bounds.

No LFR, EO, ACSIncome, clipping-norm sweep, ten-seed expansion, or final test evaluation has been completed yet.

## 5. Decisions and changes

| Date / checkpoint | What changed |
| --- | --- |
| Nov 16 | Prepared filtered CA2018 employment data; dropped ESP; retained GCL missingness and benchmark ESR==1 target |
| Nov 17, `c790371` | Archived the single-seed pilots before controlled experiments |
| Nov 17-18, `c3e9ad5` | Completed 40 main and 20 control runs, two feature policies, five seeds |
| Nov 18, `fbcc0cc` / `dfb290a` | Added paired group-rate analysis and verified report figures/tables |
| Nov 18, `1ee6df6` | Created the original two-page progress-report source and package |
| Nov 18, `21d50cf` | Saved the team's downloaded Overleaf edits unchanged |
| Nov 18, cleanup | Consolidated notes into this log and findings.md; removed V0/obsolete generators; made C sensitivity and ACSIncome the next priorities |

Tracked history, including the original proposal, processing codebook, reports and V0 settings, is recoverable from `21d50cf` (for example `git show 21d50cf:V0/SETTINGS.txt`, or inspect that commit on GitHub). No history was rewritten. Eight ignored V0 checkpoints/prediction files, which Git cannot recover, were SHA-256 checked into the local backup `E:/UCSD/ece/285/output/privacy-fairness-local-backups/v0_ignored_2026-11-18.zip` before removing V0. Current data, 60-run results and checkpoints remain in place.

The downloaded Overleaf ZIP is preserved byte-for-byte; its source was imported locally and compiled. The user-edited version currently produces **three pages**, contains a negative title offset and a small-font methods table. Those edits were preserved rather than silently reformatted. The PDF/source/ZIP are under `reports/progress_report/`; the original two-page version remains in history.

References: [Bagdasaryan et al.](https://proceedings.neurips.cc/paper/2019/hash/fc0de4e0396fff257ea362983c2dda5a-Abstract.html), [Folktables](https://arxiv.org/abs/2108.04884), [LFR](https://proceedings.mlr.press/v28/zemel13.html), [EO](https://arxiv.org/abs/1610.02413), and the [2018 Census codebook](https://www2.census.gov/programs-surveys/acs/tech_docs/pums/data_dict/PUMS_Data_Dictionary_2018.txt).
