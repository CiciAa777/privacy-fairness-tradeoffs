# Data processing log and review notes

**Prepared November 16, 2026. Status: implemented starting choices, pending your review.**

This records what the code has already done. It is not a claim that every choice has been agreed or that the models have been evaluated. [proposal.md](proposal.md) defines the research design; [plan.md](plan.md) tracks tasks; [the notebook](notebooks/01_data_audit.ipynb) lets you inspect these observations interactively.

## 1. What exists now

The Census CSV is the untouched downloaded source. `cohort.csv` is a filtered pandas-style table saved as CSV. It retains the original category codes and missing values, plus `Y`, `row_id`, and `split`. The encoded matrices are a separate numerical representation for later model training. This log describes preparation; see results/baseline for baseline training status. No LFR transformation or EO rules have been fitted.

## 2. Preparation steps and rules actually applied


| Step | Rule | Observed result |
| --- | --- | --- |
| 1. Download | California, 2018, ACS 1-Year person file | 378,817 raw people; cached locally |
| 2. Check identity | Require unique (SERIALNO, SPORDER) | No duplicate person IDs found; identical feature values are not removed |
| 3. Restrict ages | AGEP > 16, then AGEP < 90 | 72,558 excluded at age <=16; 3,619 at age >=90 |
| 4. Check weights | PWGTP >= 1 | 0 excluded; no missing age/weight values in raw file |
| 5. Check target/groups | Require ESR in 1..6, SEX in 1..2, RAC1P in 1..9 | Assertions passed; unexpected values would stop preparation, not be silently dropped |
| 6. Set target | Y = 1 exactly when ESR == 1 | Benchmark rule retained; alternatives below need a research decision |
| 7. Select inputs | 17 source columns; reserve SEX and RAC1P as protected metadata | 14 direct model inputs after dropping ESP; all people retained |
| 8. Split | Randomly permute eligible row positions using seed 285; assign fixed ratios | Disjoint person-level splits; no label/group stratification or balancing |
| 9. Handle missingness | Keep records; categorical NaN becomes NOT_APPLICABLE in model inputs | No feature-based complete-case deletion or numerical-mean imputation |
| 10. Encode | Reference-fitted StandardScaler for age and OneHotEncoder for 13 other inputs | 77 numerical columns; reference mean/category vocabulary reused for every split |
| 11. Save and inspect | Write matrices, aligned metadata, manifest, and audit tables | Current label/group summaries exclude test records |


**Count reconciliation:** 378,817 raw − 76,177 excluded = 302,640 eligible records. No additional records were removed for missing feature values, group membership, class balance, or suspected outliers. Other categorical codes have not yet all been automatically validated against the complete codebook.

## 3. Target definition: why ESR=4 is currently negative

Folktables defines its positive label as ESR=1. We inherited that benchmark choice; excluding military employment is not a technical requirement of neural networks or privacy.

| ESR | Meaning | Current Y |
| --- | --- | --- |
| 1 | Civilian employed and at work | 1 |
| 2 | Civilian employed with a job, temporarily away | 0 |
| 3 | Unemployed | 0 |
| 4 | Armed Forces at work | 0 |
| 5 | Armed Forces with a job, temporarily away | 0 |
| 6 | Outside the labor force | 0 |

Thus current Y=0 does not mean “unemployed.” A broader employment target could reasonably include code 4, but the treatment of 2 and 5 should be decided at the same time. Here are the alternatives on **non-test records only**:


| Interpretation | Positive ESR codes | Positive records | Positive fraction |
| --- | --- | --- | --- |
| Current Folktables target | [1] | 137,806 | 56.92% |
| At-work civilian or military | [1, 4] | 139,032 | 57.42% |
| Civilian employment, including temporary absence | [1, 2] | 141,656 | 58.51% |
| Civilian or military employment, including temporary absence | [1, 2, 4, 5] | 142,885 | 59.02% |


**Pending decision:** keep the benchmark target as primary, or use a broader target and describe it as modified. Changing it requires regenerating labels/audits and revisiting the proposal; no labels have been changed by this log.

## 4. Every column in cohort.csv

ESP is retained in the audit/codebook only and is absent from cohort.csv and model inputs. Other observed values below refer to the retained cohort for features/metadata; target-code summaries use non-test rows only. The appendix gives the Census code meanings. Numeric codes in categorical columns are category identifiers, not physical measurements.


| Column | Meaning | Use | Observed values |
| --- | --- | --- | --- |
| AGEP | Age | Model input | 17–89 (73 distinct values) |
| SCHL | Educational attainment | Model input | 1–24 (24 distinct values) |
| MAR | Marital status | Model input | 1, 2, 3, 4, 5 |
| SEX | Sex | Protected metadata | 1, 2 |
| DIS | Disability recode | Model input | 1, 2 |
| ESP | Employment status of parents | Dropped by user decision; audit only | 1, 2, 3, 4, 5, 6, 7, 8; missing/N/A also present |
| MIG | Mobility status (lived here 1 year ago) | Model input | 1, 2, 3 |
| CIT | Citizenship status | Model input | 1, 2, 3, 4, 5 |
| MIL | Military service | Model input | 1, 2, 3, 4 |
| ANC | Ancestry recode | Model input | 1, 2, 3, 4 |
| NATIVITY | Nativity | Model input | 1, 2 |
| RELP | Relationship | Model input | 0–17 (18 distinct values) |
| DEAR | Hearing difficulty | Model input | 1, 2 |
| DEYE | Vision difficulty | Model input | 1, 2 |
| DREM | Cognitive difficulty | Model input | 1, 2 |
| RAC1P | Recoded detailed race code | Protected metadata | 1, 2, 3, 4, 5, 6, 7, 8, 9 |
| GCL | Grandparents living with grandchildren | Model input | 1, 2; missing/N/A also present |
| ESR | Original employment code | Construct Y; not an input | 1, 2, 3, 4, 5, 6 |
| PWGTP | Person survey weight | Eligibility; no weighting of metrics yet | 1–1598 (775 distinct values) |
| Y | Binary employment target | Supervised target | 0, 1 |
| row_id | Original loaded row position | Alignment/reproducibility only | 0–378816 (302640 distinct values) |
| split | Assigned data partition | Select records for each purpose | calibration, development, reference, test, train |


`cohort.csv` keeps original missing entries; the model-input transformation handles them separately. Census blank/N/A is distinct from explicit “not reported” or suppressed category codes. We currently label all categorical blanks `NOT_APPLICABLE`; this does not prove the reason for every blank and remains a reviewable assumption. Age is the only input treated as continuous. Education has an order, but the current implementation treats its levels as categories.

## 5. Splits and encoding details


| Split | People | Purpose |
| --- | --- | --- |
| reference | 30264 | Fit preprocessing and later LFR |
| train | 151320 | Train the MLP |
| development | 30264 | Choose settings and inspect pilots |
| calibration | 30264 | Fit EO decision rules later |
| test | 60528 | Final evaluation after choices are frozen |


The 10/50/10/10/20 proportions are our project defaults, not mandated by ACS. The current shuffling is by person, not by household; different members of one household can appear in different splits. It does not guarantee matching group proportions. We should decide whether household-level separation is preferable before freezing the protocol.

The reference set is treated as public auxiliary data in the proposed privacy design. Fitting preprocessing only there supports a training-partition privacy claim. That public-data assumption is also a design choice requiring review.

Age transformation: `encoded_age = (AGEP - 47.121365) / 18.548179`. These statistics came from reference records only. The other inputs become 0/1 category indicators. Missing categories have their own indicator where observed in reference data. Unseen categories map to all-zero indicator blocks because the encoder uses `handle_unknown='ignore'`.


| input | encoded_columns |
| --- | --- |
| AGEP | 1 |
| SCHL | 24 |
| MAR | 5 |
| DIS | 2 |
| MIG | 3 |
| CIT | 5 |
| MIL | 4 |
| ANC | 4 |
| NATIVITY | 2 |
| RELP | 18 |
| DEAR | 2 |
| DEYE | 2 |
| DREM | 2 |
| GCL | 3 |


There are 77 encoded columns per person, expanded from 14 input variables. Sparse matrices are a storage format that avoids storing most zeros. Retained original feature values remain in the CSV; dropped ESP is documented in the audit only.

## 6. Current findings, with interpretation

The full missingness picture below covers all 17 original source features, including dropped ESP. Zero after filtering means no blanks remain in our selected age/weight population; these were not imputed to obtain zero. These describe data preparation, not model performance; baseline outputs are stored separately.


| feature | raw_missing | cohort_missing | cohort_missing_fraction |
| --- | --- | --- | --- |
| AGEP | 0 | 0 | 0.00% |
| SCHL | 10952 | 0 | 0.00% |
| MAR | 0 | 0 | 0.00% |
| SEX | 0 | 0 | 0.00% |
| DIS | 0 | 0 | 0.00% |
| ESP | 305431 | 298334 | 98.58% |
| MIG | 3444 | 0 | 0.00% |
| CIT | 0 | 0 | 0.00% |
| MIL | 72558 | 0 | 0.00% |
| ANC | 0 | 0 | 0.00% |
| NATIVITY | 0 | 0 | 0.00% |
| RELP | 0 | 0 | 0.00% |
| DEAR | 0 | 0 | 0.00% |
| DEYE | 0 | 0 | 0.00% |
| DREM | 18886 | 0 | 0.00% |
| RAC1P | 0 | 0 | 0.00% |
| GCL | 137905 | 65347 | 21.59% |


Non-test sex-group label counts:

| group | n | positives | negatives | positive_rate |
| --- | --- | --- | --- | --- |
| Male | 118642 | 73087 | 45555 | 61.60% |
| Female | 123470 | 64719 | 58751 | 52.42% |


- ESP is missing for almost everyone because its Census applicability is restricted. Dropping complete cases would radically shrink/change the population. User decision recorded Nov 16: drop ESP as a feature and from cohort.csv, retain every eligible person. Its missingness remains in this audit for traceability.
- GCL also has structural N/A. Its applicability is documented in the appendix.
- Sex-group label rates differ. This is a base-rate difference; no model has predicted anything yet.
- Race code 4 (Alaska Native alone) has only 17 non-test records. Fine-grained race and intersectional metrics can be unstable or undefined.
- The following race-by-sex calibration cells have fewer than 30 examples of at least one label, including empty cells. The threshold is a flag, not an automatic deletion rule. These cells would obstruct a full intersectional EO fit; separate sex/race calibration support must also be checked before fitting.


| RAC1P | SEX | n | positives | negatives |
| --- | --- | --- | --- | --- |
| 4 | 1 | 0 | 0 | 0 |
| 4 | 2 | 1 | 1 | 0 |
| 5 | 1 | 37 | 16 | 21 |
| 5 | 2 | 33 | 21 | 12 |
| 7 | 1 | 42 | 23 | 19 |
| 7 | 2 | 47 | 24 | 23 |


## 7. Choices for your review before baseline training

| Choice | What was implemented | Review question |
| --- | --- | --- |
| Target | ESR=1 only | Benchmark compatibility or broader civilian/military employment? |
| Population | Ages 17–89 | Keep the paper's population or prioritize a working-age cohort? |
| Features | 14 inputs after dropping ESP; sex/race metadata | Should protected attributes be direct inputs in a comparison? |
| Education | One-hot categories | Keep this simple choice or test an ordinal encoding? |
| Missing ESP/GCL | ESP dropped; GCL retains explicit missing category; all people retained | ESP decision agreed Nov 16; GCL remains for the initial baseline |
| Splitting | Fixed random person-level assignments | Keep these ratios or change them; separate households? |
| Auxiliary data | Reference/development/calibration treated as public | Is this acceptable for the intended privacy claim? |
| Small groups | Retained and flagged | Which supported primary comparisons should be prespecified? |
| Survey weights | Eligibility only | Unweighted benchmark analysis or additional weighted estimates? |

Record your preferred changes below. We will then change the specification and regenerate outputs consistently, rather than manually editing separate CSVs.

- Decisions agreed: Drop ESP (Nov 16); keep all 302,640 eligible people; inputs decrease from 15 to 14 and encoded dimension from 86 to 77. Labels and split assignments unchanged.
- Changes requested:
- Questions remaining:

## 8. scripts versus src, and how to reproduce this log

`src/data.py` contains reusable functions: the preparation recipe and group-summary helpers. `scripts/prepare_data.py` is a short entry point that calls that recipe. The notebook imports helpers from `src`; it does not perform another independent download/preparation. These folder names are conventions for organizing Python code, not different kinds of data.

```powershell
# Re-run the preparation recipe (changes outputs only if its rules/data change).
.\.venv\Scripts\python.exe scripts\prepare_data.py
# Regenerate this factual log from the saved data and official codebook.
.\.venv\Scripts\python.exe scripts\write_data_processing_log.py
```

The raw Census file, cohort, labels, and split assignments were only read while creating this log. To preserve your review notes, copy them to the proposal before regenerating this report.

## Appendix: official category meanings

The entries below reproduce the relevant public Census codebook sections. Blank codes (`b`, `bb`, etc.) become missing values when the source CSV is loaded. For columns like age and weights, ranges describe permitted values rather than one category per value. Standalone numeric codes for other columns are categorical. Observed values can be a subset of the official possibilities.

Sources: [2018 Census codebook](https://www2.census.gov/programs-surveys/acs/tech_docs/pums/data_dict/PUMS_Data_Dictionary_2018.txt), [Folktables task/filter implementation](https://github.com/socialfoundations/folktables/blob/main/folktables/acs.py). Exact downloaded source: `data/raw/PUMS_Data_Dictionary_2018.txt`.



### AGEP

```text
Age
            0      .Under 1 year
            1..99  .1 to 99 years (Top-coded)
```



### SCHL

```text
Educational attainment
            bb     .N/A (less than 3 years old)
            01     .No schooling completed
            02     .Nursery school, preschool
            03     .Kindergarten
            04     .Grade 1
            05     .Grade 2
            06     .Grade 3
            07     .Grade 4
            08     .Grade 5
            09     .Grade 6
            10     .Grade 7
            11     .Grade 8
            12     .Grade 9
            13     .Grade 10
            14     .Grade 11
            15     .12th grade - no diploma
            16     .Regular high school diploma
            17     .GED or alternative credential
            18     .Some college, but less than 1 year
            19     .1 or more years of college credit, no degree
            20     .Associate's degree
            21     .Bachelor's degree
            22     .Master's degree
            23     .Professional degree beyond a bachelor's degree
            24     .Doctorate degree
```



### MAR

```text
Marital status
            1    .Married
            2    .Widowed
            3    .Divorced
            4    .Separated
            5    .Never married or under 15 years old
```



### SEX

```text
Sex
            1    .Male
            2    .Female
```



### DIS

```text
Disability recode
            1    .With a disability
            2    .Without a disability
```



### ESP

```text
Employment status of parents
            b    .N/A (not own child of householder, and not child in subfamily) Living with two parents:
            1    .Both parents in labor force
            2    .Father only in labor force
            3    .Mother only in labor force
            4    .Neither parent in labor force living with one parent: Living with father:
            5    .Father in the labor force
            6    .Father not in labor force living with mother:
            7    .Mother in the labor force
            8    .Mother not in labor force
```



### MIG

```text
Mobility status (lived here 1 year ago)
            b    .N/A (less than 1 year old)
            1    .Yes, same house (nonmovers)
            2    .No, outside US and Puerto Rico
            3    .No, different house in US or Puerto Rico
```



### CIT

```text
Citizenship status
            1    .Born in the U.S.
            2    .Born in Puerto Rico, Guam, the U.S. Virgin Islands, or the Northern Marianas
            3    .Born abroad of American parent(s)
            4    .U.S. citizen by naturalization
            5    .Not a citizen of the U.S.
```



### MIL

```text
Military service
            b    .N/A (less than 17 years old)
            1    .Now on active duty
            2    .On active duty in the past, but not now
            3    .Only on active duty for training in Reserves/National Guard
            4    .Never served in the military
```



### ANC

```text
Ancestry recode
            1    .Single
            2    .Multiple
            3    .Unclassified
            4    .Not reported
            8    .Suppressed for data year 2018 for select PUMAs
```



### NATIVITY

```text
Nativity
            1    .Native
            2    .Foreign born
```



### RELP

```text
Relationship
            00     .Reference person
            01     .Husband/wife
            02     .Biological son or daughter
            03     .Adopted son or daughter
            04     .Stepson or stepdaughter
            05     .Brother or sister
            06     .Father or mother
            07     .Grandchild
            08     .Parent-in-law
            09     .Son-in-law or daughter-in-law
            10     .Other relative
            11     .Roomer or boarder
            12     .Housemate or roommate
            13     .Unmarried partner
            14     .Foster child
            15     .Other nonrelative
            16     .Institutionalized group quarters population
            17     .Noninstitutionalized group quarters population
```



### DEAR

```text
Hearing difficulty
            1    .Yes
            2    .No
```



### DEYE

```text
Vision difficulty
            1    .Yes
            2    .No
```



### DREM

```text
Cognitive difficulty
            b    .N/A (Less than 5 years old)
            1    .Yes
            2    .No
```



### RAC1P

```text
Recoded detailed race code
            1    .White alone
            2    .Black or African American alone
            3    .American Indian alone
            4    .Alaska Native alone
            5    .American Indian and Alaska Native tribes specified; or American Indian or Alaska Native, not specified and no other races
            6    .Asian alone
            7    .Native Hawaiian and Other Pacific Islander alone
            8    .Some Other Race alone
            9    .Two or More Races
```



### GCL

```text
Grandparents living with grandchildren
            b    .N/A (less than 30 years/institutional GQ)
            1    .Yes
            2    .No
```



### ESR

```text
Employment status recode
            b    .N/A (less than 16 years old)
            1    .Civilian employed, at work
            2    .Civilian employed, with a job but not at work
            3    .Unemployed
            4    .Armed forces, at work
            5    .Armed forces, with a job but not at work
            6    .Not in labor force
```



### PWGTP

```text
Person's weight
            1..9999      .Integer weight of person
```
