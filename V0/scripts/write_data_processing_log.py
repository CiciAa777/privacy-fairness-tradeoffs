"""Document existing preparation outputs without changing labels, splits, or inputs."""
from pathlib import Path
import json
import re
import joblib
import pandas as pd
import nbformat

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data/processed'
AUDIT = ROOT / 'results/data_audit'
manifest = json.loads((DATA / 'manifest.json').read_text())
df = pd.read_csv(DATA / 'cohort.csv')
visible = df.loc[df['split'].ne('test')]
raw = pd.read_csv(ROOT / 'data/raw/2018/1-Year/psam_p06.csv',
                  usecols=list(set(manifest['source_features'] + ['PWGTP', 'SERIALNO', 'SPORDER'])))
dictionary = (ROOT / 'data/raw/PUMS_Data_Dictionary_2018.txt').read_text(encoding='utf-8-sig')
headers = list(re.finditer(r'^([A-Z][A-Z0-9]*)\s+(?:Character|Numeric)\s+\d+\s*$', dictionary, re.M))
sections = {h.group(1): dictionary[h.end():headers[i+1].start() if i+1<len(headers) else len(dictionary)].strip()
            for i,h in enumerate(headers)}

def table(frame):
    def cell(x): return str(x).replace('|', '\\|').replace('\n', ' ')
    return '\n'.join(['| ' + ' | '.join(map(cell, frame.columns)) + ' |',
                      '| ' + ' | '.join(['---'] * len(frame.columns)) + ' |'] +
                     ['| ' + ' | '.join(map(cell, row)) + ' |' for row in frame.itertuples(index=False, name=None)])

def values(series):
    items = sorted(series.dropna().unique())
    if not items: return 'None'
    if pd.api.types.is_numeric_dtype(series):
        if len(items)>10:
            result = f'{min(items):g}–{max(items):g} ({len(items)} distinct values)'
        else: result = ', '.join(f'{x:g}' for x in items)
    else: result = ', '.join(str(x) for x in items)
    return result + ('; missing/N/A also present' if series.isna().any() else '')

pre = joblib.load(DATA / 'preprocessor.joblib')
age = pre.named_transformers_['age']
encoding = pd.DataFrame({'input': manifest['model_inputs'], 'encoded_columns': [
    1 if c=='AGEP' else len(pre.named_transformers_['categories'].categories_[
        [x for x in manifest['model_inputs'] if x!='AGEP'].index(c)]) for c in manifest['model_inputs']]})
parts = ['''# Data processing log and review notes

**Prepared November 16, 2026. Status: implemented starting choices, pending your review.**

This records what the code has already done. It is not a claim that every choice has been agreed or that the models have been evaluated. [proposal.md](proposal.md) defines the research design; [plan.md](plan.md) tracks tasks; [the notebook](notebooks/01_data_audit.ipynb) lets you inspect these observations interactively.

## 1. What exists now

The Census CSV is the untouched downloaded source. `cohort.csv` is a filtered pandas-style table saved as CSV. It retains the original category codes and missing values, plus `Y`, `row_id`, and `split`. The encoded matrices are a separate numerical representation for later model training. This log describes preparation; see results/baseline for baseline training status. No LFR transformation or EO rules have been fitted.

## 2. Preparation steps and rules actually applied
''']
steps = [
['1. Download', 'California, 2018, ACS 1-Year person file', f'{len(raw):,} raw people; cached locally'],
['2. Check identity', 'Require unique (SERIALNO, SPORDER)', 'No duplicate person IDs found; identical feature values are not removed'],
['3. Restrict ages', 'AGEP > 16, then AGEP < 90', f'{raw.AGEP.le(16).sum():,} excluded at age <=16; {raw.AGEP.ge(90).sum():,} at age >=90'],
['4. Check weights', 'PWGTP >= 1', f'{raw.PWGTP.lt(1).sum():,} excluded; no missing age/weight values in raw file'],
['5. Check target/groups', 'Require ESR in 1..6, SEX in 1..2, RAC1P in 1..9', 'Assertions passed; unexpected values would stop preparation, not be silently dropped'],
['6. Set target', 'Y = 1 exactly when ESR == 1', 'Benchmark rule retained; alternatives below need a research decision'],
['7. Select inputs', '17 source columns; reserve SEX and RAC1P as protected metadata', '14 direct model inputs after dropping ESP; all people retained'],
['8. Split', 'Randomly permute eligible row positions using seed 285; assign fixed ratios', 'Disjoint person-level splits; no label/group stratification or balancing'],
['9. Handle missingness', 'Keep records; categorical NaN becomes NOT_APPLICABLE in model inputs', 'No feature-based complete-case deletion or numerical-mean imputation'],
['10. Encode', 'Reference-fitted StandardScaler for age and OneHotEncoder for 13 other inputs', '77 numerical columns; reference mean/category vocabulary reused for every split'],
['11. Save and inspect', 'Write matrices, aligned metadata, manifest, and audit tables', 'Current label/group summaries exclude test records']]
parts.append(table(pd.DataFrame(steps,columns=['Step','Rule','Observed result'])))
parts.append(f'''\n**Count reconciliation:** {len(raw):,} raw − {len(raw)-len(df):,} excluded = {len(df):,} eligible records. No additional records were removed for missing feature values, group membership, class balance, or suspected outliers. Other categorical codes have not yet all been automatically validated against the complete codebook.

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
''')
alternatives=[]
for label,codes in [('Current Folktables target',[1]),('At-work civilian or military',[1,4]),('Civilian employment, including temporary absence',[1,2]),('Civilian or military employment, including temporary absence',[1,2,4,5])]:
    n=int(visible.ESR.isin(codes).sum())
    alternatives.append([label,str(codes),f'{n:,}',f'{n/len(visible):.2%}'])
parts.append(table(pd.DataFrame(alternatives,columns=['Interpretation','Positive ESR codes','Positive records','Positive fraction'])))
parts.append('''\n**Pending decision:** keep the benchmark target as primary, or use a broader target and describe it as modified. Changing it requires regenerating labels/audits and revisiting the proposal; no labels have been changed by this log.

## 4. Every column in cohort.csv

ESP is retained in the audit/codebook only and is absent from cohort.csv and model inputs. Other observed values below refer to the retained cohort for features/metadata; target-code summaries use non-test rows only. The appendix gives the Census code meanings. Numeric codes in categorical columns are category identifiers, not physical measurements.
''')
rows=[]
for c in manifest['source_features']:
    rows.append([c, sections[c].splitlines()[0], 'Dropped by user decision; audit only' if c == 'ESP' else ('Protected metadata' if c in ['SEX','RAC1P'] else 'Model input'), values(df[c] if c in df else raw.loc[raw.AGEP.gt(16) & raw.AGEP.lt(90) & raw.PWGTP.ge(1), c])])
for c,meaning,role in [('ESR','Original employment code','Construct Y; not an input'),('PWGTP','Person survey weight','Eligibility; no weighting of metrics yet'),('Y','Binary employment target','Supervised target'),('row_id','Original loaded row position','Alignment/reproducibility only'),('split','Assigned data partition','Select records for each purpose')]:
    rows.append([c,meaning,role,values(visible[c] if c in ['ESR','Y'] else df[c])])
parts.append(table(pd.DataFrame(rows,columns=['Column','Meaning','Use','Observed values'])))
parts.append('''\n`cohort.csv` keeps original missing entries; the model-input transformation handles them separately. Census blank/N/A is distinct from explicit “not reported” or suppressed category codes. We currently label all categorical blanks `NOT_APPLICABLE`; this does not prove the reason for every blank and remains a reviewable assumption. Age is the only input treated as continuous. Education has an order, but the current implementation treats its levels as categories.

## 5. Splits and encoding details
''')
purposes={'reference':'Fit preprocessing and later LFR', 'train':'Train the MLP','development':'Choose settings and inspect pilots','calibration':'Fit EO decision rules later','test':'Final evaluation after choices are frozen'}
parts.append(table(pd.DataFrame([[s,manifest['split_counts'][s],purposes[s]] for s in ['reference','train','development','calibration','test']],columns=['Split','People','Purpose'])))
parts.append(f'''\nThe 10/50/10/10/20 proportions are our project defaults, not mandated by ACS. The current shuffling is by person, not by household; different members of one household can appear in different splits. It does not guarantee matching group proportions. We should decide whether household-level separation is preferable before freezing the protocol.

The reference set is treated as public auxiliary data in the proposed privacy design. Fitting preprocessing only there supports a training-partition privacy claim. That public-data assumption is also a design choice requiring review.

Age transformation: `encoded_age = (AGEP - {age.mean_[0]:.6f}) / {age.scale_[0]:.6f}`. These statistics came from reference records only. The other inputs become 0/1 category indicators. Missing categories have their own indicator where observed in reference data. Unseen categories map to all-zero indicator blocks because the encoder uses `handle_unknown='ignore'`.
''')
parts.append(table(encoding))
assert encoding.encoded_columns.sum()==manifest['encoded_dimension']
parts.append('''\nThere are 77 encoded columns per person, expanded from 14 input variables. Sparse matrices are a storage format that avoids storing most zeros. Retained original feature values remain in the CSV; dropped ESP is documented in the audit only.

## 6. Current findings, with interpretation

The full missingness picture below covers all 17 original source features, including dropped ESP. Zero after filtering means no blanks remain in our selected age/weight population; these were not imputed to obtain zero. These describe data preparation, not model performance; baseline outputs are stored separately.
''')
missing=pd.read_csv(AUDIT/'missingness.csv')
parts.append(table(missing.assign(cohort_missing_fraction=lambda x:x.cohort_missing_fraction.map(lambda z:f'{z:.2%}'))))
sex=pd.read_csv(AUDIT/'sex.csv');race=pd.read_csv(AUDIT/'race.csv')
sex['group']=sex.SEX.map({1:'Male',2:'Female'})
sex['positive_rate']=sex.positive_rate.map(lambda x:f'{x:.2%}')
parts.append('\nNon-test sex-group label counts:\n\n'+table(sex[['group','n','positives','negatives','positive_rate']]))
support=pd.read_csv(AUDIT/'split_groups.csv')
small=support.loc[support['split'].eq('calibration') & support[['positives','negatives']].min(axis=1).lt(30)]
parts.append('''\n- ESP is missing for almost everyone because its Census applicability is restricted. Dropping complete cases would radically shrink/change the population. User decision recorded Nov 16: drop ESP as a feature and from cohort.csv, retain every eligible person. Its missingness remains in this audit for traceability.
- GCL also has structural N/A. Its applicability is documented in the appendix.
- Sex-group label rates differ. This is a base-rate difference; no model has predicted anything yet.
- Race code 4 (Alaska Native alone) has only 17 non-test records. Fine-grained race and intersectional metrics can be unstable or undefined.
- The following race-by-sex calibration cells have fewer than 30 examples of at least one label, including empty cells. The threshold is a flag, not an automatic deletion rule. These cells would obstruct a full intersectional EO fit; separate sex/race calibration support must also be checked before fitting.
''')
parts.append(table(small[['RAC1P','SEX','n','positives','negatives']]))
parts.append('''\n## 7. Choices for your review before baseline training

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
.\\.venv\\Scripts\\python.exe scripts\\prepare_data.py
# Regenerate this factual log from the saved data and official codebook.
.\\.venv\\Scripts\\python.exe scripts\\write_data_processing_log.py
```

The raw Census file, cohort, labels, and split assignments were only read while creating this log. To preserve your review notes, copy them to the proposal before regenerating this report.

## Appendix: official category meanings

The entries below reproduce the relevant public Census codebook sections. Blank codes (`b`, `bb`, etc.) become missing values when the source CSV is loaded. For columns like age and weights, ranges describe permitted values rather than one category per value. Standalone numeric codes for other columns are categorical. Observed values can be a subset of the official possibilities.

Sources: [2018 Census codebook](https://www2.census.gov/programs-surveys/acs/tech_docs/pums/data_dict/PUMS_Data_Dictionary_2018.txt), [Folktables task/filter implementation](https://github.com/socialfoundations/folktables/blob/main/folktables/acs.py). Exact downloaded source: `data/raw/PUMS_Data_Dictionary_2018.txt`.
''')
for c in manifest['source_features']+['ESR','PWGTP']:
    block=sections[c].split('\n\nPERSON RECORD')[0]
    parts.append(f'\n### {c}\n\n```text\n{block}\n```\n')
(ROOT/'data_processing_log.md').write_text('\n\n'.join(parts).rstrip()+'\n',encoding='utf-8')
path=ROOT/'notebooks/01_data_audit.ipynb'
nb=nbformat.read(path,as_version=4)
link='\n\n**Processing record:** [data_processing_log.md](../data_processing_log.md) documents every rule, column/code, finding, and pending choice for review.'
if '../data_processing_log.md' not in nb.cells[0].source:
    nb.cells[0].source+=link
    nbformat.write(nb,path)
print('Created data_processing_log.md and linked it from the notebook. Dataset unchanged.')
