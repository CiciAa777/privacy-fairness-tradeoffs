"""Create an inspectable notebook with short explanations and review prompts."""
from pathlib import Path
import nbformat as nbf

root = Path(__file__).resolve().parents[1]
nb = nbf.v4.new_notebook()
cells = []
def md(text): cells.append(nbf.v4.new_markdown_cell(text))
def code(text): cells.append(nbf.v4.new_code_cell(text))

md('''# 01 — Inspect our ACS employment data

Select the project's `.venv/Scripts/python.exe` as the notebook kernel, then run cells from the top. These cells read the already prepared files; they do not download again.

**What to learn:** who is included, what the label means, which groups are small, and which missing values are structural. Each section ends with a question to consider. Test outcomes stay reserved for final evaluation.

`PWGTP` is a person survey weight: a sampled record represents roughly that many people in population estimates. A fictional weight of 100 means approximately 100 represented people, not 100 duplicate rows. We require positive weights but initially train/evaluate with equal weight per sampled record.

References: [2018 Census dictionary](https://www2.census.gov/programs-surveys/acs/tech_docs/pums/data_dict/PUMS_Data_Dictionary_2018.txt), [research design](../proposal.md).''')
code('''from pathlib import Path
import sys, json
import numpy as np
import pandas as pd
from scipy import sparse
from IPython.display import display

ROOT = next(p for p in [Path.cwd(), *Path.cwd().parents] if (p / 'proposal.md').exists())
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from src.data import FEATURES, INPUTS, CATEGORICAL, group_summary, model_frame
DATA = ROOT / 'data' / 'processed'
AUDIT = ROOT / 'results' / 'data_audit'
manifest = json.loads((DATA / 'manifest.json').read_text())
print('Python:', sys.executable)
print('Project:', ROOT)''')
md('''## 1. Who is in the study?

The eligibility rule is **age > 16, age < 90, and PWGTP >= 1**. This keeps ages 17–89, including older adults. Split assignment uses a fixed random seed and is independent of labels and protected groups.

**Inspect:** How many records did filtering remove? How large is the training set?''')
code('''display(pd.Series({k: manifest[k] for k in ['raw_rows', 'eligible_rows', 'excluded_rows', 'encoded_dimension']}, name='value').to_frame())
display(pd.Series(manifest['split_counts'], name='records').to_frame())
cohort = pd.read_csv(DATA / 'cohort.csv')
visible = cohort.loc[cohort['split'].ne('test')].copy()
assert visible.AGEP.between(17, 89).all()
assert visible.PWGTP.ge(1).all()
display(visible[[c for c in FEATURES if c in visible.columns] + ['ESR', 'Y', 'split']].head(8))''')
md('''## 2. What does the target actually mean?

`Y = 1` exactly when `ESR = 1`. Crucially, `Y = 0` includes employed people temporarily away from work, as well as unemployed people and people outside the labor force. It is the Folktables benchmark definition.

**Inspect:** Which ESR categories contribute most negative labels? Why would calling all negatives “unemployed” be inaccurate?''')
code('''esr_names = {1: 'Civilian employed, at work', 2: 'Civilian employed, away from work',
             3: 'Unemployed', 4: 'Armed forces, at work',
             5: 'Armed forces, away from work', 6: 'Not in labor force'}
target = visible.groupby('ESR').agg(n=('Y', 'size'), label=('Y', 'first'))
target['meaning'] = target.index.map(esr_names)
display(target)
print('Positive-label fraction in non-test records:', round(visible.Y.mean(), 4))
age_bands = pd.cut(visible.AGEP, [16, 24, 44, 64, 89], labels=['17–24', '25–44', '45–64', '65–89'])
display(visible.groupby(age_bands, observed=True).Y.agg(n='size', positive_rate='mean'))''')
md('''## 3. What is PWGTP doing?

Positive weights determine eligibility. They are excluded from model features. Our current positive rates are unweighted: they describe this sample, rather than survey-weighted California population estimates.

**Inspect:** Do the weights vary substantially? Distinguish filtering on a weight from using it to weight training or metrics.''')
code('''display(visible.PWGTP.describe().to_frame())
print('PWGTP in model inputs?', 'PWGTP' in INPUTS)''')
md('''## 4. Which values are missing?

Some Census fields are not applicable to certain people. For example, parents' employment status and grandparent fields have specific applicability rules. We retain these records, drop ESP as agreed, and encode remaining categorical N/A explicitly, instead of deleting every row with an empty field.

**Inspect:** Which fields have the most missingness? Check their applicability in the Census dictionary before proposing an exclusion.''')
code('''missingness = pd.read_csv(AUDIT / 'missingness.csv')
display(missingness.sort_values('cohort_missing_fraction', ascending=False))
display(model_frame(visible).head())''')
md('''## 5. Who are our protected groups?

The recorded sex codes are 1=male and 2=female. Race codes retain the original Census categories; race does not encode Hispanic ethnicity. We audit sex and race separately and inspect their intersections. Rates below are **label base rates**, not model fairness results.

**Inspect:** Which groups have fewer records? Which have few positive or negative labels? Differences in base rates matter when interpreting later fairness metrics.''')
code('''race_names = {1:'White alone', 2:'Black alone', 3:'American Indian alone',
              4:'Alaska Native alone', 5:'Other American Indian/Alaska Native category',
              6:'Asian alone', 7:'Native Hawaiian/Other Pacific Islander alone',
              8:'Some other race alone', 9:'Two or more races'}
sex = group_summary(visible, ['SEX'])
sex['name'] = sex.SEX.map({1:'Male', 2:'Female'})
race = group_summary(visible, ['RAC1P'])
race['name'] = race.RAC1P.map(race_names)
display(sex)
display(race)
intersections = group_summary(visible, ['RAC1P', 'SEX'])
display(intersections.sort_values('n'))''')
md('''## 6. Does each non-test split support evaluation?

EO needs actual positives and negatives in every calibration group. Small denominators make conditional rates unstable. The value 30 is a flag for discussion, not proof that larger groups are reliable.

**Inspect:** Which calibration groups have fewer than 30 examples of either label? We may need a clearly labeled supported-cohort analysis; we should not silently drop groups.''')
code('''support = pd.read_csv(AUDIT / 'split_groups.csv')
support['small_label_count'] = support[['positives', 'negatives']].min(axis=1).lt(30)
display(support.loc[support['split'].eq('calibration')].sort_values(['small_label_count', 'n'], ascending=[False, True]))''')
md('''## 7. What will the neural network receive?

We use 14 source input columns: scaled age plus one-hot categorical features. Sex and race stay in separate metadata. The scaler and category vocabulary are fitted only on the designated public reference subset. Unknown categories map to all-zero indicator blocks.

**Inspect:** Verify that ESR, Y, PWGTP, SEX, and RAC1P never appear in input columns. One-hot encoding increases the dimension beyond 14.''')
code('''print('Source inputs:', INPUTS)
print('Encoded feature names:', manifest['encoded_features'])
for split, n in manifest['split_counts'].items():
    matrix = sparse.load_npz(DATA / f'X_{split}.npz')
    metadata = pd.read_csv(DATA / f'metadata_{split}.csv', usecols=['row_id'])
    assert matrix.shape == (n, manifest['encoded_dimension'])
    assert len(metadata) == n and metadata.row_id.is_unique
    assert np.isfinite(matrix.data).all()
    print(split, matrix.shape, matrix.dtype)
assert len(INPUTS) == 14
assert not set(['SEX', 'RAC1P', 'ESR', 'Y', 'PWGTP']) & set(INPUTS)
print('Alignment, dimensions, and finite-value checks passed.')''')
md('''## 8. Your review notes

Edit this Markdown cell after inspecting the tables:

- Population/filter observations:
- Target definition and interpretation:
- Missingness rules to keep or revisit:
- Small groups or calibration-support concerns:
- Questions before the baseline model:

The next experiment is the non-private MLP. Keep the test set reserved while choosing model settings. Record any design changes in proposal.md before rerunning preparation.''')
nb.cells = cells
nb.metadata = {'kernelspec': {'display_name': 'Python (project .venv)', 'language': 'python', 'name': 'python3'},
               'language_info': {'name': 'python'}}
path = root / 'notebooks' / '01_data_audit.ipynb'
path.parent.mkdir(exist_ok=True)
nbf.write(nb, path)
from annotate_audit_notebook import annotate
annotate(root)
print(path)
