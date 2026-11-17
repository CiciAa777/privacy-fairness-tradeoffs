"""Add explanations without replacing existing notebook code, outputs, or notes."""
from pathlib import Path
import json
import nbformat


def annotate(root):
    path = root / 'notebooks' / '01_data_audit.ipynb'
    notebook = nbformat.read(path, as_version=4)
    manifest = json.loads((root / 'data/processed/manifest.json').read_text())
    raw, eligible, excluded = (manifest[k] for k in ['raw_rows', 'eligible_rows', 'excluded_rows'])
    dimension = manifest['encoded_dimension']
    notes = [
        ("from pathlib import Path", "setup", '''### What this setup cell does

We will inspect one main pandas DataFrame, called `visible`. The setup cell only makes the tools and file locations available.

- `import` makes a library's functions available; it does not load or change our records.
- `pandas` (`pd`) handles tables; `numpy` (`np`) supports numerical checks.
- `Path` locates files. `ROOT` searches the current folder and its parents for our project. `DATA` and `AUDIT` are folder paths.
- `sys.path` lets Python find our project's helper code in `src/data.py`.
- `FEATURES` lists 17 source columns; `INPUTS` lists the 14 that the model uses after ESP is dropped. `group_summary` counts groups; `model_frame` prepares column types.
- `manifest` is a small dictionary describing an earlier preparation run. It contains counts and settings, not the individual records.
- `sparse` is only needed for the optional matrix checks near the end.

**The printed paths confirm which Python environment and project folder are being used.**'''),
        ("display(pd.Series", "population", f'''### How to read these results

| Result | Meaning | This preparation run |
| --- | --- | --- |
| `raw_rows` | People in the downloaded California file before our filters | {raw:,} |
| `eligible_rows` | People retained after the age and positive-weight rules | {eligible:,} |
| `excluded_rows` | People who failed at least one of those rules | {excluded:,} ({excluded/raw:.1%} of raw records) |
| `encoded_dimension` | Numerical input columns per person after encoding | {dimension} |

`raw_rows = eligible_rows + excluded_rows`. Excluded records were not necessarily errors: many people simply fall outside our study population. These counts do not mean we removed everyone with missing features, or removed retirees.

The split table divides the eligible people by purpose: **reference** fits preprocessing/LFR; **train** teaches the MLP; **development** helps choose settings; **calibration** teaches EO rules; **test** is reserved for the final assessment.

**What the code does:**

1. `pd.Series(...).to_frame()` turns saved counts into a display table.
2. `pd.read_csv(...)` loads `cohort.csv` into one DataFrame named `cohort`. It already contains filtered records, raw feature codes, the target, and split assignments.
3. `.loc[...ne('test')].copy()` creates `visible`, our inspection DataFrame, by selecting non-test rows. It has {eligible - manifest['split_counts']['test']:,} rows. Test labels stay out of the descriptive analysis.
4. `assert` checks the age/weight rules and stops with an error if a rule fails. No output means the check passed.
5. `.head(8)` displays eight example records. `ESR` is the original code, `Y` the binary target, and `split` the assigned purpose. `NaN` denotes a missing value.

**77 columns does not mean 77 people.** Encoding expands the 14 input variables into 1 scaled-age column and {dimension-1} category-indicator columns; section 7 explains why.'''),
        ("esr_names =", "target", '''### What the target tables tell us

In the first table, `n` counts records with each ESR code. `label` shows the shared value of `Y` for that code. `meaning` translates the code into words. The code uses `groupby('ESR')` to form these groups, then `agg(...)` to count records and retrieve their label.

`visible.Y.mean()` is the positive-label fraction because `Y` contains zeros and ones. For example, **0.5692 means approximately 56.92% of the inspected sample has Y=1**. It is a property of the labels, not model accuracy: no model has made predictions here.

`pd.cut(...)` places ages into four bands for display. It does not change the model's age feature or exclude additional people. The second table shows each band's size and positive-label fraction. Differences between bands help us understand the population; they do not establish discrimination.'''),
        ("display(visible.PWGTP", "weights", '''### How to read `describe()`

- `count`: records with a non-missing weight.
- `mean`: average weight; `std`: how much weights vary.
- `min`/`max`: smallest/largest weight.
- `25%`, `50%`, `75%`: quartiles; `50%` is the median.

These statistics describe the survey weights, not employment probabilities. A mean weight of 100, for example, would not mean 100% employment. We are not multiplying rows or weighting the current summaries by PWGTP.

The printed **False** confirms that `PWGTP` is absent from the model-input list. The expression `... in INPUTS` tests list membership.'''),
        ("missingness =", "missingness", '''### How to interpret missingness

`pd.read_csv(...)` loads a small audit table created during preparation. `sort_values(...)` puts the highest missing fractions first.

- `raw_missing`: empty values before eligibility filtering.
- `cohort_missing`: empty values among all eligible records.
- `cohort_missing_fraction`: that count divided by eligible records. Multiply by 100 to read a percentage.

In this run, **ESP is missing for about 98.6% of eligible people**, while GCL is missing for about 21.6%. Many entries are structurally not applicable. Deleting every record with a missing ESP would remove almost the entire study population. A zero missing count also does not guarantee that every code is substantively valid; applicability/suppression codes need their own interpretation.

`model_frame(visible).head()` shows how the input columns are represented before one-hot encoding. Categorical missing values become `NOT_APPLICABLE`; category codes become strings so we do not treat them as numerical magnitudes. ESP is now dropped by user decision; the audit table retains its original missingness. This helper returns a new table and preserves `visible`. Its name does not mean a neural network has been trained.'''),
        ("race_names =", "groups", '''### Reading the group tables

`group_summary(...)` performs the same counting operation for each requested group. `.map(...)` adds readable names to numeric codes. Sorting intersections by `n` puts the smallest groups first.

| Column | Meaning |
| --- | --- |
| `n` | Number of sampled people in the group |
| `positives` | People with Y=1 |
| `negatives` | People with Y=0 |
| `positive_rate` | `positives / n` |

For example, this run has positive-label rates of about **61.6% for recorded males and 52.4% for recorded females** in non-test records. That is a difference in recorded outcomes, not a fairness result for a predictor. Later we will compare predictions and errors as well as true labels.

The race-by-sex table forms combinations of the two attributes. Some are very small: the Alaska Native-alone category has only 17 non-test records in total. A rate from a handful of people can change sharply with one record. Rows with `n=0` deliberately show absent groups; their positive rates are undefined.'''),
        ("support =", "support", '''### What the support flag means

The support table breaks each split down by race and sex. The code selects the calibration rows, computes the smaller of `positives` and `negatives` for each group, and flags values below 30. It sorts flagged groups first, then by group size.

**True means review this group**, not that the data are corrupt or that it must be deleted. The threshold 30 is our discussion flag, not a statistical guarantee.

In this run, some race-by-sex calibration cells are empty or contain only one label. Without actual negatives, a group's false-positive rate is undefined; without actual positives, its true-positive rate is undefined. That prevents an ordinary equalized-odds fit over those intersectional groups. This is a support check for a possible intersectional analysis; our primary interventions still target sex and race separately. We must check those groupings too before fitting them.'''),
        ("print('Source inputs:'", "encoding", f'''### Why do 14 variables become {dimension} columns?

A categorical variable becomes several 0/1 indicator columns. For illustration, a variable with categories A, B, and C becomes three columns: A is `[1,0,0]`, B is `[0,1,0]`, and C is `[0,0,1]`. This is **one-hot encoding**. It avoids treating category codes as measurements with meaningful numerical distances.

Our saved representation has **1 scaled-age column + {dimension-1} indicator columns = {dimension} inputs per person**. Encoded names identify their origin: `categories__MAR_1`, for example, indicates membership in marital-status category 1. Age is centered/scaled using reference-set statistics; it is still one variable.

This final code cell is a file-consistency check. You can focus on the ordinary DataFrame sections first:

- `sparse.load_npz(...)` reads the numerical matrix for a split. Sparse storage saves space by storing mostly nonzero entries; it represents the same rectangular data idea as a table.
- `metadata` loads row identifiers to check row counts without displaying test labels.
- A shape such as `(151320, {dimension})` means **151,320 people × {dimension} numerical columns**.
- `float32` is the numerical storage type, not a performance score.
- The assertions check dimensions, unique identifiers, finite numerical values, and exclusion of target/protected columns from direct inputs.

**“Checks passed” means these preparation checks succeeded. It does not establish predictive accuracy or fairness.**''')
    ]
    # Refresh our own annotations, preserving all original/user-authored cells.
    notebook.cells = [c for c in notebook.cells if not c.metadata.get('audit_annotation')]
    for prefix, key, text in notes:
        for i, cell in enumerate(notebook.cells):
            if cell.cell_type == 'code' and cell.source.lstrip().startswith(prefix):
                note = nbformat.v4.new_markdown_cell(text)
                note.metadata['audit_annotation'] = key
                notebook.cells.insert(i + 1, note)
                break
        else:
            raise ValueError(f'Could not locate code cell: {key}')
    nbformat.write(notebook, path)
    print('Added explanations beside all eight code-cell outputs; preserved code and existing notes.')


if __name__ == '__main__':
    annotate(Path(__file__).resolve().parents[1])
