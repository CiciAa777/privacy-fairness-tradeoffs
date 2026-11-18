r"""Present saved pilot fairness gaps and their underlying group rates.

Run: .\.venv\Scripts\python.exe scripts\report_fairness.py
Reads aggregate results only; no retraining or test-data access.
"""
from pathlib import Path
import json
import os

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / 'tmp/matplotlib'
CACHE.mkdir(parents=True, exist_ok=True)
os.environ.setdefault('MPLCONFIGDIR', str(CACHE))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

RUN = ROOT / 'results/dp_pilot'
FIG = ROOT / 'figures/dp_pilot'
REPORT = ROOT / 'reports/fairness_results'
FIG.mkdir(parents=True, exist_ok=True)
REPORT.mkdir(parents=True, exist_ok=True)
SEX = {'1': 'Male', '2': 'Female'}
RACE = {'1': 'White alone', '2': 'Black or African American alone',
        '3': 'American Indian alone', '4': 'Alaska Native alone',
        '5': 'AIAN specified/unspecified; no other races', '6': 'Asian alone',
        '7': 'Native Hawaiian / Other Pacific Islander alone',
        '8': 'Some Other Race alone', '9': 'Two or More Races'}
GAPS = [('accuracy_gap', 'Accuracy parity'), ('independence', 'Independence'),
        ('separation', 'Separation'), ('sufficiency', 'Binary sufficiency')]
RATES = [('accuracy', 'Accuracy'), ('selection_rate', 'Predicted positive'),
         ('tpr', 'TPR'), ('fpr', 'FPR'), ('ppv', 'PPV'), ('npv', 'NPV')]

runs = []
folders = [ROOT / 'results/baseline'] + sorted(RUN.glob('epsilon_*'),
           key=lambda p: json.loads((p/'summary.json').read_text(encoding='utf-8'))['epsilon'])
for folder in folders:
    summary = json.loads((folder/'summary.json').read_text(encoding='utf-8'))
    groups = pd.read_csv(folder/'group_metrics.csv', dtype={'group': str})
    label = 'SGD' if folder.name == 'baseline' else f"DP epsilon {summary['target_epsilon']:g}"
    runs.append(dict(label=label, summary=summary, groups=groups))

# Independently check the stored rates against their confusion-matrix counts,
# then check full-group gaps without discarding undefined categories.
def checked_rate(num, den):
    return num/den if den else np.nan

for run in runs:
    for row in run['groups'].itertuples():
        assert row.n == row.tp + row.tn + row.fp + row.fn
        pairs = [(row.tp+row.tn, row.n), (row.tp+row.fp, row.n),
                 (row.tp, row.tp+row.fn), (row.fp, row.fp+row.tn),
                 (row.tp, row.tp+row.fp), (row.tn, row.tn+row.fn)]
        for (key, _), (num, den) in zip(RATES, pairs):
            assert np.isclose(getattr(row, key), checked_rate(num, den), equal_nan=True)
    for attribute in ['SEX', 'RAC1P']:
        frame = run['groups'].loc[run['groups'].attribute.eq(attribute)]
        def gap(key):
            return frame[key].max()-frame[key].min() if frame[key].notna().all() else np.nan
        expected = dict(accuracy_gap=gap('accuracy'), independence=gap('selection_rate'),
                        separation=np.maximum(gap('tpr'), gap('fpr')),
                        sufficiency=np.maximum(gap('ppv'), gap('npv')))
        for key, value in expected.items():
            stored = run['summary']['gaps'][attribute][key]
            assert np.isclose(value, np.nan if stored is None else stored, equal_nan=True)
    assert run['summary']['overall']['n'] == runs[0]['summary']['overall']['n']
    assert run['summary']['manifest_sha256'] == runs[0]['summary']['manifest_sha256']

plt.rcParams.update({'font.size': 11, 'axes.spines.top': False, 'axes.spines.right': False})
x = np.array([r['summary']['epsilon'] for r in runs[1:]])
legend = [Line2D([], [], color='#2877ad', marker='o', label='DP-SGD (one run per budget)'),
          Line2D([], [], color='#777777', linestyle='--', label='Ordinary SGD baseline')]
for attribute, title in [('SEX', 'Sex groups'), ('RAC1P', 'All nine Census race groups')]:
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.2))
    for ax, (key, name) in zip(axes.flat, GAPS):
        ax.set_title(name, weight='bold')
        values = [r['summary']['gaps'][attribute][key] for r in runs]
        if all(v is None for v in values):
            ax.text(.5, .53, 'Undefined at every budget', ha='center', fontsize=13, weight='bold')
            reason = ('No actual positives: TPR undefined' if key == 'separation'
                      else 'No predicted positives: PPV undefined')
            ax.text(.5, .30, f'Alaska Native alone: n = 2\n{reason}\n'
                    'No groups silently excluded',
                    ha='center', va='center', fontsize=10, linespacing=1.5)
            ax.set_axis_off()
            continue
        base, private = 100*values[0], 100*np.asarray(values[1:])
        ax.plot(x, private, color='#2877ad', marker='o', linewidth=2)
        ax.axhline(base, color='#777777', linestyle='--')
        for budget, value in zip(x, private):
            ax.annotate(f'{value:.2f}', (budget, value), xytext=(0, 8),
                        textcoords='offset points', ha='center', fontsize=10)
        ax.text(.98, .06, f'SGD: {base:.2f} pp', transform=ax.transAxes,
                ha='right', fontsize=10, color='#555555')
        ax.set(ylim=(0, 1.24*max([base, *private])), xlim=(.4, 8.8),
               ylabel='Group gap (percentage points)', xlabel='Achieved epsilon (approximately)')
        ax.set_xticks(x, ['1', '4', '8'])
        ax.grid(alpha=.15)
    plot_title = (f'{title}: fairness gaps across privacy budgets' if attribute == 'SEX'
                  else 'Race groups: fairness gaps across privacy budgets\nAll nine original Census categories retained')
    fig.suptitle(plot_title, weight='bold', fontsize=15)
    fig.legend(handles=legend, loc='lower center', bbox_to_anchor=(.5, .035), ncol=2, frameon=False)
    note = ('Single-run development results; lower gaps mean closer parity, not necessarily better predictions. '
            'No uncertainty bars yet.')
    if attribute == 'RAC1P':
        note = 'All groups retained, including n = 2; extreme race gaps are sensitive to very small groups. Undefined is not zero.'
    fig.text(.5, .015, note, ha='center', fontsize=8.2)
    fig.tight_layout(rect=(0, .09, 1, .95))
    name = 'sex' if attribute == 'SEX' else 'race'
    fig.savefig(FIG/f'{name}_fairness_gaps.png', dpi=180)
    plt.close(fig)

# Underlying rates: positive-prediction rates, actual-class error/detection
# rates, and predicted-class reliability are rates, not themselves gaps.
fig, axes = plt.subplots(2, 3, figsize=(12, 7))
for ax, (key, name) in zip(axes.flat, RATES):
    for code, color in [('1', '#2877ad'), ('2', '#dd8543')]:
        values = [100*r['groups'].loc[(r['groups'].attribute.eq('SEX')) &
                  r['groups'].group.eq(code), key].iloc[0] for r in runs]
        ax.plot(x, values[1:], color=color, marker='o', linewidth=2)
        ax.axhline(values[0], color=color, linestyle='--', alpha=.7)
    ax.set(title=name, ylim=(0, 100), ylabel='Rate (%)', xlabel='Achieved epsilon (approximately)')
    ax.set_xticks(x, ['1', '4', '8'])
    ax.grid(alpha=.15)
handles = [Line2D([], [], color=color, linestyle=style, marker='o' if style == '-' else None,
           label=f'{label}: {method}') for label, color in [('Male', '#2877ad'), ('Female', '#dd8543')]
           for method, style in [('DP-SGD', '-'), ('SGD', '--')]]
fig.suptitle('Sex groups: the underlying rates behind the fairness gaps', weight='bold', fontsize=15)
fig.legend(handles=handles, loc='lower center', bbox_to_anchor=(.5, .035), ncol=4, frameon=False)
fig.text(.5, .015, 'Single-run development results. Higher FPR means more false alarms; higher TPR/PPV/NPV means better rates.',
         ha='center', fontsize=9)
fig.tight_layout(rect=(0, .10, 1, .95))
fig.savefig(FIG/'sex_group_rates.png', dpi=180)
plt.close(fig)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
cmap = plt.colormaps['YlGnBu'].copy()
cmap.set_bad('#eeeeee')
for ax, run in zip(axes.flat, runs):
    frame = run['groups'].loc[run['groups'].attribute.eq('RAC1P')].set_index('group').loc[list(RACE)]
    values = 100*frame[[key for key, _ in RATES]].to_numpy()
    heat = ax.imshow(np.ma.masked_invalid(values), vmin=0, vmax=100, cmap=cmap, aspect='auto')
    ax.set_xticks(range(6), ['Accuracy', 'Pred. +', 'TPR', 'FPR', 'PPV', 'NPV'],
                  rotation=25, ha='right', fontsize=10)
    short_names = ['White', 'Black', 'American Indian', 'Alaska Native', 'AIAN code 5',
                   'Asian', 'NHPI', 'Some Other Race', 'Two or More Races']
    ax.set_yticks(range(9), [f'{label} (n={n:,})' for label, n in zip(short_names, frame.n)])
    ax.set_title(run['label'], weight='bold')
    for (i, j), value in np.ndenumerate(values):
        ax.text(j, i, 'N/D' if np.isnan(value) else f'{value:.1f}', ha='center', va='center',
                fontsize=9, color='white' if value >= 65 else '#202020')
fig.suptitle('All nine race groups: underlying rates (%)', weight='bold', fontsize=16)
fig.text(.5, .025, 'Single-run development data. N/D = undefined, not zero. '
         'Rare-group values are unstable; high FPR is undesirable. Full Census labels are in the results table.',
         ha='center', fontsize=9)
fig.tight_layout(rect=(0, .055, .93, .95))
cax = fig.add_axes([.945, .15, .012, .68])
fig.colorbar(heat, cax=cax, label='Rate (%)')
fig.savefig(FIG/'race_group_rates.png', dpi=180)
plt.close(fig)

def md(columns, rows):
    return '\n'.join(['| '+' | '.join(columns)+' |', '| '+' | '.join(['---']*len(columns))+' |']+
                     ['| '+' | '.join(map(str, row))+' |' for row in rows])

def fmt(value, scale=100):
    return 'undefined' if value is None or pd.isna(value) else f'{scale*value:.2f}'

text = ['# Observed fairness results across privacy budgets\n',
        '**Single-run development results.** Same 30,264 records, threshold 0.5, no fairness intervention yet. '
        'No retraining or test evaluation was performed to produce these tables.\n',
        '## Read the units first\n',
        'The four fairness measurements below are **group gaps in percentage points**; lower means closer parity. '
        'They are not per-group rates. The later tables show the rates themselves, in percent. '
        'A 10-point gap means a difference such as 80% versus 70%.\n',
        '- Accuracy parity: gap in accuracy.\n- Independence: gap in positive-prediction rate.\n'
        '- Separation: maximum of TPR and FPR gaps.\n- Binary sufficiency: maximum of PPV and NPV gaps.\n']
for attribute, title in [('SEX', 'Sex'), ('RAC1P', 'Race: all nine original categories')]:
    rows = [[r['label'], *[fmt(r['summary']['gaps'][attribute][key]) for key, _ in GAPS]] for r in runs]
    name = 'sex' if attribute == 'SEX' else 'race'
    text += [f'## {title}: fairness gaps (pp)\n', md(['Model / target epsilon', *[title for _, title in GAPS]], rows),
             f'\n![{title} fairness gaps](../../figures/dp_pilot/{name}_fairness_gaps.png)\n']
text += ['Race separation/sufficiency remain undefined because Alaska Native alone has n=2, '
         'zero actual positives, and zero predicted positives in every run. TPR and PPV have zero denominators. '
         'That group also has 100% accuracy and a 0% positive-prediction rate, strongly affecting the reported '
         'full-race accuracy/independence gaps. No categories have been dropped or merged.\n',
         '## Sex: component gaps (pp)\n',
         md(['Model', 'TPR gap', 'FPR gap', 'PPV gap', 'NPV gap'],
            [[r['label'], *[fmt(r['summary']['gaps']['SEX'][key]) for key in
              ['tpr_gap', 'fpr_gap', 'ppv_gap', 'npv_gap']]] for r in runs]),
         '\n## Sex: underlying group rates (%)\n',
         'TPR = actual positives found; FPR = actual negatives falsely predicted positive; '
         'PPV/NPV = reliability of positive/negative predictions.\n']
rows = []
for run in runs:
    for row in run['groups'].loc[run['groups'].attribute.eq('SEX')].itertuples():
        rows.append([run['label'], SEX[row.group], f'{row.n:,}', *[fmt(getattr(row, key)) for key, _ in RATES]])
text += [md(['Model', 'Group', 'n', *[name+' (%)' for _, name in RATES]], rows),
         '\n![Sex group rates](../../figures/dp_pilot/sex_group_rates.png)\n',
         '## Race: underlying group rates (%)\n']
for run in runs:
    rows = [[RACE[row.group], f'{row.n:,}', row.tp+row.fn, row.tp+row.fp,
             *[fmt(getattr(row, key)) for key, _ in RATES]]
            for row in run['groups'].loc[run['groups'].attribute.eq('RAC1P')].itertuples()]
    text += [f"### {run['label']}\n", md(['Group', 'n', 'Actual +', 'Predicted +',
             *[name+' (%)' for _, name in RATES]], rows), '\n']
text += ['![Race group rates](../../figures/dp_pilot/race_group_rates.png)\n',
         '## Interpretation and provenance\n',
         'At epsilon approximately 1 versus SGD, the sex accuracy gap increases (6.67 to 7.38 pp) and independence '
         'gap increases (0.64 to 0.78 pp), while separation decreases (12.01 to 11.38 pp) and sufficiency decreases '
         '(12.58 to 12.12 pp). A smaller gap can accompany worse rates: both male and female false-positive rates '
         'increase. Inspect absolute group performance alongside parity.\n',
         'All settings have only one run; no uncertainty bars or significance claims are supported yet. '
         'Rates with fewer than 30 observations in the relevant denominator are unstable under our reporting convention; '
         'the raw counts/denominators can be recovered from each group-metrics CSV. '
         'No threshold, cohort, or metric definition was changed.\n',
         'Sources: [baseline summary](../../results/baseline/summary.json), '
         '[baseline group rates](../../results/baseline/group_metrics.csv), and '
         '`results/dp_pilot/epsilon_{1,4,8}/{summary.json,group_metrics.csv}`. '
         'The script independently checks rates against confusion-matrix counts and full-group gaps against summaries.\n',
         'Rebuild from the repository root with '
         '` .\\.venv\\Scripts\\python.exe scripts\\report_fairness.py `. '
         'Figures are saved under `figures/dp_pilot/`.\n']
(REPORT/'README.md').write_text('\n'.join(text), encoding='utf-8')
print('Checked stored rates/gaps; wrote reports/fairness_results/README.md and four plots under figures/dp_pilot/.')
