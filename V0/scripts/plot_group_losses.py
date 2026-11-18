"""Build the group accuracy-loss figure and progress-report tables from saved results.

Reads results/ only (no retraining, no test data). Writes:
  figures/dp_pilot/group_loss_vs_epsilon.png
  results/dp_pilot/report_tables.md
Run from anywhere:  .\\.venv\\Scripts\\python.exe scripts\\plot_group_losses.py
"""
from pathlib import Path
import json
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / 'results/dp_pilot'
FIG = ROOT / 'figures/dp_pilot'
FIG.mkdir(parents=True, exist_ok=True)

MIN_N = 500  # plot only groups with enough records; small groups are tabulated, not plotted
SEX_NAMES = {'1': 'Male', '2': 'Female'}
RACE_NAMES = {'1': 'White alone', '2': 'Black alone', '3': 'American Indian alone',
              '4': 'Alaska Native alone', '5': 'Other AIAN category', '6': 'Asian alone',
              '7': 'NHPI alone', '8': 'Some other race alone', '9': 'Two or more races'}

summaries = [json.loads(p.read_text()) for p in sorted(RUN.glob('epsilon_*/summary.json'))]
losses = []
for s in summaries:
    f = pd.read_csv(RUN / f"epsilon_{s['target_epsilon']:g}" / 'group_accuracy_losses.csv',
                    dtype={'group': str})
    f['target_epsilon'] = s['target_epsilon']
    f['epsilon'] = s['epsilon']
    losses.append(f)
losses = pd.concat(losses, ignore_index=True)

# ---- Figure: accuracy loss vs epsilon, sex (left) and larger race groups (right)
fig, axes = plt.subplots(1, 2, figsize=(10, 3.6), sharey=True)
for ax, attribute, names in zip(axes, ['SEX', 'RAC1P'], [SEX_NAMES, RACE_NAMES]):
    frame = losses[losses.attribute.eq(attribute)]
    for group, g in frame.groupby('group'):
        if g.n.iloc[0] < MIN_N:
            continue
        g = g.sort_values('epsilon')
        ax.plot(g.epsilon, g.accuracy_loss_pp, marker='o', label=f'{names[group]} (n={g.n.iloc[0]:,})')
    ax.set(title=f'{attribute}: accuracy loss vs. ordinary SGD',
           xlabel='Achieved epsilon (smaller = stronger privacy)')
    ax.axhline(0, color='gray', linewidth=.5)
    ax.legend(fontsize=7)
axes[0].set_ylabel('Accuracy loss (percentage points)')
fig.text(0.5, -0.02, f'Single run per epsilon, development data. Race groups with n < {MIN_N} omitted '
         '(see table).', ha='center', fontsize=7)
plt.tight_layout()
fig.savefig(FIG / 'group_loss_vs_epsilon.png', dpi=180, bbox_inches='tight')

# ---- Tables (markdown)
def md(frame):
    frame = frame.astype(object).where(frame.notna(), 'undefined')
    rows = [' | '.join(map(str, r)) for r in frame.itertuples(index=False, name=None)]
    return '\n'.join(['| ' + ' | '.join(frame.columns) + ' |',
                      '| ' + ' | '.join(['---'] * len(frame.columns)) + ' |'] +
                     ['| ' + r + ' |' for r in rows])

comparison = pd.read_csv(RUN / 'comparison.csv')
overall = comparison[['method', 'epsilon', 'accuracy', 'balanced_accuracy', 'accuracy_loss_pp']].copy()
overall['accuracy'] = (100 * overall.accuracy).round(2)
overall['balanced_accuracy'] = (100 * overall.balanced_accuracy).round(2)
overall['epsilon'] = overall.epsilon.round(3)
overall['accuracy_loss_pp'] = overall.accuracy_loss_pp.round(2)

gap_cols = ['method', 'target_epsilon'] + [f'{a}_{m}' for a in ['SEX', 'RAC1P']
                                           for m in ['accuracy_gap', 'independence', 'separation', 'sufficiency']]
gaps = comparison[gap_cols].copy()
for c in gap_cols[2:]:
    gaps[c] = (100 * gaps[c]).round(2)

wide = (losses[losses.attribute.isin(['SEX', 'RAC1P'])]
        .pivot_table(index=['attribute', 'group', 'n'], columns='target_epsilon',
                     values='accuracy_loss_pp').round(2).reset_index())
wide['name'] = [ (SEX_NAMES if a == 'SEX' else RACE_NAMES)[g] for a, g in zip(wide.attribute, wide.group)]
wide['flag'] = wide.n.map(lambda n: 'n<100: unstable' if n < 100 else '')
wide.columns = [str(c) for c in wide.columns]

privacy = pd.DataFrame([{k: s[k] for k in ['target_epsilon', 'epsilon', 'delta', 'noise_multiplier',
                                           'sample_rate', 'steps', 'seconds']} for s in summaries]).round(6)

text = ('## Overall (percent; loss in percentage points)\n\n' + md(overall) +
        '\n\n## Fairness gaps (percentage points; undefined = a required group rate is undefined)\n\n' + md(gaps) +
        '\n\n## Group accuracy loss (pp) by target epsilon\n\n' + md(wide) +
        '\n\n## Privacy settings\n\n' + md(privacy) + '\n')
(RUN / 'report_tables.md').write_text(text, encoding='utf-8')
print('Wrote', FIG / 'group_loss_vs_epsilon.png', 'and', RUN / 'report_tables.md')
