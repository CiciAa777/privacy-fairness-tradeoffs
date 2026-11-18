"""Build compact report assets from frozen, verified controlled-v2 results.

From the repo root: .venv/Scripts/python.exe reports/progress_report/build_figures.py
No model training, new evaluation, or aggregate rewriting occurs.
"""
from pathlib import Path
import hashlib
import json
import os

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
SOURCES = ROOT/'results/controlled_v2'
OUT = HERE/'figures'


def main():
    os.environ.setdefault('MPLCONFIGDIR', str(ROOT/'tmp/matplotlib'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    OUT.mkdir(parents=True, exist_ok=True)
    (HERE/'tables').mkdir(exist_ok=True)
    paths = [SOURCES/'report_assets/tables/figure1_group_changes.csv',
             SOURCES/'report_assets/tables/table1_sgd_vs_dp1.csv',
             SOURCES/'aggregate/group_rate_changes_mean_std.csv',
             SOURCES/'aggregate/metrics_mean_std.csv']
    hashes = {str(p.relative_to(ROOT)).replace('\\', '/'): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    changes, table, raw_changes, raw_stats = [pd.read_csv(p, dtype={'group': str}) for p in paths]
    raw_changes = raw_changes.loc[raw_changes.feature_policy.eq('include_sensitive') &
                                  raw_changes.method.eq('dp_epsilon_1') & raw_changes.attribute.eq('RAC1P')]
    for _, row in changes.iterrows():
        raw = raw_changes.loc[raw_changes.group.eq(row.group) & raw_changes.metric.eq(row.metric)].iloc[0]
        np.testing.assert_allclose([row.mean_pp, row.sd_pp],
            [(-100 if row.metric=='accuracy' else 100)*raw['mean'], 100*raw['std']], atol=1e-12)
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9, 'axes.titlesize': 9.5,
        'axes.labelsize': 9, 'xtick.labelsize': 9, 'ytick.labelsize': 9,
        'axes.spines.top': False, 'axes.spines.right': False, 'axes.linewidth': .7,
        'pdf.fonttype': 42, 'svg.fonttype': 'none', 'savefig.facecolor': 'white'})
    races = ['1', '2', '6', '8', '9']
    names = ['White alone', 'Black alone', 'Asian alone', 'Some Other Race alone', 'Two or More Races']
    labels = [f'{name} ({int(changes.loc[changes.group.eq(g), "n"].iloc[0]):,})' for name, g in zip(names, races)]
    fig, axes = plt.subplots(1, 2, figsize=(6.5, 2.1), sharey=True)
    fig.subplots_adjust(left=.34, right=.985, top=.80, bottom=.24, wspace=.25)
    for ax, metric, color, title, limits, ticks in zip(axes, ['accuracy', 'fpr'],
              ['#2166AC', '#C65C28'], ['(a) Accuracy loss: SGD - DP', '(b) FPR increase: DP - SGD'],
              [(-.4, 5.5), (-1.5, 31)], [[0, 2, 4], [0, 10, 20, 30]]):
        block = changes.loc[changes.metric.eq(metric)].set_index('group').loc[races]
        assert (block.mean_pp-block.sd_pp).min()>limits[0] and (block.mean_pp+block.sd_pp).max()<limits[1]
        ax.axvline(0, color='#888888', lw=.7)
        ax.errorbar(block.mean_pp, range(5), xerr=block.sd_pp, fmt='o', color=color,
                    capsize=2.5, markersize=4, elinewidth=1)
        ax.set_xlim(limits); ax.set_xticks(ticks); ax.set_yticks(range(5), labels)
        ax.tick_params(axis='y', length=0)
        ax.set_title(title, loc='left', pad=8, fontsize=9)
        ax.set_xlabel('Percentage points')
        ax.grid(axis='x', alpha=.18)
    axes[0].invert_yaxis()
    for extension in ['pdf', 'png', 'svg']:
        path = OUT/f'racial_impact.{extension}'
        fig.savefig(path, dpi=350)
        if extension=='svg':
            path.write_text('\n'.join(line.rstrip() for line in path.read_text(encoding='utf-8').splitlines())+'\n', encoding='utf-8')
    plt.close(fig)

    rows = [r'\begin{tabular*}{\linewidth}{@{\extracolsep{\fill}}lrrrr@{}}', r'\toprule',
            r' & \multicolumn{2}{c}{Sex} & \multicolumn{2}{c}{Supported race} \\',
            r'Gap (pp) & SGD & DP1 & SGD & DP1 \\', r'\midrule']
    for metric, label in [('accuracy_gap', 'Accuracy parity'), ('independence', 'Independence'),
                           ('separation', 'Separation'), ('sufficiency', 'Binary sufficiency')]:
        cells = [label]
        for scope in ['SEX', 'RAC1P_n500']:
            row = table.loc[table.scope.eq(scope) & table.metric.eq(metric)].iloc[0]
            for method, prefix in [('sgd', 'sgd'), ('dp_epsilon_1', 'dp1')]:
                raw = raw_stats.loc[raw_stats.feature_policy.eq('include_sensitive') & raw_stats.method.eq(method) &
                                    raw_stats.metric.eq(scope+'_'+metric)].iloc[0]
                np.testing.assert_allclose([row[prefix+'_mean'], row[prefix+'_sd']], 100*raw[['mean', 'std']].to_numpy(dtype=float), atol=1e-12)
                cells.append(f'${row[prefix+"_mean"]:.2f}\\!\\pm\\!{row[prefix+"_sd"]:.2f}$')
        rows.append(' & '.join(cells)+r' \\')
    row = table.loc[table.metric.eq('accuracy')].iloc[0]
    rows += [r'\midrule', r'\multicolumn{5}{l}{Overall accuracy (\%): SGD $'+
             f'{row.sgd_mean:.2f}\\pm{row.sgd_sd:.2f}$; DP1 ${row.dp1_mean:.2f}\\pm{row.dp1_sd:.2f}'+r'$} \\',
             r'\bottomrule', r'\end{tabular*}']
    (HERE/'tables/sgd_dp1.tex').write_text('\n'.join(rows)+'\n', encoding='utf-8')
    for p in paths:
        assert hashlib.sha256(p.read_bytes()).hexdigest()==hashes[str(p.relative_to(ROOT)).replace('\\', '/')]
    (HERE/'asset_provenance.json').write_text(json.dumps({'source_sha256': hashes,
        'no_training_or_new_evaluation': True, 'sources_unchanged': True}, indent=2)+'\n', encoding='utf-8')
    print('Built compact 2-panel racial figure and one comparison table; source values checked and unchanged.')


if __name__=='__main__':
    main()
