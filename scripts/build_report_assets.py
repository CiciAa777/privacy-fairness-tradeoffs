"""Verify frozen controlled-v2 aggregates and build report assets without training.

Run from the repository root with .venv/Scripts/python.exe scripts/build_report_assets.py.
Only numeric report tables and figures are generated. Root findings.md is maintained by the team.
"""
from pathlib import Path
import hashlib
import json
import os
import re

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT/'results/controlled_v2'
AGG = RESULTS/'aggregate'
OUT = RESULTS/'report_assets'
PRIMARY = 'include_sensitive'
SEEDS = [42, 123, 285, 2026, 2027]
METHODS = ['sgd', 'poisson_unclipped', 'poisson_clipped', 'dp_epsilon_1', 'dp_epsilon_4', 'dp_epsilon_8']
DP = METHODS[3:]
RACES = ['1', '2', '6', '8', '9']
RACE_NAMES = ['White alone', 'Black alone', 'Asian alone', 'Some Other Race alone', 'Two or More Races']
GAPS = [('accuracy_gap', 'Accuracy parity'), ('independence', 'Independence'),
        ('separation', 'Separation'), ('sufficiency', 'Binary sufficiency')]
GROUP_METRICS = ['accuracy', 'balanced_accuracy', 'selection_rate', 'tpr', 'fpr', 'ppv', 'npv']


def read(name):
    return pd.read_csv(AGG/(name+'.csv'), dtype={'group': str})


def dump(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n', encoding='utf-8')


def frozen_hashes():
    # Includes local checkpoints/predictions, but publishes only already-tracked aggregate hashes.
    paths = list(AGG.glob('*'))+list((RESULTS/'runs').rglob('*'))
    return {str(p.relative_to(ROOT)).replace('\\', '/'): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in paths if p.is_file()}


def one(frame, **identity):
    for key, value in identity.items():
        frame = frame.loc[frame[key].eq(value)]
    if len(frame) != 1:
        raise ValueError(f'Expected one row for {identity}, got {len(frame)}')
    return frame.iloc[0]


def close(actual, expected):
    np.testing.assert_allclose(actual, expected, rtol=0, atol=2e-12, equal_nan=True)


def verify_statistics(raw, stored, keys):
    metrics = stored.metric.unique().tolist()
    melted = raw.melt(id_vars=keys+['model_seed'], value_vars=metrics, var_name='metric', value_name='value')
    expected = melted.groupby(keys+['metric'], dropna=False).value.agg(mean='mean', std='std', valid_seeds='count').reset_index()
    joined = stored.merge(expected, on=keys+['metric'], suffixes=('', '_recomputed'), validate='one_to_one')
    assert len(joined) == len(stored) == len(expected)
    for column in ['mean', 'std', 'valid_seeds']:
        close(joined[column], joined[column+'_recomputed'])
    return len(joined)


def verify_sources():
    seeds, groups = read('seed_metrics'), read('group_seed_metrics')
    assert len(seeds) == 60 and len(groups) == 60*29
    assert seeds.groupby(['feature_policy', 'method']).model_seed.apply(lambda x: sorted(x)==SEEDS).all()
    assert not seeds.duplicated(['feature_policy', 'method', 'model_seed']).any()
    checks = {}
    checks['overall_and_gap_statistics'] = verify_statistics(seeds, read('metrics_mean_std'), ['feature_policy', 'method'])
    checks['group_statistics'] = verify_statistics(groups, read('group_mean_std'), ['feature_policy', 'method', 'attribute', 'group'])
    for name in ['paired_vs_sgd', 'group_rate_changes', 'group_accuracy_losses', 'budget_differences']:
        keys = ['feature_policy', 'method']
        if name.startswith('group_'):
            keys += ['attribute', 'group']
        if name == 'budget_differences':
            keys = ['feature_policy', 'comparison']
        checks[name+'_statistics'] = verify_statistics(read(name+'_seed'), read(name+'_mean_std'), keys)

    # Independently reconstruct all group rates from confusion counts.
    def divide(a, b):
        return np.divide(a, b, out=np.full(len(b), np.nan), where=np.asarray(b)!=0)
    tp, tn, fp, fn = [groups[k].to_numpy(dtype=float) for k in ['tp', 'tn', 'fp', 'fn']]
    close(tp+tn+fp+fn, groups.n)
    values = {'accuracy': divide(tp+tn, tp+tn+fp+fn), 'tpr': divide(tp, tp+fn),
              'fpr': divide(fp, fp+tn), 'selection_rate': divide(tp+fp, tp+tn+fp+fn),
              'ppv': divide(tp, tp+fp), 'npv': divide(tn, tn+fn)}
    values['balanced_accuracy'] = (values['tpr']+1-values['fpr'])/2
    for metric, expected in values.items():
        close(groups[metric], expected)
    # Reconstruct per-seed gaps, including strict undefined handling for tiny groups.
    def gap(frame, metric):
        rates = frame[metric]
        return rates.max()-rates.min() if rates.notna().all() else np.nan
    def combined(a, b):
        return max(a, b) if np.isfinite(a) and np.isfinite(b) else np.nan
    for (policy, method, seed), block in groups.groupby(['feature_policy', 'method', 'model_seed']):
        reference = one(seeds, feature_policy=policy, method=method, model_seed=seed)
        for scope in ['SEX', 'RAC1P', 'RAC1P_SEX', 'RAC1P_n500']:
            selected = block.loc[block.attribute.eq('RAC1P' if scope=='RAC1P_n500' else scope)]
            if scope == 'RAC1P_n500':
                selected = selected.loc[selected.n.ge(500)]
                assert set(selected.group)==set(RACES)
            expected = {'accuracy_gap': gap(selected, 'accuracy'), 'independence': gap(selected, 'selection_rate'),
                        'separation': combined(gap(selected, 'tpr'), gap(selected, 'fpr')),
                        'sufficiency': combined(gap(selected, 'ppv'), gap(selected, 'npv'))}
            for metric, value in expected.items():
                close(reference[f'{scope}_{metric}'], value)
        counts = block.loc[block.attribute.eq('SEX')]
        close(reference.accuracy, (counts.tp.sum()+counts.tn.sum())/counts.n.sum())
    checks['group_confusion_rows_and_seed_gap_sets'] = [len(groups), len(seeds)*4]

    # Confirm all saved paired differences, without trusting their aggregate means.
    keys = ['feature_policy', 'model_seed', 'attribute', 'group']
    base = groups.loc[groups.method.eq('sgd'), keys+GROUP_METRICS]
    expected = groups.loc[groups.method.ne('sgd')].merge(base, on=keys, suffixes=('', '_sgd'), validate='many_to_one')
    actual = read('group_rate_changes_seed')
    joined = actual.merge(expected, on=keys+['method'], suffixes=('_saved', ''), validate='one_to_one')
    for metric in GROUP_METRICS:
        close(joined[metric+'_saved'], joined[metric]-joined[metric+'_sgd'])
    loss = read('group_accuracy_losses_seed').merge(actual[keys+['method', 'accuracy']], on=keys+['method'], validate='one_to_one')
    close(loss.accuracy_loss_pp, -100*loss.accuracy)
    paired = read('paired_vs_sgd_seed')
    base = seeds.loc[seeds.method.eq('sgd')]
    joined = seeds.loc[seeds.method.ne('sgd')].merge(base, on=['feature_policy', 'model_seed'], suffixes=('', '_sgd'), validate='many_to_one')
    for _, row in paired.iterrows():
        ref = one(joined, feature_policy=row.feature_policy, method=row.method, model_seed=row.model_seed)
        for metric in read('paired_vs_sgd_mean_std').metric.unique():
            expected = -100*(ref.accuracy-ref.accuracy_sgd) if metric=='accuracy_loss_pp' else ref[metric]-ref[metric+'_sgd']
            close(row[metric], expected)
    budgets = read('budget_differences_seed')
    for _, row in budgets.iterrows():
        later, earlier = row.comparison.split(' minus ')
        a, b = [one(seeds, feature_policy=row.feature_policy, method=m, model_seed=row.model_seed) for m in [later, earlier]]
        for metric in read('budget_differences_mean_std').metric.unique():
            close(row[metric], a[metric]-b[metric])
    checks['paired_group_rows'] = len(actual)
    checks['paired_budget_rows'] = len(budgets)
    checks['complete_runs'] = len(seeds)
    return checks


class Evidence:
    def __init__(self):
        self.stats = read('metrics_mean_std')
        self.group = read('group_mean_std')
        self.delta = read('group_rate_changes_mean_std')
        self.paired = read('paired_vs_sgd_mean_std')
        self.budget = read('budget_differences_mean_std')
        self.raw_group = read('group_seed_metrics')
        self.raw_delta = read('group_rate_changes_seed')
        self.seeds = read('seed_metrics')

    def stat(self, method, metric, policy=PRIMARY):
        return one(self.stats, feature_policy=policy, method=method, metric=metric)

    def g(self, method, attribute, group, metric, policy=PRIMARY):
        return one(self.group, feature_policy=policy, method=method, attribute=attribute, group=group, metric=metric)

    def change(self, attribute, group, metric, policy=PRIMARY):
        return one(self.delta, feature_policy=policy, method='dp_epsilon_1', attribute=attribute, group=group, metric=metric)


def fmt(row, scale=100, precision=2, signed=False):
    mean = format(scale*row['mean'], f'+.{precision}f' if signed else f'.{precision}f')
    return f'{mean} +/- {scale*row["std"]:.{precision}f}'


def export_table(name, records):
    frame = pd.DataFrame(records)
    frame.to_csv(OUT/'tables'/f'{name}.csv', index=False)
    return frame


def latex_table(name, display, caption, label):
    def tex(value):
        value = (str(value).replace('&', r'\&').replace('%', r'\%').replace('_', r'\_')
                 .replace('+/-', r'$\pm$').replace('n>=500', r'$n\geq500$'))
        return re.sub(r'eps([148])', lambda m: r'$\epsilon='+m[1]+'$', value)
    content = [r'\begin{table}[t]', r'\centering', r'\small', r'\setlength{\tabcolsep}{3pt}',
               '\\begin{tabular}{'+'l'*len(display.columns)+'}', r'\toprule',
               ' & '.join(map(tex, display.columns))+r' \\', r'\midrule']
    content += [' & '.join(map(tex, row))+r' \\' for row in display.itertuples(index=False, name=None)]
    content += [r'\bottomrule', r'\end{tabular}', '\\caption{'+caption+'}', '\\label{'+label+'}', r'\end{table}']
    (OUT/'tables'/f'{name}.tex').write_text('\n'.join(content)+'\n', encoding='utf-8')


def tables(e):
    numeric, display = [], []
    for metric, label, unit in [('accuracy', 'Overall accuracy', '%'), ('accuracy_loss_pp', 'Paired accuracy loss', 'pp')]:
        row, shown = {'scope': 'Overall', 'metric': metric, 'unit': unit}, {'Comparison': 'Overall', 'Measure': label+f' ({unit})'}
        for method in ['sgd', 'dp_epsilon_1']:
            result = (dict(mean=0., std=0.) if method=='sgd' else one(e.paired, feature_policy=PRIMARY, method=method, metric=metric)) if metric=='accuracy_loss_pp' else e.stat(method, metric)
            scale = 1 if metric=='accuracy_loss_pp' else 100
            prefix = 'sgd' if method=='sgd' else 'dp1'
            row.update({prefix+'_mean': scale*result['mean'], prefix+'_sd': scale*result['std']})
            shown['SGD' if method=='sgd' else 'DP eps1'] = 'reference (0)' if method=='sgd' and metric=='accuracy_loss_pp' else fmt(result, scale)
        numeric.append(row); display.append(shown)
    for scope, title in [('SEX', 'Sex'), ('RAC1P_n500', 'Race n>=500')]:
        for metric, label in GAPS:
            row, shown = {'scope': scope, 'metric': metric, 'unit': 'pp'}, {'Comparison': title, 'Measure': label+' (pp)'}
            for method in ['sgd', 'dp_epsilon_1']:
                result = e.stat(method, scope+'_'+metric)
                prefix = 'sgd' if method=='sgd' else 'dp1'
                row.update({prefix+'_mean': 100*result['mean'], prefix+'_sd': 100*result['std']})
                shown['SGD' if method=='sgd' else 'DP eps1'] = fmt(result)
            numeric.append(row); display.append(shown)
    export_table('table1_sgd_vs_dp1', numeric)
    t1 = pd.DataFrame(display)
    latex_table('table1_sgd_vs_dp1', t1, 'Primary setting. Means $\\pm$ sample SD over five seeds. '
                'Positive paired accuracy loss means harm. All fairness gaps are in percentage points; smaller means closer to parity. '
                'Race gaps use the five groups with development $n\\geq500$.', 'tab:sgd-dp1')
    numeric, display = [], []
    for metric, label in [('accuracy', 'Accuracy (%)'), ('accuracy_loss_pp', 'Paired accuracy loss (pp)'),
                          ('SEX_separation', 'Sex separation (pp)'), ('RAC1P_n500_separation', 'Race separation (pp)')]:
        row, shown = {'metric': metric, 'unit': '%' if metric=='accuracy' else 'pp'}, {'Measure': label}
        for method in DP:
            result = one(e.paired, feature_policy=PRIMARY, method=method, metric=metric) if metric=='accuracy_loss_pp' else e.stat(method, metric)
            scale = 1 if metric=='accuracy_loss_pp' else 100
            eps = method.rsplit('_', 1)[1]
            row.update({f'eps{eps}_mean': scale*result['mean'], f'eps{eps}_sd': scale*result['std']})
            shown['eps'+eps] = fmt(result, scale, precision=3)
        numeric.append(row); display.append(shown)
    export_table('table2_privacy_budgets', numeric)
    t2 = pd.DataFrame(display)
    latex_table('table2_privacy_budgets', t2, 'Primary setting, fixed clipping norm $C=1$. Means $\\pm$ sample SD over five seeds. '
                'Paired losses use each initialization-matched SGD baseline. Race separation uses development $n\\geq500$ groups.', 'tab:budgets')
    return t1, t2


def figures(e):
    os.environ.setdefault('MPLCONFIGDIR', str(ROOT/'tmp/matplotlib'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MultipleLocator
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9, 'axes.titlesize': 10,
                         'axes.labelsize': 9, 'xtick.labelsize': 8.5, 'ytick.labelsize': 9,
                         'axes.spines.top': False, 'axes.spines.right': False,
                         'axes.linewidth': .7, 'pdf.fonttype': 42, 'ps.fonttype': 42, 'svg.fonttype': 'none',
                         'savefig.facecolor': 'white', 'figure.facecolor': 'white'})
    blue, orange, gray, green = '#2166AC', '#C65C28', '#586575', '#347B62'
    def save(fig, name):
        for extension in ['pdf', 'png', 'svg']:
            path = OUT/'figures'/f'{name}.{extension}'
            fig.savefig(path, dpi=350)
            if extension == 'svg':
                # Matplotlib emits trailing spaces inside multiline path attributes.
                path.write_text('\n'.join(line.rstrip() for line in path.read_text(encoding='utf-8').splitlines())+'\n', encoding='utf-8')
        plt.close(fig)
    points = []
    labels = []
    for group, name in zip(RACES, RACE_NAMES):
        n = one(e.raw_group, feature_policy=PRIMARY, method='sgd', model_seed=42, attribute='RAC1P', group=group).n
        labels.append(name+f'\n(n={n:,})')
        for metric in ['accuracy', 'fpr', 'selection_rate']:
            result = e.change('RAC1P', group, metric)
            points.append(dict(group=group, label=name, n=int(n), metric=metric,
                               mean_pp=(-100 if metric=='accuracy' else 100)*result['mean'],
                               sd_pp=100*result['std']))
    points = export_table('figure1_group_changes', points)
    fig, axes = plt.subplots(1, 3, figsize=(6.5, 3.0), sharey=True)
    fig.subplots_adjust(left=.255, right=.985, bottom=.19, top=.83, wspace=.22)
    for ax, metric, title, color, limit in zip(axes, ['accuracy', 'fpr', 'selection_rate'],
              ['(a) Accuracy loss', '(b) FPR increase', '(c) Selection increase'],
              [blue, orange, green], [5.5, 31, 21]):
        frame = points.loc[points.metric.eq(metric)]
        ax.axvline(0, color='#909090', lw=.8)
        ax.errorbar(frame.mean_pp, range(5), xerr=frame.sd_pp, fmt='o', color=color,
                    markersize=4.6, capsize=2.5, elinewidth=1.1, zorder=3)
        lower = -.35 if metric=='accuracy' else -1.5
        assert (frame.mean_pp-frame.sd_pp).min() > lower
        assert (frame.mean_pp+frame.sd_pp).max() < limit, 'Plot would clip an SD error bar'
        ax.set_xlim(lower, limit)
        ax.set_title(title, loc='left', pad=11, fontsize=9)
        ax.set_xlabel('Percentage points')
        ax.grid(axis='x', alpha=.18, zorder=0)
        ax.xaxis.set_major_locator(MultipleLocator(2 if metric=='accuracy' else 10 if metric=='fpr' else 5))
        ax.set_yticks(range(5), labels)
        ax.tick_params(axis='y', length=0)
    axes[0].invert_yaxis()
    fig.suptitle(r'Unequal group impact at $\epsilon=1$', x=.255, ha='left', fontsize=11, y=.97)
    fig.text(.255, .025, 'Primary setting | paired changes | mean '+r'$\pm$'+' sample SD, 5 seeds', fontsize=8.2, color=gray)
    save(fig, 'figure1_racial_impact')

    rows = []
    for method in METHODS:
        result = e.stat(method, 'accuracy')
        values = e.seeds.loc[e.seeds.feature_policy.eq(PRIMARY) & e.seeds.method.eq(method)].sort_values('model_seed')
        rows.append(dict(method=method, mean_percent=100*result['mean'], sd_percent=100*result['std']))
    acc = export_table('figure2_accuracy', rows)
    fig, axes = plt.subplots(1, 2, figsize=(6.5, 2.8), gridspec_kw={'width_ratios': [1.65, 1]})
    fig.subplots_adjust(left=.09, right=.985, bottom=.30, top=.81, wspace=.35)
    labels = ['SGD', 'Poisson\nunclipped', 'Poisson\nclipped', 'DP1', 'DP4', 'DP8']
    colors = [gray, green, orange, blue, blue, blue]
    for i, (method, color) in enumerate(zip(METHODS, colors)):
        values = e.seeds.loc[e.seeds.feature_policy.eq(PRIMARY) & e.seeds.method.eq(method)].sort_values('model_seed')
        axes[0].scatter(i+np.linspace(-.065, .065, len(values)), 100*values.accuracy, color=color, alpha=.45, s=10, zorder=2)
        axes[0].errorbar(i, acc.iloc[i].mean_percent, yerr=acc.iloc[i].sd_percent, fmt='o',
                         color=color, markersize=4.5, capsize=3, zorder=3)
    axes[0].set_xticks(range(6), labels)
    axes[0].set_ylim(74.6, 79.05)
    axes[0].set_yticks([75, 76, 77, 78, 79])
    axes[0].set_ylabel('Accuracy (%)')
    axes[0].set_title('(a) Sampling, clipping, and noise', loc='left', fontsize=9)
    axes[0].grid(axis='y', alpha=.18)
    clipping = acc.iloc[2].mean_percent-acc.iloc[1].mean_percent
    axes[0].annotate('', xy=(1.5, acc.iloc[2].mean_percent), xytext=(1.5, acc.iloc[1].mean_percent),
                     arrowprops=dict(arrowstyle='<->', color=orange, lw=1.1))
    axes[0].text(1.65, 77.9, f'Clipped vs unclipped: {clipping:.2f} pp', color=orange, fontsize=8.1,
                 ha='left', va='center')
    # Paired changes expose small budget effects without an unmarked, tightly zoomed absolute axis.
    base = e.seeds.loc[e.seeds.feature_policy.eq(PRIMARY) & e.seeds.method.eq('dp_epsilon_1')].set_index('model_seed')
    comparisons = []
    for i, method in enumerate(['dp_epsilon_4', 'dp_epsilon_8']):
        other = e.seeds.loc[e.seeds.feature_policy.eq(PRIMARY) & e.seeds.method.eq(method)].set_index('model_seed')
        values = 100*(other.accuracy-base.accuracy)
        comparisons.append(dict(comparison=method+' minus dp_epsilon_1', mean_pp=values.mean(), sd_pp=values.std(ddof=1)))
        axes[1].scatter(i+np.linspace(-.06, .06, 5), values, color=blue, alpha=.55, s=12, zorder=2)
        axes[1].errorbar(i, values.mean(), yerr=values.std(ddof=1), fmt='o', color=blue, markersize=5, capsize=4, zorder=3)
    axes[1].axhline(0, color=gray, lw=.8, linestyle='--')
    axes[1].set_xticks([0, 1], ['DP4 - DP1', 'DP8 - DP1'])
    axes[1].set(xlim=(-.5, 1.5), ylim=(-.16, .16), ylabel='Paired accuracy change (pp)')
    axes[1].set_yticks([-.15, -.10, -.05, 0, .05, .10, .15])
    axes[1].set_title('(b) Small budget differences', loc='left', fontsize=9)
    axes[1].grid(axis='y', alpha=.18)
    fig.suptitle('Accuracy across the training ablations', x=.09, ha='left', fontsize=11, y=.98)
    fig.text(.09, .07, 'Primary setting | mean '+r'$\pm$'+' sample SD, 5 seeds | dots: individual seeds', fontsize=8.2, color=gray)
    save(fig, 'figure2_accuracy_ablations')
    export_table('figure2_paired_budget_changes', comparisons)

    fig, axes = plt.subplots(1, 2, figsize=(6.5, 3.0), sharey=True)
    fig.subplots_adjust(left=.20, right=.985, bottom=.29, top=.80, wspace=.15)
    rows = []
    for ax, scope, title in zip(axes, ['SEX', 'RAC1P_n500'], ['(a) Sex', '(b) Supported race groups']):
        for method, offset, color, marker, name in [('sgd', -.12, gray, 's', 'SGD'), ('dp_epsilon_1', .12, blue, 'o', 'DP1')]:
            means, sd = [], []
            for metric, label in GAPS:
                result = e.stat(method, scope+'_'+metric)
                means.append(100*result['mean']); sd.append(100*result['std'])
                rows.append(dict(scope=scope, method=method, metric=metric, mean_pp=means[-1], sd_pp=sd[-1]))
            ax.errorbar(means, np.arange(4)+offset, xerr=sd, fmt=marker, color=color, markersize=4,
                        capsize=2.5, label=name, elinewidth=1)
        ax.set_yticks(range(4), [label for _, label in GAPS])
        ax.set_xlim(0, 35)
        ax.set_xticks([0, 10, 20, 30])
        ax.set_title(title, loc='left', fontsize=9)
        ax.grid(axis='x', alpha=.18)
        ax.tick_params(axis='y', length=0)
    axes[0].invert_yaxis()
    axes[1].legend(loc='lower right', fontsize=8, frameon=False)
    fig.suptitle('Fairness changes depend on attribute and criterion', x=.20, ha='left', fontsize=11, y=.98)
    fig.supxlabel('Fairness gap (pp); smaller means closer to parity', x=.59, y=.105, fontsize=9)
    fig.text(.20, .025, 'Primary setting | gaps per seed, then mean '+r'$\pm$'+' sample SD, 5 seeds', fontsize=8.2, color=gray)
    save(fig, 'figure3_fairness_optional')
    export_table('figure3_fairness_gaps', rows)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    for folder in ['figures', 'tables']:
        (OUT/folder).mkdir(exist_ok=True)
    before = frozen_hashes()
    checks = verify_sources()
    evidence = Evidence()
    tables(evidence)
    figures(evidence)
    after = frozen_hashes()
    assert before == after, 'Frozen source changed during generation'
    publish = {name: sha for name, sha in before.items() if '/runs/' not in name}
    dump(OUT/'source_provenance.json', {'source_sha256': publish, 'builder': 'scripts/build_report_assets.py',
                                      'builder_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()})
    dump(OUT/'validation.json', dict(checks, frozen_source_files_checked=len(before),
        all_frozen_sources_unchanged=True, training_executed=False, test_evaluated=False,
        score_diagnostics_executed=False, clipping_sensitivity_executed=False,
        main_figures=2, optional_figures=1, formats=['pdf', 'png', 'svg'], summary_tables=2,
        visual_review='Pending PDF rendering and human/model inspection after build'))
    print(f'Validated {checks["complete_runs"]} runs; built 2 main + 1 optional figures and 2 tables. Sources unchanged.')


if __name__ == '__main__':
    main()
