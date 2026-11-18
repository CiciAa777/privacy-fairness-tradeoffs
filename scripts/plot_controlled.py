"""Reusable plotting entry point. Default lists candidates; render only after review."""
from pathlib import Path
import argparse
import json
import os

ROOT = Path(__file__).resolve().parents[1]


def render(root, figure, policy):
    import pandas as pd
    os.environ.setdefault('MPLCONFIGDIR', str(ROOT/'tmp/matplotlib'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    output = ROOT/'figures/controlled_v2'
    output.mkdir(parents=True, exist_ok=True)
    if figure == 'accuracy_fairness':
        frame = pd.read_csv(root/'aggregate/metrics_mean_std.csv')
        fig, axes = plt.subplots(1, 3, figsize=(12, 3.6))
        for ax, metric in zip(axes, ['accuracy', 'SEX_separation', 'RAC1P_n500_separation']):
            subset = frame.loc[frame.feature_policy.eq(policy) & frame.metric.eq(metric)]
            private = subset.loc[subset.method.str.startswith('dp_epsilon_')].copy()
            private['epsilon'] = private.method.str.removeprefix('dp_epsilon_').astype(float)
            private = private.sort_values('epsilon')
            ax.errorbar(private.epsilon, 100*private['mean'], yerr=100*private['std'], marker='o', capsize=3)
            for method, style, color in [('sgd', '--', 'gray'), ('poisson_unclipped', ':', '#668866'),
                                         ('poisson_clipped', '-.', '#bb7744')]:
                baseline = subset.loc[subset.method.eq(method)]
                if len(baseline):
                    ax.axhline(100*baseline['mean'].iloc[0], linestyle=style, color=color, label=method)
            ax.set(title=metric, xlabel='Epsilon', ylabel='Percent / percentage points')
            ax.legend(fontsize=7)
    elif figure == 'group_loss':
        frame = pd.read_csv(root/'aggregate/group_accuracy_losses_mean_std.csv', dtype={'group': str})
        counts = pd.read_csv(root/'aggregate/group_seed_metrics.csv', dtype={'group': str})
        supported = set(counts.loc[counts.attribute.eq('RAC1P') & counts.n.ge(500), 'group'])
        names = {'SEX': {'1': 'Male', '2': 'Female'},
                 'RAC1P': {'1': 'White', '2': 'Black', '3': 'American Indian', '4': 'Alaska Native',
                           '5': 'AIAN code5', '6': 'Asian', '7': 'NHPI', '8': 'Some Other Race', '9': 'Two or More Races'}}
        fig, axes = plt.subplots(1, 2, figsize=(11, 4))
        for ax, attribute in zip(axes, ['SEX', 'RAC1P']):
            selected = frame.loc[frame.feature_policy.eq(policy) & frame.attribute.eq(attribute)]
            if attribute == 'RAC1P':
                selected = selected.loc[selected.group.isin(supported)]
            for group, subset in selected.groupby('group'):
                subset = subset.loc[subset.method.str.startswith('dp_epsilon_')].copy()
                subset['epsilon'] = subset.method.str.removeprefix('dp_epsilon_').astype(float)
                subset = subset.sort_values('epsilon')
                ax.errorbar(subset.epsilon, subset['mean'], yerr=subset['std'], marker='o', capsize=3,
                            label=names[attribute][group])
            ax.set(title='Sex groups' if attribute == 'SEX' else 'Race groups with n>=500',
                   xlabel='Epsilon', ylabel='Paired accuracy loss (pp)')
            ax.legend(fontsize=8)
    else:
        frame = pd.read_csv(root/'aggregate/group_mean_std.csv', dtype={'group': str})
        frame = frame.loc[frame.feature_policy.eq(policy) & frame.attribute.eq('SEX')]
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        for ax, metric in zip(axes, ['tpr', 'fpr']):
            for group, label in [('1', 'Male'), ('2', 'Female')]:
                subset = frame.loc[frame.group.eq(group) & frame.metric.eq(metric)].copy()
                order = ['sgd', 'poisson_unclipped', 'poisson_clipped', 'dp_epsilon_1', 'dp_epsilon_4', 'dp_epsilon_8']
                subset = subset.set_index('method').reindex(order)
                ax.errorbar(range(len(order)), 100*subset['mean'], yerr=100*subset['std'],
                            marker='o', capsize=3, label=label)
                ax.set_xticks(range(len(order)), order, rotation=30, ha='right', fontsize=8)
            ax.set(title=metric.upper(), ylabel='Rate (%)')
            ax.legend()
    fig.suptitle(f'{policy}; mean +/- sample SD across training seeds')
    fig.tight_layout()
    fig.savefig(output/f'{figure}_{policy}.png', dpi=250, bbox_inches='tight')
    plt.close(fig)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results', type=Path, default=ROOT/'results/controlled_v2')
    parser.add_argument('--render', choices=['accuracy_fairness', 'group_loss', 'sex_rates'])
    parser.add_argument('--policy', choices=['include_sensitive', 'exclude_sensitive'], default='include_sensitive')
    args = parser.parse_args()
    if args.render:
        render(args.results, args.render, args.policy)
    else:
        print('No figures rendered. After reviewing findings, select at most three candidates:')
        for item in json.loads((args.results/'plot_plan.json').read_text(encoding='utf-8'))['figures']:
            print(f"  {item['id']}: {item['description']}")
