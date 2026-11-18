"""Seed-level/paired aggregation of completed controlled runs; no model training."""
import json
from itertools import combinations

import numpy as np
import pandas as pd

from .baseline import ROOT
from .controlled_data import json_write
from .controlled_training import MAIN, CONTROLS, clean


def aggregate(frame, keys, metrics):
    """Sample SD is missing for <2 valid seeds; undefined rates never become 0."""
    if frame.empty:
        return pd.DataFrame(columns=keys+['metric', 'mean', 'std', 'valid_seeds', 'completed_seeds'])
    long = frame.melt(id_vars=keys+['model_seed'], value_vars=metrics, var_name='metric', value_name='value')
    result = long.groupby(keys+['metric'], dropna=False).value.agg(
        mean='mean', std='std', valid_seeds='count').reset_index()
    counts = frame.groupby(keys, dropna=False).model_seed.nunique().rename('completed_seeds').reset_index()
    return result.merge(counts, on=keys, validate='many_to_one')


def paired_difference(frame, metrics, reference='sgd'):
    """Method minus matched reference, joined by feature policy AND initialization."""
    keys = ['feature_policy', 'method', 'model_seed']
    if frame.duplicated(keys).any():
        raise ValueError('Duplicate method/policy/seed rows')
    base = frame.loc[frame.method.eq(reference), ['feature_policy', 'model_seed']+metrics]
    candidates = frame.loc[~frame.method.eq(reference)]
    if candidates.empty:
        return pd.DataFrame(columns=keys+metrics)
    joined = candidates.merge(base, on=['feature_policy', 'model_seed'],
                              suffixes=('', '_reference'), how='left', validate='many_to_one', indicator=True)
    if joined._merge.ne('both').any():
        raise ValueError('Missing same-policy, same-seed reference; refusing unpaired comparison')
    result = joined[keys].copy()
    for metric in metrics:
        result[metric] = joined[metric]-joined[metric+'_reference']
    return result


def markdown(frame):
    display = frame.astype(object).where(frame.notna(), 'undefined')
    return '\n'.join(['| '+' | '.join(map(str, frame.columns))+' |',
                     '| '+' | '.join(['---']*len(frame.columns))+' |']+
                    ['| '+' | '.join(map(str, row))+' |' for row in display.itertuples(index=False, name=None)])


def summarize_controlled(config):
    root = ROOT/config['output_dir']
    out = root/'aggregate'
    out.mkdir(parents=True, exist_ok=True)
    rows, groups, esr, losses = [], [], [], []
    summaries = []
    paths = sorted((root/'runs').glob('*/*/seed_*/summary.json'))
    for path in paths:
        s = json.loads(path.read_text(encoding='utf-8'))
        assert s['completed'] and not s['is_smoke'] and not s['test_evaluated']
        summaries.append(s)
        identity = {key: s[key] for key in ['feature_policy', 'method', 'model_seed']}
        row = dict(identity, **s['overall'])
        for key in ['epsilon', 'delta', 'noise_multiplier', 'clipping_norm', 'steps', 'seconds']:
            row[key] = s[key]
        for attribute, metrics in s['gaps'].items():
            for name, value in metrics.items():
                row[f'{attribute}_{name}'] = value
        for name, value in s['headline_race_gaps'].items():
            row[f'RAC1P_n500_{name}'] = value
        rows.append(row)
        for file, destination in [('group_metrics.csv', groups), ('esr_diagnostics.csv', esr),
                                  ('group_accuracy_losses.csv', losses)]:
            if (path.parent/file).exists():
                destination.append(pd.read_csv(path.parent/file, dtype={'group': str}).assign(**identity))
    if not rows:
        raise ValueError('No completed main/control runs to aggregate')
    for key in ['protocol', 'environment', 'data_signature', 'n_train', 'n_development']:
        if len({json.dumps(summary[key], sort_keys=True) for summary in summaries}) != 1:
            raise ValueError(f'Inconsistent {key} across completed runs; refusing to mix protocols/environments')
    for policy in config['feature_policies']:
        for seed in config['seeds']:
            matched = [summary for summary in summaries
                       if summary['feature_policy'] == policy and summary['model_seed'] == seed]
            if len({summary['initialization_sha256'] for summary in matched}) > 1:
                raise ValueError(f'Unmatched initialization for {policy}/seed_{seed}')
    seeds = pd.DataFrame(rows)
    if seeds.duplicated(['feature_policy', 'method', 'model_seed']).any():
        raise ValueError('Duplicate completed run')
    seeds.to_csv(out/'seed_metrics.csv', index=False)
    group_frame = pd.concat(groups, ignore_index=True)
    esr_frame = pd.concat(esr, ignore_index=True)
    group_frame.to_csv(out/'group_seed_metrics.csv', index=False)
    esr_frame.to_csv(out/'esr_seed_metrics.csv', index=False)
    group_changes = paired_group_rates(group_frame)
    group_changes.to_csv(out/'group_rate_changes_seed.csv', index=False)
    aggregate(group_changes, ['feature_policy', 'method', 'attribute', 'group'], GROUP_RATES).to_csv(
        out/'group_rate_changes_mean_std.csv', index=False)
    base_keys = ['feature_policy', 'method']
    overall_metrics = ['accuracy', 'balanced_accuracy', 'roc_auc', 'bce', 'positive_rate', 'selection_rate',
                       'tpr', 'fpr', 'ppv', 'npv']
    fairness_metrics = [name for name in seeds if any(name.startswith(prefix) for prefix in
                        ['SEX_', 'RAC1P_', 'RAC1P_SEX_'])]
    metrics = overall_metrics+fairness_metrics
    for metric in metrics:
        seeds[metric] = pd.to_numeric(seeds[metric], errors='raise')
    stats = aggregate(seeds, base_keys, metrics)
    stats.to_csv(out/'metrics_mean_std.csv', index=False)
    json_write(out/'metrics_mean_std.json', clean(stats.to_dict('records')))
    group_metrics = ['accuracy', 'balanced_accuracy', 'selection_rate', 'tpr', 'fpr', 'ppv', 'npv',
                     'actual_positives', 'actual_negatives', 'tp', 'tn', 'fp', 'fn']
    aggregate(group_frame, base_keys+['attribute', 'group'], group_metrics).to_csv(out/'group_mean_std.csv', index=False)
    aggregate(esr_frame, base_keys+['attribute', 'group', 'ESR'],
              ['n', 'actual_negatives', 'false_positives', 'fpr', 'fp_composition']).to_csv(out/'esr_mean_std.csv', index=False)
    paired = paired_difference(seeds, metrics)
    paired['accuracy_loss_pp'] = -100*paired.accuracy
    paired.to_csv(out/'paired_vs_sgd_seed.csv', index=False)
    paired_stats = aggregate(paired, base_keys, metrics+['accuracy_loss_pp'])
    paired_stats.to_csv(out/'paired_vs_sgd_mean_std.csv', index=False)
    if losses:
        loss_frame = pd.concat(losses, ignore_index=True)
        loss_frame.to_csv(out/'group_accuracy_losses_seed.csv', index=False)
        aggregate(loss_frame, base_keys+['attribute', 'group'], ['accuracy_loss_pp']).to_csv(
            out/'group_accuracy_losses_mean_std.csv', index=False)
        disparities = []
        for keys, block in loss_frame.groupby(base_keys+['model_seed', 'attribute']):
            policy, method, seed, attribute = keys
            subsets = [(attribute, block)]
            if attribute == 'RAC1P':
                subsets.append(('RAC1P_n500', block.loc[block.n.ge(config['minimum_headline_race_n'])]))
            for label, subset in subsets:
                values = subset.accuracy_loss_pp
                disparities.append(dict(feature_policy=policy, method=method, model_seed=seed,
                    attribute=label, min_group_loss_pp=values.min() if values.notna().all() else np.nan,
                    max_group_loss_pp=values.max() if values.notna().all() else np.nan,
                    loss_disparity_pp=values.max()-values.min() if values.notna().all() else np.nan))
        disparity_frame = pd.DataFrame(disparities)
        disparity_frame.to_csv(out/'loss_disparity_seed.csv', index=False)
        aggregate(disparity_frame, base_keys+['attribute'],
                  ['min_group_loss_pp', 'max_group_loss_pp', 'loss_disparity_pp']).to_csv(out/'loss_disparity_mean_std.csv', index=False)
    budget_rows = []
    for a, b in combinations([f'dp_epsilon_{epsilon:g}' for epsilon in config['target_epsilons']], 2):
        subset = seeds.loc[seeds.method.isin([a, b])]
        differences = paired_difference(subset, metrics, reference=a)
        differences['comparison'] = f'{b} minus {a}'
        budget_rows.append(differences)
    budgets = pd.concat(budget_rows, ignore_index=True)
    budgets.to_csv(out/'budget_differences_seed.csv', index=False)
    ordering = []
    for (policy, comparison), block in budgets.groupby(['feature_policy', 'comparison']):
        differences = 100*block.accuracy.dropna()
        ordering.append(dict(feature_policy=policy, comparison=comparison, valid_seeds=len(differences),
            later_budget_higher_seeds=int((differences > 0).sum()), later_budget_lower_seeds=int((differences < 0).sum()),
            tied_seeds=int((differences == 0).sum()), accuracy_change_pp=differences.mean(),
            accuracy_change_sample_sd_pp=differences.std(ddof=1)))
    pd.DataFrame(ordering).to_csv(out/'budget_accuracy_ordering.csv', index=False)
    budget_stats = aggregate(budgets, ['feature_policy', 'comparison'], metrics)
    budget_stats.to_csv(out/'budget_differences_mean_std.csv', index=False)
    feature_rows = []
    for method, block in seeds.groupby('method'):
        excluded = block.loc[block.feature_policy.eq('exclude_sensitive')]
        included = block.loc[block.feature_policy.eq('include_sensitive')]
        joined = included.merge(excluded, on=['method', 'model_seed'], suffixes=('', '_excluded'), validate='one_to_one')
        for _, row in joined.iterrows():
            feature_rows.append(dict(method=method, model_seed=int(row.model_seed),
                **{metric: row[metric]-row[metric+'_excluded'] for metric in metrics}))
    features = pd.DataFrame(feature_rows, columns=['method', 'model_seed']+metrics)
    features.to_csv(out/'feature_policy_differences_seed.csv', index=False)
    aggregate(features, ['method'], metrics).to_csv(out/'feature_policy_differences_mean_std.csv', index=False)
    sex = group_frame.loc[group_frame.attribute.eq('SEX')].pivot(
        index=base_keys+['model_seed'], columns='group', values='fpr').reset_index()
    sex['female_minus_male_fpr_pp'] = 100*(sex['2']-sex['1'])
    sex.to_csv(out/'sex_fpr_difference_seed.csv', index=False)
    persistence = sex.groupby(base_keys).female_minus_male_fpr_pp.agg(
        valid_seeds='count', female_higher_seeds=lambda values: int((values > 0).sum()), mean='mean', std='std').reset_index()
    persistence.to_csv(out/'sex_fpr_persistence.csv', index=False)
    within_esr = esr_frame.loc[esr_frame.attribute.eq('SEX')].pivot(
        index=base_keys+['model_seed', 'ESR'], columns='group', values='fpr').reset_index()
    within_esr['female_minus_male_fpr_pp'] = 100*(within_esr['2']-within_esr['1'])
    within_esr.to_csv(out/'within_esr_sex_fpr_difference_seed.csv', index=False)
    aggregate(within_esr, base_keys+['ESR'], ['female_minus_male_fpr_pp']).to_csv(
        out/'within_esr_sex_fpr_difference_mean_std.csv', index=False)
    expected = {(policy, method, seed) for policy in config['feature_policies']
                for method in MAIN+CONTROLS for seed in config['seeds']}
    present = {(s['feature_policy'], s['method'], s['model_seed']) for s in summaries}
    missing = sorted(expected-present)
    private = [s for s in summaries if s['private_training_mechanism']]
    v0 = json.loads((ROOT/'V0/results/dp_pilot/composition.json').read_text(encoding='utf-8'))
    status = dict(expected_main_runs=len(config['feature_policies'])*len(config['seeds'])*len(MAIN),
        expected_control_runs=len(config['feature_policies'])*len(config['seeds'])*len(CONTROLS), completed_runs=len(summaries),
        completed_main_runs=sum(s['method'] in MAIN for s in summaries),
        completed_control_runs=sum(s['method'] in CONTROLS for s in summaries),
        complete=not missing, missing=[dict(feature_policy=p, method=m, model_seed=s) for p, m, s in missing],
        seed_variation='Training/initialization/sampling/noise variation on a fixed split, not population uncertainty',
        final_test_evaluated=False)
    json_write(out/'completion.json', status)
    json_write(out/'privacy_composition.json', dict(
        private_main_runs=len(private), basic_composition_epsilon=sum(s['epsilon'] for s in private),
        basic_composition_delta=sum(s['delta'] for s in private),
        with_v0_basic_composition_epsilon=sum(s['epsilon'] for s in private)+v0['basic_composition_epsilon'],
        with_v0_basic_composition_delta=sum(s['delta'] for s in private)+v0['basic_composition_delta'],
        excludes_smoke='Smoke trains on public reference records only',
        includes_v0=False, v0_note='If released jointly with V0 private outputs, their bounds also compose',
        whole_repository_private=False, nonprivate_models_and_audits='Not DP releases; per-run bounds do not cover them'))
    specs = [dict(id='accuracy_fairness', metrics=['accuracy', 'SEX_separation', 'RAC1P_n500_separation'],
                  source='aggregate/metrics_mean_std.csv', description='Accuracy and selected fairness gaps versus epsilon'),
             dict(id='group_loss', metrics=['accuracy_loss_pp'], source='aggregate/group_accuracy_losses_mean_std.csv',
                  description='Paired DP-versus-SGD accuracy losses by supported group'),
             dict(id='sex_rates', metrics=['tpr', 'fpr'], source='aggregate/group_mean_std.csv',
                  description='Male/female TPR and FPR with ordinary and clipping-only controls')]
    json_write(root/'plot_plan.json', dict(rendered=False, review_required_before_publication=True,
               variability='Error bars represent sample SD across seeds, not confidence intervals', figures=specs))
    write_findings(config, root, seeds, stats, paired_stats, budget_stats, persistence, status)
    (out/'README.md').write_text('''# Aggregate table conventions

- `seed_metrics.csv`: one row per completed policy/method/initialization seed. Raw rates and fairness gaps are fractions; multiply by100 for percent or percentage points.
- `metrics_mean_std.csv/json`: mean, sample SD (ddof=1), valid seed count, and completed seed count for each metric. Undefined metrics remain missing, never zero. Means use valid values only; a metric with fewer valid than completed seeds does not describe every seed.
- `paired_vs_sgd_*`: method minus same-policy/same-seed SGD. `accuracy_loss_pp` reverses the sign and multiplies by100, so positive means harm. Other differences retain their original units.
- `group_*`: all sex, nine race, and race-by-sex groups; counts, rates, and denominator flags are retained. Headline race gaps use a separately labeled n>=500 comparison.
- `group_rate_changes_*`: group-specific rate differences (method minus matched SGD, in fractions). Convert to percentage points by multiplying by100. These are paired before calculating sample SD. See `../GROUP_COMPARISONS.md` for readable levels and changes across all budgets.
- `budget_differences_*`: later-listed DP budget minus earlier-listed budget, paired by initialization within policy. Fresh sampling/noise still vary.
- `feature_policy_differences_*`: include_sensitive minus exclude_sensitive at the same method/seed. Input dimensions differ, so identical weights across policies are not claimed.
- `loss_disparity_*`: group loss ranges computed per run before averaging; columns ending `_pp` are already percentage points.
- `esr_*`, `within_esr_*`, and `sex_fpr_*`: development-only ESR distributions/errors, FP composition, and female-minus-male FPR comparisons. Counts are observations per fixed development set, not additional people across seeds. ESR1 is positive, so its FPR is undefined.
- `completion.json`: exact observed run counts and missing configurations. Missing configurations are not fabricated.
- `privacy_composition.json`: conservative basic composition for released private v2 mechanisms, plus a separate V0+v2 bound. Public-reference smoke runs do not train on the private partition. Non-private models/audits are outside those guarantees.

Seed variation is training variation on this fixed split, not population-sampling uncertainty. The human-readable findings table converts units for display. No publication figures are generated by aggregation.
''', encoding='utf-8')
    print(f'Aggregated {len(summaries)} completed runs; missing={len(missing)}. No figures rendered.', flush=True)
    return status


GROUP_RATES = ['accuracy', 'balanced_accuracy', 'selection_rate', 'tpr', 'fpr', 'ppv', 'npv']
GROUP_NAMES = {
    'SEX': {'1': 'Male', '2': 'Female'},
    'RAC1P': {'1': 'White alone', '2': 'Black or African American alone',
              '3': 'American Indian alone', '4': 'Alaska Native alone',
              '5': 'AIAN specified/unspecified; no other races', '6': 'Asian alone',
              '7': 'Native Hawaiian / Other Pacific Islander alone',
              '8': 'Some Other Race alone', '9': 'Two or More Races'},
}


def paired_group_rates(groups):
    """Compare the same people/group/initialization, preserving undefined rates."""
    keys = ['feature_policy', 'model_seed', 'attribute', 'group']
    if groups.duplicated(keys+['method']).any():
        raise ValueError('Duplicate group/policy/method/seed rows')
    baseline = groups.loc[groups.method.eq('sgd'), keys+['n']+GROUP_RATES]
    joined = groups.loc[groups.method.ne('sgd')].merge(
        baseline, on=keys, how='left', suffixes=('', '_sgd'), validate='many_to_one', indicator=True)
    if joined._merge.ne('both').any() or joined.n.ne(joined.n_sgd).any():
        raise ValueError('Missing or inconsistent same-group SGD reference')
    result = joined[keys+['method']].copy()
    for metric in GROUP_RATES:
        result[metric] = joined[metric]-joined[metric+'_sgd']
    return result


def group_label(attribute, group):
    if attribute == 'RAC1P_SEX':
        race, sex = group.split('_')
        return GROUP_NAMES['RAC1P'][race]+' / '+GROUP_NAMES['SEX'][sex]
    return GROUP_NAMES[attribute][group]


def write_group_findings(config, root, stats, paired):
    """Put group harm and fairness first; retain every group in a readable appendix."""
    out = root/'aggregate'
    groups = pd.read_csv(out/'group_mean_std.csv', dtype={'group': str})
    raw = pd.read_csv(out/'group_seed_metrics.csv', dtype={'group': str})
    changes = pd.read_csv(out/'group_rate_changes_mean_std.csv', dtype={'group': str})
    losses = pd.read_csv(out/'group_accuracy_losses_mean_std.csv', dtype={'group': str})
    paired_raw = pd.read_csv(out/'paired_vs_sgd_seed.csv')
    methods = ['sgd']+[f'dp_epsilon_{eps:g}' for eps in config['target_epsilons']]
    primary = config['feature_policies'][0]

    def lookup(frame, **identity):
        selected = frame
        for column, value in identity.items():
            selected = selected.loc[selected[column].eq(value)]
        if len(selected) != 1:
            raise ValueError(f'Expected one statistic for {identity}, found {len(selected)}')
        return selected.iloc[0]

    def show(row, scale=100, signed=False):
        if pd.isna(row['mean']):
            return 'undefined'
        mean = format(scale*row['mean'], '+.2f' if signed else '.2f')
        sd = 'undefined' if pd.isna(row['std']) else f'{scale*row["std"]:.2f}'
        count = '' if row.valid_seeds == len(config['seeds']) else f' (n={int(row.valid_seeds)})'
        return f'{mean} +/- {sd}{count}'

    def gs(policy, method, attribute, group, metric):
        return lookup(groups, feature_policy=policy, method=method, attribute=attribute, group=group, metric=metric)

    def mean(policy, method, attribute, group, metric):
        return 100*gs(policy, method, attribute, group, metric)['mean']

    def rate_pair(policy, attribute, group, metric):
        a, b = [mean(policy, method, attribute, group, metric) for method in ['sgd', 'dp_epsilon_1']]
        return 'undefined' if pd.isna(a) or pd.isna(b) else f'{a:.2f} -> {b:.2f}'

    support = raw.loc[raw.feature_policy.eq(primary) & raw.method.eq('sgd') &
                      raw.model_seed.eq(config['seeds'][0])].set_index(['attribute', 'group'])
    summary = [
        '## What DP changes for different groups\n',
        'The main comparison is **DP-SGD versus non-private SGD on the same feature policy, split, and initialization seed**. '
        f'The primary setting is `{primary}` (sex and race are available as input features). '
        'The secondary setting repeats the comparison without those two predictors. '
        'Every reported group is still evaluated under both settings.\n',
        'Read accuracy and disparity together: two groups can both lose accuracy while their gap shrinks. '
        'A smaller fairness gap alone does not mean either group became better off. '
        'All results below use the fixed development set and five training seeds.\n',
        '### Which groups lose accuracy?\n',
        'Accuracy is percent mean +/- sample SD. Loss is **SGD accuracy minus DP accuracy**, '
        'in percentage points (pp), paired by seed before averaging. Positive loss means worse accuracy under DP. '
        'The three loss columns cover every tested privacy budget.\n',
    ]
    for policy in config['feature_policies']:
        records = []
        for attribute in ['SEX', 'RAC1P']:
            for group in GROUP_NAMES[attribute]:
                n = int(support.loc[(attribute, group), 'n'])
                row = {'Group': group_label(attribute, group), 'People': n,
                       'SGD accuracy (%)': show(gs(policy, 'sgd', attribute, group, 'accuracy')),
                       'DP1 accuracy (%)': show(gs(policy, 'dp_epsilon_1', attribute, group, 'accuracy'))}
                for method in methods[1:]:
                    loss = lookup(losses, feature_policy=policy, method=method, attribute=attribute,
                                  group=group, metric='accuracy_loss_pp')
                    row[f'Loss eps{method.rsplit("_", 1)[1]} (pp)'] = show(loss, scale=1, signed=True)
                records.append(row)
        summary += [f'#### {policy}\n', markdown(pd.DataFrame(records)), '\n']
        supported = losses.loc[losses.feature_policy.eq(policy) & losses.method.eq('dp_epsilon_1') &
                               losses.attribute.eq('RAC1P') & losses.group.isin(
                                   [g for g in GROUP_NAMES['RAC1P'] if support.loc[('RAC1P', g), 'n'] >= config['minimum_headline_race_n']])]
        worst = supported.loc[supported['mean'].idxmax()]
        male, female = [lookup(losses, feature_policy=policy, method='dp_epsilon_1', attribute='SEX',
                              group=g, metric='accuracy_loss_pp')['mean'] for g in ['1', '2']]
        summary += [f'At eps1, male accuracy loses {male:.2f} pp and female accuracy loses {female:.2f} pp. '
                    f'Among race groups with at least {config["minimum_headline_race_n"]} people, '
                    f'{group_label("RAC1P", worst.group)} has the largest mean loss ({worst["mean"]:.2f} pp). '
                    'These are observed training-seed averages, not a claim that a particular group always loses most.\n']
    summary += ['Race categories are original ACS categories; none were merged. Small groups remain visible. '
                'In particular, the two-person Alaska Native-alone group has no actual positives, so its 100% accuracy '
                'does not establish good predictive performance.\n',
                '### What each fairness gap asks\n',
                markdown(pd.DataFrame([
                    {'Metric': 'Accuracy parity', 'Rates that should match': 'Accuracy across groups', 'Reported gap': 'max accuracy - min accuracy'},
                    {'Metric': 'Independence', 'Rates that should match': 'Predicted-positive / selection rate', 'Reported gap': 'max selection rate - min selection rate'},
                    {'Metric': 'Separation (equalized odds)', 'Rates that should match': 'TPR and FPR separately', 'Reported gap': 'larger of TPR range and FPR range'},
                    {'Metric': 'Binary sufficiency', 'Rates that should match': 'PPV and NPV separately', 'Reported gap': 'larger of PPV range and NPV range'},
                ])), '\n',
                'For every gap, **0 means parity and smaller means closer to that fairness criterion**. '
                'This binary sufficiency diagnostic concerns thresholded predictions, not full score calibration. '
                'Gaps are computed within each seed before averaging. They need not equal the gap between averaged rates.\n',
                '### Did DP improve or worsen these gaps?\n',
                'Each level is mean +/- sample SD in pp. The signed change is DP1 minus matched SGD: '
                '**positive means a larger gap; negative means a smaller gap**. '
                'Sex uses both groups. Headline race uses only the fixed five groups with development n>=500.\n']
    for policy in config['feature_policies']:
        records = []
        for scope, title in [('SEX', 'Sex'), ('RAC1P_n500', 'Race n>=500')]:
            for metric, label in [('accuracy_gap', 'Accuracy parity'), ('independence', 'Independence'),
                                  ('separation', 'Separation'), ('sufficiency', 'Binary sufficiency')]:
                key = f'{scope}_{metric}'
                row = {'Comparison': title, 'Metric': label}
                for method in methods:
                    row['SGD' if method == 'sgd' else 'DP eps'+method.rsplit('_', 1)[1]] = show(
                        lookup(stats, feature_policy=policy, method=method, metric=key))
                delta = lookup(paired, feature_policy=policy, method='dp_epsilon_1', metric=key)
                row['DP1 change (pp)'] = show(delta, signed=True)
                values = paired_raw.loc[paired_raw.feature_policy.eq(policy) & paired_raw.method.eq('dp_epsilon_1'), key].dropna()
                row['Smaller / larger gap seeds'] = f'{int((values < 0).sum())} / {int((values > 0).sum())} of {len(values)}'
                records.append(row)
        summary += [f'#### {policy}\n', markdown(pd.DataFrame(records)), '\n']
    for policy in config['feature_policies']:
        for scope, title in [('SEX', 'sex'), ('RAC1P_n500', 'headline race')]:
            directions = {'smaller': [], 'larger': [], 'unchanged': []}
            for metric, label in [('accuracy_gap', 'accuracy parity'), ('independence', 'independence'),
                                  ('separation', 'separation'), ('sufficiency', 'binary sufficiency')]:
                change = lookup(paired, feature_policy=policy, method='dp_epsilon_1', metric=f'{scope}_{metric}')['mean']
                directions['smaller' if change < 0 else 'larger' if change > 0 else 'unchanged'].append(label)
            description = '; '.join(f'{direction} mean gaps: {", ".join(labels)}' for direction, labels in directions.items() if labels)
            summary += [f'- {policy}, {title}, DP1 versus SGD: {description}.\n']
    summary += ['The answer therefore depends on the fairness definition and group comparison.\n',
                '### Which actual rates drive the change? Primary setting, SGD -> DP1\n',
                'Entries show mean rates in percent. Selection is the fraction predicted positive. '
                'TPR is the fraction of actual positives correctly predicted positive. '
                'FPR is the fraction of actual negatives incorrectly predicted positive. '
                'PPV is the fraction of predicted positives that are actually positive. '
                'NPV is the fraction of predicted negatives that are actually negative. '
                'For predictive performance, higher TPR/PPV/NPV and lower FPR are desirable; '
                'for the corresponding fairness criterion, groups should have matching rates.\n']
    rows = []
    for attribute in ['SEX', 'RAC1P']:
        for group in GROUP_NAMES[attribute]:
            if attribute == 'RAC1P' and support.loc[(attribute, group), 'n'] < config['minimum_headline_race_n']:
                continue
            rows.append({'Group': group_label(attribute, group), **{label: rate_pair(primary, attribute, group, metric)
                        for metric, label in [('selection_rate', 'Selection'), ('tpr', 'TPR'), ('fpr', 'FPR'), ('ppv', 'PPV'), ('npv', 'NPV')]}})
    summary += [markdown(pd.DataFrame(rows)), '\n']
    for attribute, a, b, label in [('SEX', '1', '2', 'Male versus female'), ('RAC1P', '1', '2', 'White-alone versus Black-alone')]:
        for metric in ['tpr', 'fpr']:
            old = abs(mean(primary, 'sgd', attribute, a, metric)-mean(primary, 'sgd', attribute, b, metric))
            new = abs(mean(primary, 'dp_epsilon_1', attribute, a, metric)-mean(primary, 'dp_epsilon_1', attribute, b, metric))
            summary += [f'- {label}: the absolute difference between mean {metric.upper()} rates changes '
                        f'from {old:.2f} to {new:.2f} pp.\n']
    summary += ['Both sex groups have higher TPR **and higher FPR** under DP1. The smaller sex separation gap '
                'therefore coexists with more false positives and lower accuracy. White-alone and Black-alone '
                'are one concrete pair, not a reference population or the entire multi-race comparison. '
                'The headline race separation range covers all five supported groups.\n',
                '**Full group tables:** [GROUP_COMPARISONS.md](GROUP_COMPARISONS.md) includes both policies, '
                'all three budgets, all nine race groups and all 18 race-by-sex intersections, '
                'with denominator support, rate levels, and paired changes.\n']

    detail = ['# DP versus non-private SGD: full group comparisons\n',
              '[Return to findings](FINDINGS.md). Development set only, fixed threshold 0.5. '
              'Levels are percentages; changes are percentage points. Each cell is mean +/- sample SD across five seeds. '
              'Changes use same-policy, same-group, same-initialization SGD before averaging. '
              'Positive accuracy change means improvement (opposite to the positive-loss convention in FINDINGS). '
              'Positive FPR change means more false positives. Equal selection/TPR/FPR/PPV/NPV across groups '
              'is the relevant fairness goal; a rate change within one group is not itself a fairness gap.\n',
              '## Fixed group support\n',
              'Counts describe people in one development set. Actual-positive and actual-negative counts are '
              'the TPR and FPR denominators. Predicted-positive and predicted-negative counts (PPV/NPV denominators) '
              'vary by model and seed. The smallest of these four conditional denominators across seeds is shown '
              'in each levels table. Counts below30 give weak support. Undefined rates retain missing values.\n']
    for attribute in ['SEX', 'RAC1P', 'RAC1P_SEX']:
        block = support.loc[attribute]
        detail += [f'### {attribute}\n', markdown(pd.DataFrame([
            {'Group': group_label(attribute, group), 'People': int(row.n), 'Actual positive': int(row.actual_positives),
             'Actual negative': int(row.actual_negatives), 'Actual positive (%)': f'{100*row.positive_rate:.2f}' if row.n else 'undefined'}
            for group, row in block.iterrows()])), '\n']
    for policy in config['feature_policies']:
        detail += [f'## {policy}\n']
        for attribute in ['SEX', 'RAC1P', 'RAC1P_SEX']:
            detail += [f'### {attribute}\n']
            for method in methods:
                rows = []
                subset = raw.loc[raw.feature_policy.eq(policy) & raw.method.eq(method) & raw.attribute.eq(attribute)]
                for group, block in subset.groupby('group', sort=False):
                    rows.append({'Group': group_label(attribute, group),
                        **{label: show(gs(policy, method, attribute, group, metric)) for metric, label in
                           [('accuracy', 'Accuracy'), ('selection_rate', 'Selection'), ('tpr', 'TPR'), ('fpr', 'FPR'), ('ppv', 'PPV'), ('npv', 'NPV')]},
                        'Min conditional n': int(block.min_conditional_count.min())})
                detail += [f'#### {method}: levels (%)\n', markdown(pd.DataFrame(rows)), '\n']
                if method != 'sgd':
                    rows = []
                    subset = changes.loc[changes.feature_policy.eq(policy) & changes.method.eq(method) & changes.attribute.eq(attribute)]
                    for group, block in subset.groupby('group', sort=False):
                        rows.append({'Group': group_label(attribute, group), **{label: show(block.loc[block.metric.eq(metric)].iloc[0], signed=True)
                            for metric, label in [('accuracy', 'Accuracy change'), ('selection_rate', 'Selection change'),
                                ('tpr', 'TPR change'), ('fpr', 'FPR change'), ('ppv', 'PPV change'), ('npv', 'NPV change')]}})
                    detail += [f'#### {method} minus matched SGD (pp)\n', markdown(pd.DataFrame(rows)), '\n']
    detail += ['## Source tables and reproduction\n',
               'Levels: `aggregate/group_seed_metrics.csv` and `aggregate/group_mean_std.csv`. '
               'Paired changes: `aggregate/group_rate_changes_seed.csv` and `aggregate/group_rate_changes_mean_std.csv`. '
               'Accuracy losses: `aggregate/group_accuracy_losses_mean_std.csv`. '
               'All raw rate/change CSV values are fractions except explicitly named `_pp` columns. '
               'Mean fairness gaps in FINDINGS are computed per seed, not from rounded table entries. '
               'Regenerate with `scripts/run_controlled.py --stage summarize`; no retraining or test evaluation is needed.\n']
    (root/'GROUP_COMPARISONS.md').write_text('\n'.join(detail), encoding='utf-8')
    return summary


def write_findings(config, root, seeds, stats, paired, budget_stats, persistence, status):
    def stat(policy, method, metric):
        row = stats.loc[stats.feature_policy.eq(policy) & stats.method.eq(method) & stats.metric.eq(metric)]
        return row.iloc[0] if len(row) else None
    def formatted(row, scale=100):
        if row is None:
            return 'not yet completed'
        if pd.isna(row['mean']):
            return 'undefined'
        precision = 4 if scale == 1 else 3
        sd = 'undefined' if pd.isna(row['std']) else f'{scale*row["std"]:.{precision}f}'
        return f'{scale*row["mean"]:.{precision}f} +/- {sd} (n={int(row["valid_seeds"])})'
    text = ['# Controlled v2: baseline findings\n',
            f'Completed main runs: **{status["completed_main_runs"]}/{status["expected_main_runs"]}**; controls: **{status["completed_control_runs"]}/{status["expected_control_runs"]}**. '
            'Development data only. Means +/- sample SD across training seeds; no population-sampling uncertainty is estimated.\n',
            '## Overall accuracy and selected fairness outcomes\n']
    if status['complete']:
        headline = [f'## What persists across the {len(config["seeds"])} seeds\n']
        for policy in config['feature_policies']:
            unclipped = stat(policy, 'poisson_unclipped', 'accuracy')['mean']
            clipped = stat(policy, 'poisson_clipped', 'accuracy')['mean']
            dp1 = stat(policy, 'dp_epsilon_1', 'accuracy')['mean']
            headline += [f'- **{policy}:** adding clipping to matched non-private Poisson SGD changes '
                         f'mean accuracy by {100*(clipped-unclipped):+.3f} pp; DP1 minus the clipped/no-noise '
                         f'control is {100*(dp1-clipped):+.3f} pp. '
                         'These contrasts identify where the measured utility changes occur.\n']
            private = pd.read_csv(root/'aggregate/paired_vs_sgd_seed.csv')
            selected = private.loc[private.feature_policy.eq(policy) & private.method.eq('dp_epsilon_1')]
            headline += [f'- **{policy}, DP1 versus SGD:** sex separation improves in '
                         f'{int((selected.SEX_separation < 0).sum())}/{len(selected)} seeds; '
                         f'headline race separation worsens in {int((selected.RAC1P_n500_separation > 0).sum())}/{len(selected)}. '
                         'A single statement that DP improved or worsened every notion/group would be inaccurate.\n']
        budget_order = pd.read_csv(root/'aggregate/budget_accuracy_ordering.csv')
        for policy in config['feature_policies']:
            fpr = persistence.loc[persistence.feature_policy.eq(policy) & persistence.method.eq('dp_epsilon_1')].iloc[0]
            order = budget_order.loc[budget_order.feature_policy.eq(policy) &
                                      budget_order.comparison.eq('dp_epsilon_8 minus dp_epsilon_1')].iloc[0]
            headline += [f'- **{policy}, FPR and budget persistence:** female FPR is higher in '
                         f'{int(fpr.female_higher_seeds)}/{int(fpr.valid_seeds)} DP1 seeds. '
                         f'Epsilon8 accuracy is higher than epsilon1 in {int(order.later_budget_higher_seeds)}/'
                         f'{int(order.valid_seeds)} seeds and lower in {int(order.later_budget_lower_seeds)}/'
                         f'{int(order.valid_seeds)}. Mean change is {order.accuracy_change_pp:+.3f} pp. '
                         'Use these observed signs rather than dismissing every budget difference as random noise.\n']
        headline += ['Ten-seed follow-up is recommended before final generalizations. Headline race gaps use the fixed '
                     'n>=500 development groups (White-alone, Black-alone, Asian-alone, Some Other Race-alone, '
                     'Two or More Races). All nine races/intersections remain in detailed CSVs.\n']
        text[2:2] = write_group_findings(config, root, stats, paired)+headline
    for policy in config['feature_policies']:
        records = []
        for method in MAIN+CONTROLS:
            records.append(dict(Method=method,
                **{'Accuracy (%)': formatted(stat(policy, method, 'accuracy')),
                   'Balanced accuracy (%)': formatted(stat(policy, method, 'balanced_accuracy')),
                   'ROC-AUC': formatted(stat(policy, method, 'roc_auc'), 1),
                   'BCE': formatted(stat(policy, method, 'bce'), 1),
                   'Sex separation (pp)': formatted(stat(policy, method, 'SEX_separation')),
                   'Race n>=500 separation (pp)': formatted(stat(policy, method, 'RAC1P_n500_separation'))}))
        text += [f'### {policy}\n', markdown(pd.DataFrame(records)), '\n']
    private_losses = paired.loc[paired.metric.eq('accuracy_loss_pp') & paired.method.str.startswith('dp_')].copy()
    text += ['## Paired accuracy losses versus same-seed SGD (pp)\n',
             markdown(private_losses[['feature_policy', 'method', 'mean', 'std', 'valid_seeds']].round(5)), '\n',
             'Positive loss means DP was less accurate. Each difference is calculated within policy/seed first, then averaged.\n',
             '## Budget differences versus training variation\n']
    display = pd.read_csv(root/'aggregate/budget_accuracy_ordering.csv')
    text += [markdown(display.round(5)), '\n',
             'Differences above are later method minus earlier method in percentage points. '
             'Compare each mean with its paired SD and seed-level signs; neither monotonicity nor significance is assumed. '
             'All budget differences for fairness components remain in the full aggregate CSV. '
             'Small budget effects motivate extending prespecified important settings to ten seeds for final analysis.\n',
             '## Female versus male false-positive rates\n', markdown(persistence.round(5)), '\n',
             'Positive mean means female FPR is higher. Counts show how many valid seeds have this ordering. '
             'ESR-specific comparisons are in `aggregate/within_esr_sex_fpr_difference_mean_std.csv`; '
             'ESR1 is not an actual-negative category, so its FPR stays undefined. '
             'Small conditional denominators remain flagged in the seed-level ESR tables.\n',
             '## Audit findings and interpretation limits\n',
             '- Feature inclusion uses 16 input variables/87 encoded columns; exclusion uses 14/77. '
             'Race code4 is absent from reference data: its one-hot block uses the existing unknown-category rule, '
             'while all race labels/records remain in evaluation.\n'
             '- Household overlap exists under the fixed original person split. No split was changed; '
             'these are individual-record DP claims, not household-level guarantees.\n'
             '- Ordinary SGD uses actual-batch mean gradients. Both Poisson controls match DP expected-batch normalization. '
             'Compare shuffled SGD -> Poisson-unclipped -> Poisson-clipped -> DP to diagnose the sequence of changes. '
             'Clipped/unclipped controls share a deterministic sampling stream; DP sampling/noise are fresh.\n'
             '- Full nine-race and intersection metrics remain in detailed tables, with missing values preserved. '
             'The n>=500 headline race comparison is separately labeled and is not the full-race result.\n'
             '- Protected-feature inclusion versus exclusion uses the same initialization seed, but input dimensions differ: '
             'weights cannot be identical across those policies. Exact initialization hashes match within each policy/seed.\n'
             '- No interventions, target changes, final-test evaluation, or publication figures were performed. '
             'The repository includes non-private baselines/controls/audits; per-run DP estimates are not a whole-repository guarantee.\n',
             '## Tables and next review\n',
             'All seed-level, component/group, paired, feature-policy, ESR, and composition tables are under `aggregate/`. '
             '`plot_plan.json` specifies at most three candidate main figures; the reusable plotting command waits for explicit selection. '
             'Review whether clipping/sampling explains the main loss, whether budget changes exceed variability, '
             'whether female FPR persists within ESR categories, and which settings need ten seeds.\n',
             '## Resume/reproduce\n',
             'From the repository root: ` .\\.venv\\Scripts\\python.exe scripts\\run_controlled.py --stage main `, '
             'then `--stage controls`, then `--stage summarize`. Completed matching runs are skipped. '
             'See the root README for preparation, tests, and interrupted-lock handling.\n']
    # Numerical interpretations use the aggregate tables, never hard-coded pilot trends.
    extra = ['## Concrete training contrasts\n']
    for policy in config['feature_policies']:
        rates = {method: stat(policy, method, 'accuracy') for method in MAIN+CONTROLS}
        if all(rates[method] is not None for method in ['sgd', 'poisson_unclipped', 'poisson_clipped', 'dp_epsilon_1']):
            changes = [100*(rates[b]['mean']-rates[a]['mean']) for a, b in
                       [('sgd', 'poisson_unclipped'), ('poisson_unclipped', 'poisson_clipped'),
                        ('poisson_clipped', 'dp_epsilon_1')]]
            extra += [f'- {policy}: SGD -> Poisson-unclipped accuracy change {changes[0]:+.3f} pp; '
                      f'Poisson-unclipped -> clipped {changes[1]:+.3f} pp; '
                      f'clipped -> DP1 {changes[2]:+.3f} pp.\n']
        budget = budget_stats.loc[budget_stats.feature_policy.eq(policy) &
                 budget_stats.comparison.eq('dp_epsilon_8 minus dp_epsilon_1') & budget_stats.metric.eq('accuracy')]
        if len(budget):
            row = budget.iloc[0]
            extra += [f'- {policy}: epsilon8 minus epsilon1 accuracy is {100*row["mean"]:+.3f} pp '
                      f'(paired sample SD {100*row["std"]:.3f} pp; {int(row["valid_seeds"])} valid seeds). '
                      'This is a variability comparison, not a significance claim.\n']
    extra += ['The two non-private Poisson controls share sample streams; DP draws fresh samples/noise. '
              'Use these contrasts to distinguish sampling/normalization and clipping from additional noise, '
              'while retaining seed variability.\n', '## Effect of including sensitive predictors\n']
    feature_stats = pd.read_csv(root/'aggregate/feature_policy_differences_mean_std.csv')
    feature_display = feature_stats.loc[feature_stats.metric.isin(['accuracy', 'SEX_separation', 'RAC1P_n500_separation']) &
                                      feature_stats.method.isin(['sgd', 'dp_epsilon_1'])].copy()
    feature_display[['mean', 'std']] *= 100
    extra += [markdown(feature_display[['method', 'metric', 'mean', 'std', 'valid_seeds']].round(5)), '\n',
              'Differences are include_sensitive minus exclude_sensitive, in percentage points. '
              'Positive accuracy difference favors inclusion; positive gap difference indicates more disparity on that criterion. '
              'All budgets and controls are retained in the full feature-policy CSV.\n']
    esr_stats = pd.read_csv(root/'aggregate/esr_mean_std.csv', dtype={'group': str})
    distribution = pd.read_csv(root/'audit/development_esr_distributions.csv', dtype={'group': str})
    within = pd.read_csv(root/'aggregate/within_esr_sex_fpr_difference_seed.csv')
    primary = config['feature_policies'][0]
    chosen = esr_stats.loc[esr_stats.feature_policy.eq(primary) & esr_stats.method.eq('dp_epsilon_1') &
                           esr_stats.attribute.eq('SEX')]
    descriptions = {2: 'Civilian employed, temporarily absent', 3: 'Unemployed',
                    4: 'Armed forces, at work', 5: 'Armed forces, temporarily absent', 6: 'Not in labor force'}
    esr_rows, composition_rows = [], []
    if len(chosen):
        for code, description in descriptions.items():
            values = {}
            for group in ['1', '2']:
                group_stats = chosen.loc[chosen.group.eq(group) & chosen.ESR.eq(code)]
                count = distribution.loc[distribution.attribute.eq('SEX') & distribution.group.eq(group) & distribution.ESR.eq(code), 'n'].iloc[0]
                fpr = group_stats.loc[group_stats.metric.eq('fpr')].iloc[0]
                share = group_stats.loc[group_stats.metric.eq('fp_composition')].iloc[0]
                values[group] = (int(count), formatted(fpr), formatted(share))
            differences = within.loc[within.feature_policy.eq(primary) & within.method.eq('dp_epsilon_1') &
                                      within.ESR.eq(code), 'female_minus_male_fpr_pp'].dropna()
            persistence_label = f'{int((differences > 0).sum())}/{len(differences)}'
            esr_rows.append(dict(ESR=code, Meaning=description, Male_n=values['1'][0], Female_n=values['2'][0],
                                 Male_FPR=values['1'][1], Female_FPR=values['2'][1], Female_higher_seeds=persistence_label))
            composition_rows.append(dict(ESR=code, Male_FP_share=values['1'][2], Female_FP_share=values['2'][2]))
        extra += [f'## ESR diagnostic example: {primary}, DP epsilon1\n', markdown(pd.DataFrame(esr_rows)), '\n',
                  'FPR entries are percent mean +/- sample SD. Counts are development people in each actual-negative ESR category, '
                  'not five times that population. Categories with fewer than30 people have weak support even if the training-seed ordering is consistent. '
                  'ESR1 is actual positive and has no FPR denominator.\n',
                  'Share of each sex group\'s false positives contributed by each ESR category (percent):\n',
                  markdown(pd.DataFrame(composition_rows)), '\n',
                  'These tables distinguish within-category error rates from differences in the composition of the negative class. '
                  'They do not justify relabeling ESR2/4/5 without a separate target-definition decision.\n']
    text += extra
    text += ['## Recommended figures after review (at most three)\n',
             '1. Accuracy and sex/headline-race separation versus epsilon, with SGD and Poisson controls: '
             'show the small budget differences alongside the much larger clipping-related utility change.\n'
             '2. Paired accuracy losses by sex and supported race group under both feature policies: '
             'show the observed change in which groups bear larger costs.\n'
             '3. Sex-specific TPR/FPR across methods, with an ESR diagnostic inset if selected: '
             'show error levels, feature-policy dependence, and label-composition effects.\n',
             'These are recommendations only; no publication figures have been rendered. '
             'Review the five-seed findings, prespecify ten-seed follow-up settings, and decide whether an '
             'alternative target deserves a separate experiment before implementing interventions.\n']
    (root/'FINDINGS.md').write_text('\n'.join(text), encoding='utf-8')
