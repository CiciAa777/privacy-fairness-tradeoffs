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
    v0 = json.loads((ROOT/'configs/historical_pilot_reference.json').read_text(encoding='utf-8'))
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
    print(f'Aggregated {len(summaries)} completed runs; missing={len(missing)}. No figures rendered.', flush=True)
    return status


GROUP_RATES = ['accuracy', 'balanced_accuracy', 'selection_rate', 'tpr', 'fpr', 'ppv', 'npv']

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
