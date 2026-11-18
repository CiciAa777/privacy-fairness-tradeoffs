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
    seeds = pd.DataFrame(rows)
    if seeds.duplicated(['feature_policy', 'method', 'model_seed']).any():
        raise ValueError('Duplicate completed run')
    seeds.to_csv(out/'seed_metrics.csv', index=False)
    group_frame = pd.concat(groups, ignore_index=True)
    esr_frame = pd.concat(esr, ignore_index=True)
    group_frame.to_csv(out/'group_seed_metrics.csv', index=False)
    esr_frame.to_csv(out/'esr_seed_metrics.csv', index=False)
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
    status = dict(expected_main_runs=40, expected_control_runs=20, completed_runs=len(summaries),
        completed_main_runs=sum(s['method'] in MAIN for s in summaries),
        completed_control_runs=sum(s['method'] in CONTROLS for s in summaries),
        complete=not missing, missing=[dict(feature_policy=p, method=m, model_seed=s) for p, m, s in missing],
        seed_variation='Training/initialization/sampling/noise variation on a fixed split, not population uncertainty',
        final_test_evaluated=False)
    json_write(out/'completion.json', status)
    json_write(out/'privacy_composition.json', dict(
        private_main_runs=len(private), basic_composition_epsilon=sum(s['epsilon'] for s in private),
        basic_composition_delta=sum(s['delta'] for s in private),
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
    print(f'Aggregated {len(summaries)} completed runs; missing={len(missing)}. No figures rendered.', flush=True)
    return status


def write_findings(config, root, seeds, stats, paired, budget_stats, persistence, status):
    def stat(policy, method, metric):
        row = stats.loc[stats.feature_policy.eq(policy) & stats.method.eq(method) & stats.metric.eq(metric)]
        return row.iloc[0] if len(row) else None
    def formatted(row, scale=100):
        if row is None or pd.isna(row['mean']):
            return 'undefined'
        sd = 'undefined' if pd.isna(row['std']) else f'{scale*row["std"]:.3f}'
        return f'{scale*row["mean"]:.3f} +/- {sd} (n={int(row["valid_seeds"])})'
    text = ['# Controlled v2: baseline findings\n',
            f'Completed main runs: **{status["completed_main_runs"]}/40**; controls: **{status["completed_control_runs"]}/20**. '
            'Development data only. Means +/- sample SD across training seeds; no population-sampling uncertainty is estimated.\n',
            '## Overall accuracy and selected fairness outcomes\n']
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
    display = budget_stats.loc[budget_stats.metric.isin(['accuracy', 'SEX_separation', 'RAC1P_n500_separation'])].copy()
    display[['mean', 'std']] *= 100
    text += [markdown(display[['feature_policy', 'comparison', 'metric', 'mean', 'std', 'valid_seeds']].round(5)), '\n',
             'Differences above are later method minus earlier method in percentage points. '
             'Compare each mean with its paired SD and seed-level signs; neither monotonicity nor significance is assumed. '
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
    (root/'FINDINGS.md').write_text('\n'.join(text), encoding='utf-8')
