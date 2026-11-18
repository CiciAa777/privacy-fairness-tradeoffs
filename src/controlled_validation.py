"""Validate the completed study from counts, checkpoints, and public dev scores."""
import json
import math
import subprocess

import numpy as np
import pandas as pd
import torch
from opacus.accountants import RDPAccountant
from sklearn.metrics import roc_auc_score

from .baseline import ROOT, make_model
from .controlled_data import json_write
from .controlled_training import MAIN, CONTROLS


def validate_controlled(config):
    root = ROOT/config['output_dir']
    expected = [(policy, method, seed) for policy in config['feature_policies']
                for method in MAIN+CONTROLS for seed in config['seeds']]
    records = {}
    for policy, method, seed in expected:
        path = root/'runs'/policy/method/f'seed_{seed}'
        if not (path/'summary.json').exists():
            raise ValueError(f'Incomplete study: missing {path}')
        summary = json.loads((path/'summary.json').read_text(encoding='utf-8'))
        groups = pd.read_csv(path/'group_metrics.csv', dtype={'group': str})
        records[(policy, method, seed)] = (path, summary, groups)
    torch.set_num_threads(config['cpu_threads'])
    accountant_checks = model_checks = 0
    checkpoints_missing = []
    for (policy, method, seed), (path, summary, groups) in records.items():
        assert summary['completed'] and not summary['is_smoke'] and not summary['test_evaluated']
        assert (summary['feature_policy'], summary['method'], summary['model_seed']) == (policy, method, seed)
        assert summary['n_train'] == 151320 and summary['n_development'] == 30264
        assert summary['steps'] == summary['epochs']*296 == 5920
        assert summary['protocol']['threshold'] == .5
        assert not groups.duplicated(['attribute', 'group']).any() and len(groups) == 29
        for attribute in ['SEX', 'RAC1P', 'RAC1P_SEX']:
            group = groups.loc[groups.attribute.eq(attribute)]
            assert group.n.sum() == summary['n_development']
            for count in ['tp', 'tn', 'fp', 'fn']:
                assert group[count].sum() == summary['overall'][count]
        assert (groups.tp+groups.tn+groups.fp+groups.fn == groups.n).all()
        for key, numerator, denominator in [
            ('accuracy', groups.tp+groups.tn, groups.n),
            ('selection_rate', groups.tp+groups.fp, groups.n),
            ('tpr', groups.tp, groups.tp+groups.fn), ('fpr', groups.fp, groups.fp+groups.tn),
            ('ppv', groups.tp, groups.tp+groups.fp), ('npv', groups.tn, groups.tn+groups.fn)]:
            expected_rate = np.divide(numerator.to_numpy(dtype=float), denominator.to_numpy(dtype=float),
                                     out=np.full(len(groups), np.nan), where=denominator.to_numpy() > 0)
            np.testing.assert_allclose(groups[key], expected_rate, atol=1e-12, rtol=0, equal_nan=True)
        base_path, base, base_groups = records[(policy, 'sgd', seed)]
        assert base['initialization_sha256'] == summary['initialization_sha256']
        assert base['data_signature'] == summary['data_signature']
        assert base['feature_signature'] == summary['feature_signature']
        history = pd.read_csv(path/'history.csv')
        assert history.epoch.tolist() == list(range(1, 21))
        assert history.steps.tolist() == [epoch*296 for epoch in range(1, 21)]
        assert math.isclose(history.development_accuracy.iloc[-1], summary['overall']['accuracy'], abs_tol=1e-12)
        checkpoint_path = path/'model.pt'
        checkpoint = torch.load(checkpoint_path, weights_only=True, map_location='cpu') if checkpoint_path.exists() else None
        if checkpoint is None:
            checkpoints_missing.append(str(checkpoint_path.relative_to(ROOT)))
        else:
            assert set(checkpoint) == {'state_dict', 'initialization_sha256', 'model_seed', 'feature_signature', 'accountant_state'}
            assert checkpoint['initialization_sha256'] == summary['initialization_sha256']
        if summary['private_training_mechanism']:
            assert summary['sampling'] == 'Poisson' and summary['clipping_norm'] == 1
            assert summary['accounted_steps'] == 5920 and summary['expected_batch_normalizer'] == 511
            assert math.isclose(summary['delta'], 1/(10*151320))
            assert math.isclose(summary['sample_rate'], 1/296)
            accountant = RDPAccountant()
            accountant.history = [(summary['noise_multiplier'], summary['sample_rate'], summary['steps'])]
            epsilon = accountant.get_epsilon(summary['delta'])
            assert math.isclose(epsilon, summary['epsilon'], abs_tol=1e-12)
            assert abs(epsilon-summary['target_epsilon']) <= .001
            assert history.epsilon.is_monotonic_increasing
            if checkpoint is not None:
                accountant.load_state_dict(checkpoint['accountant_state'])
                assert sum(item[2] for item in accountant.history) == 5920
                assert math.isclose(accountant.get_epsilon(summary['delta']), epsilon, abs_tol=1e-12)
            accountant_checks += 1
        else:
            assert summary['epsilon'] is None and summary['delta'] is None
            if checkpoint is not None:
                assert checkpoint['accountant_state'] is None
        if method != 'sgd':
            actual = pd.read_csv(path/'group_accuracy_losses.csv', dtype={'group': str})
            pair = actual.merge(base_groups[['attribute', 'group', 'accuracy']], on=['attribute', 'group'], validate='one_to_one')
            np.testing.assert_allclose(pair.accuracy_loss_pp,
                100*(pair.accuracy-pair.accuracy_method), equal_nan=True, atol=1e-12)
        esr = pd.read_csv(path/'esr_diagnostics.csv', dtype={'group': str})
        for (attribute, code), rows in esr.groupby(['attribute', 'group']):
            source = groups.loc[groups.attribute.eq(attribute) & groups.group.eq(code)].iloc[0]
            assert rows.n.sum() == source.n
            assert rows.false_positives.sum() == source.fp
            assert rows.actual_negatives.sum() == source.tn+source.fp
            if source.fp > 0:
                assert math.isclose(rows.fp_composition.sum(), 1., abs_tol=1e-12)
        # Reconstruct every method for one representative initialization in each
        # policy; inspect no test records and never retrain.
        if seed == config['seeds'][0] and checkpoint is not None:
            model = make_model(summary['input_dimension'], config['hidden_sizes'])
            model.load_state_dict(checkpoint['state_dict'])
            model.eval()
            dev_x = np.load(root/'cache'/policy/'X_development.npy')
            with torch.no_grad():
                logits = model(torch.from_numpy(dev_x).float()).squeeze(1)
                scores = logits.sigmoid().numpy()
            saved = pd.read_csv(path/'development_predictions.csv')
            np.testing.assert_allclose(scores, saved.score, rtol=1e-6, atol=1e-7)
            accuracy = ((scores >= .5).astype(int) == saved.Y).mean()
            assert math.isclose(accuracy, summary['overall']['accuracy'], abs_tol=1e-12)
            assert math.isclose(roc_auc_score(saved.Y, scores), summary['overall']['roc_auc'], abs_tol=1e-12)
            bce = torch.nn.functional.binary_cross_entropy_with_logits(
                logits, torch.tensor(saved.Y.to_numpy(), dtype=torch.float32)).item()
            assert math.isclose(bce, summary['overall']['bce'], abs_tol=1e-7)
            model_checks += 1
    original = json.loads((ROOT/'V0/results/baseline/summary.json').read_text(encoding='utf-8'))
    assert records[('exclude_sensitive', 'sgd', 285)][1]['overall']['accuracy'] == original['overall']['accuracy']
    for filename in ['main.tex', 'progress_report.pdf']:
        source = subprocess.check_output(['git', 'show', f'12e6822:reports/progress_report/{filename}'], cwd=ROOT)
        archived = ROOT/'V0/reports/progress_report'/filename
        if filename.endswith('.pdf'):
            assert source == archived.read_bytes()
        else:
            assert source.decode('utf-8').replace('\r\n', '\n') == archived.read_text(encoding='utf-8')
    assert not (ROOT/'figures/controlled_v2').exists(), 'Publication figures were generated before review'
    result = dict(completed_full_runs=len(records), private_accountant_checks=accountant_checks,
                  representative_checkpoint_prediction_checks=model_checks, checkpoints_missing=checkpoints_missing,
                  matched_initialization_groups=len(config['feature_policies'])*len(config['seeds']),
                  group_counts_rates_and_esr_checked=True, seed285_v0_baseline_reproduced_exactly=True,
                  archived_progress_report_unchanged=True, publication_figures_generated=False,
                  test_outcomes_used=False, private_rng_state_exported=False)
    json_write(root/'audit/validation.json', result)
    print(json.dumps(result, indent=2), flush=True)
    return result
