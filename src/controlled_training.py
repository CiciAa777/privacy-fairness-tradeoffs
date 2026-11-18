"""Matched-seed baseline, DP-SGD, and non-private Poisson diagnostic controls."""
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import random
import time
import traceback

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
import opacus
from opacus import PrivacyEngine
from opacus.data_loader import DPDataLoader
from opacus.validators import ModuleValidator
from sklearn.metrics import roc_auc_score

from .baseline import ROOT, make_model
from .controlled_data import digest, json_write, load_policy, sha
from .metrics import evaluate

MAIN = ['sgd', 'dp_epsilon_1', 'dp_epsilon_4', 'dp_epsilon_8']
CONTROLS = ['poisson_unclipped', 'poisson_clipped']


def clean(value):
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(v) for v in value]
    if isinstance(value, np.generic):
        return clean(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def gaps_from_frame(frame):
    def gap(column):
        rates = frame[column]
        return float(rates.max()-rates.min()) if len(rates) and rates.notna().all() else None
    def both(a, b):
        return max(a, b) if a is not None and b is not None else None
    return dict(accuracy_gap=gap('accuracy'), independence=gap('selection_rate'),
                tpr_gap=gap('tpr'), fpr_gap=gap('fpr'),
                separation=both(gap('tpr'), gap('fpr')),
                ppv_gap=gap('ppv'), npv_gap=gap('npv'),
                sufficiency=both(gap('ppv'), gap('npv')))


def esr_diagnostics(metadata, prediction):
    rows = []
    for attribute, values in [('SEX', [1, 2]), ('RAC1P', list(range(1, 10)))]:
        for value in values:
            group = metadata[attribute].eq(value).to_numpy()
            all_fp = int((group & metadata.Y.eq(0).to_numpy() & (prediction == 1)).sum())
            for esr in range(1, 7):
                mask = group & metadata.ESR.eq(esr).to_numpy()
                negatives = mask & metadata.Y.eq(0).to_numpy()
                denominator = int(negatives.sum())
                fp = int((negatives & (prediction == 1)).sum())
                rows.append(dict(attribute=attribute, group=str(value), ESR=esr,
                                 n=int(mask.sum()), actual_negatives=denominator,
                                 false_positives=fp, fpr=fp/denominator if denominator else None,
                                 fp_composition=fp/all_fp if all_fp else None,
                                 low_denominator=denominator < 30))
    return pd.DataFrame(rows)


def state_hash(model):
    result = hashlib.sha256()
    for name, tensor in model.state_dict().items():
        result.update(name.encode())
        result.update(tensor.detach().cpu().numpy().tobytes())
    return result.hexdigest()


def run_one(config, policy, method, seed, data, smoke=False):
    train_x, train_meta, dev_x, dev_meta, manifest = data
    assert set(train_meta.row_id).isdisjoint(dev_meta.row_id)
    private = method.startswith('dp_epsilon_')
    target = float(method.removeprefix('dp_epsilon_')) if private else None
    n = len(train_meta)
    steps_per_epoch = math.ceil(n/config['batch_size'])
    q = 1/steps_per_epoch
    expected_batch = int(n*q)
    delta = 1/(10*n)
    protocol = {key: config[key] for key in ['hidden_sizes', 'epochs', 'batch_size',
                'learning_rate', 'momentum', 'threshold', 'max_grad_norm', 'accountant',
                'delta_rule', 'secure_mode', 'cpu_threads']}
    code_fingerprints = {name: sha(ROOT/'src'/name) for name in
                         ['controlled_data.py', 'controlled_training.py', 'metrics.py', 'baseline.py']}
    signature = digest(dict(protocol=protocol, policy=policy, method=method, seed=seed,
                            feature_signature=manifest['feature_signature'], n_train=n,
                            n_development=len(dev_meta), smoke=smoke, code=code_fingerprints))
    root = ROOT/config['output_dir']
    output = root/('smoke' if smoke else 'runs')/policy/method/f'seed_{seed}'
    output.mkdir(parents=True, exist_ok=True)
    summary_path = output/'summary.json'
    if summary_path.exists():
        previous = json.loads(summary_path.read_text(encoding='utf-8'))
        if previous['run_signature'] != signature:
            raise ValueError(f'Completed run has a different protocol/code: {output}')
        print(f'SKIP completed {policy}/{method}/seed_{seed}', flush=True)
        return previous
    marker = output/'.running'
    with marker.open('x', encoding='utf-8') as handle:
        json.dump(dict(pid=os.getpid(), started=time.time()), handle)
    try:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.set_num_threads(config['cpu_threads'])
        model = make_model(train_x.shape[1], config['hidden_sizes'])
        ModuleValidator.validate(model, strict=True)
        initial_hash = state_hash(model)
        optimizer = torch.optim.SGD(model.parameters(), lr=config['learning_rate'], momentum=config['momentum'])
        sampling = torch.Generator()
        if private:
            sampling.seed()
        else:
            sampling.manual_seed(seed if method == 'sgd' else seed+100000)
        loader = DataLoader(TensorDataset(torch.from_numpy(train_x).float(),
                            torch.tensor(train_meta.Y.to_numpy(), dtype=torch.float32)),
                            batch_size=config['batch_size'], shuffle=True, generator=sampling)
        engine = None
        if private or method == 'poisson_clipped':
            engine = PrivacyEngine(accountant=config['accountant'], secure_mode=config['secure_mode'])
            if private:
                noise_rng = torch.Generator()
                noise_rng.seed()
                model, optimizer, loader = engine.make_private_with_epsilon(
                    module=model, optimizer=optimizer, data_loader=loader, target_epsilon=target,
                    target_delta=delta, epochs=config['epochs'], max_grad_norm=config['max_grad_norm'],
                    poisson_sampling=True, noise_generator=noise_rng, epsilon_tolerance=.001)
            else:
                model, optimizer, loader = engine.make_private(module=model, optimizer=optimizer,
                    data_loader=loader, noise_multiplier=0, max_grad_norm=config['max_grad_norm'],
                    poisson_sampling=True)
            assert loader.sample_rate == q and optimizer.expected_batch_size == expected_batch
        elif method == 'poisson_unclipped':
            loader = DPDataLoader.from_data_loader(loader, generator=sampling)
            assert loader.sample_rate == q
        elif method != 'sgd':
            raise ValueError(f'Unknown method {method}')
        assert len(loader) == steps_per_epoch
        dev_tensor = torch.from_numpy(dev_x).float()
        dev_y = torch.tensor(dev_meta.Y.to_numpy(), dtype=torch.float32)
        criterion = nn.BCEWithLogitsLoss()
        history = []
        steps = 0
        started = time.perf_counter()
        print(f'START {policy}/{method}/seed_{seed}: {train_x.shape[1]} inputs, {n} training rows', flush=True)
        for epoch in range(1, config['epochs']+1):
            model.train()
            for xb, yb in loader:
                optimizer.zero_grad()
                logits = model(xb).squeeze(1)
                if method == 'poisson_unclipped':
                    loss = nn.functional.binary_cross_entropy_with_logits(logits, yb, reduction='sum')/expected_batch
                else:
                    loss = criterion(logits, yb) if len(yb) else logits.sum()
                if not torch.isfinite(loss):
                    raise ValueError('Nonfinite training objective')
                loss.backward()
                if steps == 0 and engine is not None:
                    assert all(hasattr(parameter, 'grad_sample') for parameter in model.parameters())
                optimizer.step()
                steps += 1
            model.eval()
            with torch.no_grad():
                logits = model(dev_tensor).squeeze(1)
                scores = logits.sigmoid().numpy()
                bce = criterion(logits, dev_y).item()
            epsilon = engine.get_epsilon(delta) if private else None
            history.append(dict(epoch=epoch, steps=steps, epsilon=epsilon, development_bce=bce,
                                development_accuracy=float(((scores >= config['threshold']) == dev_meta.Y).mean())))
            pd.DataFrame(history).to_csv(output/'history.csv', index=False)
            if epoch == 1 or epoch % 5 == 0 or epoch == config['epochs']:
                print(f'  epoch {epoch}/{config["epochs"]}: accuracy={history[-1]["development_accuracy"]:.5f}', flush=True)
        assert steps == config['epochs']*steps_per_epoch
        accounted = sum(entry[2] for entry in engine.accountant.history) if engine is not None else 0
        if engine is not None:
            assert accounted == steps
        if private:
            assert epsilon <= target+.001 and abs(epsilon-target) <= .001
        overall, groups, gaps = evaluate(dev_meta, scores, config['threshold'])
        overall.update(bce=bce, roc_auc=float(roc_auc_score(dev_meta.Y, scores))
                       if dev_meta.Y.nunique() == 2 else None)
        groups['actual_positives'] = groups.tp+groups.fn
        groups['actual_negatives'] = groups.tn+groups.fp
        groups['predicted_positives'] = groups.tp+groups.fp
        groups['predicted_negatives'] = groups.tn+groups.fn
        groups['low_denominator'] = groups.min_conditional_count < config['minimum_conditional_n']
        headline = groups.loc[groups.attribute.eq('RAC1P') & groups.n.ge(config['minimum_headline_race_n'])]
        headline_gaps = gaps_from_frame(headline)
        groups.to_csv(output/'group_metrics.csv', index=False)
        prediction = (scores >= config['threshold']).astype(int)
        predictions = dev_meta.copy()
        predictions['score'], predictions['prediction'] = scores, prediction
        predictions.to_csv(output/'development_predictions.csv', index=False)
        esr_diagnostics(dev_meta, prediction).to_csv(output/'esr_diagnostics.csv', index=False)
        if method != 'sgd':
            baseline_folder = output.parent.parent/'sgd'/f'seed_{seed}'
            baseline = json.loads((baseline_folder/'summary.json').read_text(encoding='utf-8'))
            assert baseline['initialization_sha256'] == initial_hash
            assert baseline['feature_signature'] == manifest['feature_signature']
            base_groups = pd.read_csv(baseline_folder/'group_metrics.csv', dtype={'group': str})
            losses = groups.merge(base_groups[['attribute', 'group', 'accuracy']], on=['attribute', 'group'],
                                  suffixes=('_method', '_sgd'), validate='one_to_one')
            losses['accuracy_loss_pp'] = 100*(losses.accuracy_sgd-losses.accuracy_method)
            losses.to_csv(output/'group_accuracy_losses.csv', index=False)
        checkpoint_model = model._module if engine is not None else model
        torch.save(dict(state_dict=checkpoint_model.state_dict(), initialization_sha256=initial_hash,
                        model_seed=seed, feature_signature=manifest['feature_signature'],
                        accountant_state=engine.accountant.state_dict() if private else None), output/'model.pt')
        summary = clean(dict(
            completed=True, feature_policy=policy, method=method, model_seed=seed, is_smoke=smoke,
            training_population='public reference smoke subset' if smoke else 'fixed original training partition',
            run_signature=signature, code_sha256=code_fingerprints, protocol=protocol,
            initialization_sha256=initial_hash, data_signature=manifest['data_signature'],
            feature_signature=manifest['feature_signature'], input_dimension=train_x.shape[1],
            n_train=n, n_development=len(dev_meta), target_epsilon=target, epsilon=epsilon if private else None,
            delta=delta if private else None, noise_multiplier=optimizer.noise_multiplier if private else 0.,
            clipping_norm=config['max_grad_norm'] if private or method == 'poisson_clipped' else None,
            private_training_mechanism=private, secure_mode=config['secure_mode'] if private else None,
            privacy_scope='Per-run individual training records conditional on fixed public auxiliary data; releases compose',
            non_private_control=not private, epochs=config['epochs'], steps=steps, accounted_steps=accounted,
            sampling='shuffled without replacement' if method == 'sgd' else 'Poisson',
            sample_rate=q if method != 'sgd' else None,
            expected_batch_normalizer=expected_batch if method != 'sgd' else None,
            gradient_normalization='actual-batch mean' if method == 'sgd' else 'clipped/unclipped gradient sum divided by expected batch normalizer',
            private_sampling_noise_rng_exported=False, nonprivate_sampler_seed=seed if method == 'sgd' else (seed+100000 if not private else None),
            selection='fixed final epoch', test_evaluated=False, overall=overall, gaps=gaps,
            headline_race_groups=headline.group.tolist(), headline_race_min_n=config['minimum_headline_race_n'],
            headline_race_gaps=headline_gaps, seconds=time.perf_counter()-started,
            environment=dict(python=platform.python_version(), torch=torch.__version__, opacus=opacus.__version__)))
        json_write(output/'summary.tmp.json', summary)
        (output/'summary.tmp.json').replace(summary_path)
        print(f'DONE {policy}/{method}/seed_{seed}: {summary["seconds"]:.1f}s', flush=True)
        return summary
    except Exception as error:
        json_write(output/'failure.json', dict(error=str(error), traceback=traceback.format_exc()))
        raise
    finally:
        marker.unlink(missing_ok=True)


def run_grid(config, stage='main', policies=None, seeds=None):
    smoke = stage == 'smoke'
    settings = dict(config)
    if smoke:
        settings.update(epochs=1, batch_size=64)
    if stage == 'controls':
        methods = CONTROLS
    else:
        methods = ['sgd']+[f'dp_epsilon_{epsilon:g}' for epsilon in config['target_epsilons']]
        if smoke:
            methods += CONTROLS
    for policy in policies or config['feature_policies']:
        data = load_policy(config, policy, smoke=smoke)
        for seed in ([42] if smoke else seeds or config['seeds']):
            for method in methods:
                run_one(settings, policy, method, seed, data, smoke=smoke)
