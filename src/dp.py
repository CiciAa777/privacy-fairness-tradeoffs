"""Opacus DP-SGD pilots. Public development evaluation; no raw training-loss logs."""
from pathlib import Path
import hashlib
import json
import random
import time
import numpy as np
import pandas as pd
from scipy import sparse
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
import opacus
from opacus import PrivacyEngine
from opacus.validators import ModuleValidator
from .baseline import ROOT, make_model
from .metrics import evaluate


def clean(value):
    if isinstance(value, dict): return {k: clean(v) for k,v in value.items()}
    if isinstance(value, list): return [clean(v) for v in value]
    if isinstance(value, float) and not np.isfinite(value): return None
    return value


def run_pilots(config_path=None, epsilons=None):
    config = json.loads(Path(config_path or ROOT/'configs/dp_pilot.json').read_text())
    targets = config['target_epsilons'] if epsilons is None else epsilons
    if not config['poisson_sampling']: raise ValueError('Pilot requires Poisson sampling')
    torch.set_num_threads(config['cpu_threads'])
    data = ROOT/'data/processed'
    manifest_text = (data/'manifest.json').read_text()
    fingerprint = hashlib.sha256(manifest_text.encode()).hexdigest()
    manifest = json.loads(manifest_text)
    baseline = json.loads((ROOT/'results/baseline/summary.json').read_text())
    assert baseline['manifest_sha256'] == fingerprint, 'Baseline has different data'
    for key in ['seed','hidden_sizes','epochs','batch_size','learning_rate','momentum','threshold']:
        assert config[key] == baseline['config'][key], f'Unmatched baseline setting: {key}'
    metadata, tensors = {}, {}
    for split in ['train','development']:
        metadata[split] = pd.read_csv(data/f'metadata_{split}.csv')
        x = sparse.load_npz(data/f'X_{split}.npz').toarray()
        assert x.shape == (len(metadata[split]),manifest['encoded_dimension'])
        tensors[split] = torch.from_numpy(x).float()
    assert set(metadata['train'].row_id).isdisjoint(metadata['development'].row_id)
    n = len(metadata['train'])
    delta = 1/(10*n)
    dataset = TensorDataset(tensors['train'], torch.tensor(metadata['train'].Y.to_numpy(),dtype=torch.float32))
    dev_y = torch.tensor(metadata['development'].Y.to_numpy(),dtype=torch.float32)
    base_groups = pd.read_csv(ROOT/'results/baseline/group_metrics.csv',dtype={'group':str})
    root_output = ROOT/config['output_dir']; root_output.mkdir(parents=True,exist_ok=True)
    for target in targets:
        if target <= 0: raise ValueError('Epsilon must be positive')
        output = root_output/f'epsilon_{target:g}'
        if (output/'summary.json').exists():
            raise FileExistsError(f'Completed run already exists: {output}; use another output directory for a new run')
        output.mkdir(exist_ok=True)
        random.seed(config['seed']); np.random.seed(config['seed']); torch.manual_seed(config['seed'])
        model = make_model(manifest['encoded_dimension'],config['hidden_sizes'])
        ModuleValidator.validate(model,strict=True)
        # Initialization is reproducible. Sampling/noise use separate, unexported entropy.
        sampling_rng = torch.Generator(); sampling_rng.seed()
        noise_rng = torch.Generator(); noise_rng.seed()
        loader = DataLoader(dataset,batch_size=config['batch_size'],shuffle=True,generator=sampling_rng)
        optimizer = torch.optim.SGD(model.parameters(),lr=config['learning_rate'],momentum=config['momentum'])
        engine = PrivacyEngine(accountant=config['accountant'],secure_mode=config['secure_mode'])
        model,optimizer,loader = engine.make_private_with_epsilon(
            module=model,optimizer=optimizer,data_loader=loader,target_epsilon=float(target),
            target_delta=delta,epochs=config['epochs'],max_grad_norm=config['max_grad_norm'],
            poisson_sampling=True,noise_generator=noise_rng,epsilon_tolerance=.001)
        criterion = nn.BCEWithLogitsLoss()
        history=[]; steps=0; start=time.perf_counter()
        print(f'epsilon target={target:g}; delta={delta:.8g}; noise multiplier={optimizer.noise_multiplier:.6f}',flush=True)
        for epoch in range(1,config['epochs']+1):
            model.train()
            for xb,yb in loader:
                optimizer.zero_grad()
                logits=model(xb).squeeze(1)
                loss=criterion(logits,yb) if len(yb) else logits.sum()
                if not torch.isfinite(loss): raise ValueError('Nonfinite private-training loss')
                loss.backward(); optimizer.step(); steps+=1
            model.eval()
            with torch.no_grad():
                logits=model(tensors['development']).squeeze(1)
                scores=logits.sigmoid().numpy()
                dev_loss=criterion(logits,dev_y).item()
            accuracy=float(((scores>=config['threshold'])==dev_y.numpy()).mean())
            epsilon=engine.get_epsilon(delta)
            history.append(dict(epoch=epoch,steps=steps,epsilon=epsilon,
                                development_loss=dev_loss,development_accuracy=accuracy))
            pd.DataFrame(history).to_csv(output/'history.csv',index=False)
            print(f'  epoch {epoch:02d}; epsilon={epsilon:.4f}; dev accuracy={accuracy:.4f}',flush=True)
        epsilon=engine.get_epsilon(delta)
        assert epsilon <= target + .001, 'Achieved epsilon exceeds requested budget'
        accounted_steps=sum(entry[2] for entry in engine.accountant.history)
        assert accounted_steps == steps == config['epochs']*len(loader)
        overall,groups,gaps=evaluate(metadata['development'],scores,config['threshold'])
        groups.to_csv(output/'group_metrics.csv',index=False)
        comparison=groups.merge(base_groups[['attribute','group','accuracy']],on=['attribute','group'],
                                suffixes=('_dp','_sgd'),validate='one_to_one')
        comparison['accuracy_loss_pp']=100*(comparison.accuracy_sgd-comparison.accuracy_dp)
        comparison.to_csv(output/'group_accuracy_losses.csv',index=False)
        predictions=metadata['development'].copy();predictions['score']=scores
        predictions['prediction']=(scores>=config['threshold']).astype(int)
        predictions.to_csv(output/'development_predictions.csv',index=False)
        summary=dict(scope='single-run development pilot; not final test results',target_epsilon=target,
                     epsilon=epsilon,delta=delta,accountant=config['accountant'],noise_multiplier=optimizer.noise_multiplier,
                     max_grad_norm=config['max_grad_norm'],sampling='Poisson',sample_rate=loader.sample_rate,
                     expected_batch_size=optimizer.expected_batch_size,steps=steps,accounted_steps=accounted_steps,
                     epochs=config['epochs'],secure_mode=config['secure_mode'],
                     randomness='Fixed model initialization seed; independent sampling/noise seeds are not exported. Research RNG, not cryptographically hardened.',
                     privacy_scope='Model-training records only, conditional on fixed public auxiliary partitions and preprocessing. Per-run accountant estimate; runs compose.',
                     training_statistics='No raw training losses/gradient statistics exported',
                     overall=overall,gaps=gaps,seconds=time.perf_counter()-start,
                     input_dimension=manifest['encoded_dimension'],manifest_sha256=fingerprint,
                     torch_version=torch.__version__,opacus_version=opacus.__version__,config=config,
                     selection='Fixed final epoch; no private-data early stopping',device='cpu')
        (output/'summary.json').write_text(json.dumps(clean(summary),indent=2,allow_nan=False))
        torch.save(dict(state_dict=model._module.state_dict(),config=config,input_dimension=manifest['encoded_dimension'],
                        epsilon=epsilon,delta=delta,manifest_sha256=fingerprint,
                        accountant_state=engine.accountant.state_dict()),output/'model.pt')
        print(f'Finished epsilon {target:g} in {summary["seconds"]:.1f}s',flush=True)
    return summarize(root_output,baseline)


def summarize(output,baseline):
    rows=[dict(method='SGD',target_epsilon=None,epsilon=None,accuracy=baseline['overall']['accuracy'],
               balanced_accuracy=baseline['overall']['balanced_accuracy'],accuracy_loss_pp=0.)]
    summaries=[]
    for path in sorted(output.glob('epsilon_*/summary.json')):
        s=json.loads(path.read_text());summaries.append(s)
        row=dict(method='DP-SGD',target_epsilon=s['target_epsilon'],epsilon=s['epsilon'],delta=s['delta'],
                 noise_multiplier=s['noise_multiplier'],accuracy=s['overall']['accuracy'],
                 balanced_accuracy=s['overall']['balanced_accuracy'],
                 accuracy_loss_pp=100*(baseline['overall']['accuracy']-s['overall']['accuracy']))
        rows.append(row)
    for row,summary in zip(rows,[baseline]+summaries):
        for attribute in ['SEX','RAC1P']:
            for metric in ['accuracy_gap','independence','separation','sufficiency']:
                row[f'{attribute}_{metric}']=summary['gaps'][attribute][metric]
    frame=pd.DataFrame(rows);frame.to_csv(output/'comparison.csv',index=False)
    # Basic composition is a valid conservative bound for releasing all pilot models together.
    (output/'composition.json').write_text(json.dumps(dict(
        completed_runs=len(summaries),basic_composition_epsilon=sum(s['epsilon'] for s in summaries),
        basic_composition_delta=sum(s['delta'] for s in summaries),
        note='Per-run epsilons do not describe the entire sweep. Non-private baseline/public-data audit is not a private release of the training partition.'),indent=2))
    return frame
