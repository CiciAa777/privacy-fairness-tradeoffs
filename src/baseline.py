"""Ordinary SGD MLP baseline. Only training/development records are loaded."""
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
from torch.utils.data import TensorDataset, DataLoader
from .metrics import evaluate

ROOT = Path(__file__).resolve().parents[1]


def train(config_path=None):
    config_path = Path(config_path) if config_path else ROOT/'configs/baseline.json'
    config = json.loads(config_path.read_text())
    seed = config['seed']
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.set_num_threads(config['cpu_threads'])
    data = ROOT/'data/processed'
    manifest_text = (data/'manifest.json').read_text()
    manifest = json.loads(manifest_text)
    assert 'ESP' not in manifest['model_inputs'], 'Reprepare data after removing ESP'
    tensors, metadata = {}, {}
    for split in ['train', 'development']:
        x = sparse.load_npz(data/f'X_{split}.npz').toarray()
        metadata[split] = pd.read_csv(data/f'metadata_{split}.csv')
        assert len(x) == len(metadata[split]) and x.shape[1] == manifest['encoded_dimension']
        tensors[split] = torch.from_numpy(x).float()
    assert set(metadata['train'].row_id).isdisjoint(metadata['development'].row_id)
    y = torch.tensor(metadata['train'].Y.to_numpy(), dtype=torch.float32)
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(TensorDataset(tensors['train'], y), batch_size=config['batch_size'],
                        shuffle=True, generator=generator)
    layers = []
    width = manifest['encoded_dimension']
    for hidden in config['hidden_sizes']:
        layers.extend([nn.Linear(width, hidden), nn.ReLU()]); width=hidden
    layers.append(nn.Linear(width,1))
    model = nn.Sequential(*layers)
    optimizer = torch.optim.SGD(model.parameters(), lr=config['learning_rate'], momentum=config['momentum'])
    loss_fn = nn.BCEWithLogitsLoss()
    history=[]; start=time.perf_counter()
    for epoch in range(1, config['epochs']+1):
        model.train(); total=0.
        for xb,yb in loader:
            optimizer.zero_grad()
            loss = loss_fn(model(xb).squeeze(1), yb)
            if not torch.isfinite(loss): raise ValueError('Nonfinite training loss')
            loss.backward(); optimizer.step(); total += loss.item()*len(yb)
        model.eval()
        with torch.no_grad():
            logits=model(tensors['development']).squeeze(1)
            scores=logits.sigmoid().numpy()
            dev_loss=loss_fn(logits, torch.tensor(metadata['development'].Y.to_numpy(),dtype=torch.float32)).item()
        accuracy=float(((scores >= config['threshold'])==metadata['development'].Y.to_numpy()).mean())
        history.append(dict(epoch=epoch,train_loss=total/len(y),development_loss=dev_loss,development_accuracy=accuracy))
        print(f'Epoch {epoch:02d}: train loss={history[-1]["train_loss"]:.4f}; dev accuracy={accuracy:.4f}',flush=True)
    output=ROOT/config['output_dir'];output.mkdir(parents=True,exist_ok=True)
    overall,groups,gaps=evaluate(metadata['development'], scores, config['threshold'])
    majority_label=int(metadata['train'].Y.mean() >= .5)
    majority,majority_groups,majority_gaps=evaluate(metadata['development'],np.full(len(scores),majority_label),config['threshold'])
    pd.DataFrame(history).to_csv(output/'history.csv',index=False)
    groups.to_csv(output/'group_metrics.csv',index=False)
    majority_groups.to_csv(output/'majority_group_metrics.csv',index=False)
    predictions=metadata['development'].copy();predictions['score']=scores
    predictions['prediction']=(scores>=config['threshold']).astype(int)
    predictions.to_csv(output/'development_predictions.csv',index=False)
    summary=dict(scope='single-seed non-private development baseline; not final test results',
                 overall=overall,gaps=gaps,majority_label=majority_label,majority=majority,
                 majority_gaps=majority_gaps,seconds=time.perf_counter()-start,
                 input_dimension=manifest['encoded_dimension'],torch_version=torch.__version__,
                 manifest_sha256=hashlib.sha256(manifest_text.encode()).hexdigest(),config=config,
                 selection='fixed last epoch; no best-checkpoint selection', device='cpu')
    def clean(value):
        if isinstance(value,dict): return {k:clean(v) for k,v in value.items()}
        if isinstance(value,float) and not np.isfinite(value): return None
        return value
    (output/'summary.json').write_text(json.dumps(clean(summary),indent=2,allow_nan=False))
    (output/'config.json').write_text(json.dumps(config,indent=2))
    torch.save(dict(state_dict=model.state_dict(),config=config,input_dimension=manifest['encoded_dimension'],
                    manifest_sha256=summary['manifest_sha256']),output/'model.pt')
    print(f'Saved baseline artifacts to {output}',flush=True)
    return summary
