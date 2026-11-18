"""Build the DP pilot log and explanatory notebook from completed runs only."""
from pathlib import Path
import json
import os
import nbformat as nbf
from nbclient import NotebookClient
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
RUN=ROOT/'results/dp_pilot'
comparison=pd.read_csv(RUN/'comparison.csv')
summaries=[json.loads(p.read_text()) for p in sorted(RUN.glob('epsilon_*/summary.json'))]
config=json.loads((ROOT/'configs/dp_pilot.json').read_text())
assert sorted(s['target_epsilon'] for s in summaries)==sorted(config['target_epsilons'])
baseline=json.loads((ROOT/'results/baseline/summary.json').read_text())
assert all(s['manifest_sha256']==baseline['manifest_sha256'] for s in summaries)

def table(frame):
    frame=frame.astype(object).where(frame.notna(),'undefined')
    return '\n'.join(['| '+' | '.join(frame.columns)+' |','| '+' | '.join(['---']*len(frame.columns))+' |']+
                     ['| '+' | '.join(str(v) for v in row)+' |' for row in frame.itertuples(index=False,name=None)])

privacy=pd.DataFrame([{k:s[k] for k in ['target_epsilon','epsilon','delta','noise_multiplier','sample_rate','expected_batch_size','steps','seconds']} for s in summaries])
display=comparison[['method','target_epsilon','epsilon','accuracy','balanced_accuracy','accuracy_loss_pp']].copy()
display['accuracy']=display.accuracy.map(lambda v:f'{v:.2%}')
display['balanced_accuracy']=display.balanced_accuracy.map(lambda v:f'{v:.2%}')
display['accuracy_loss_pp']=display.accuracy_loss_pp.map(lambda v:f'{v:.3f}')
groups=pd.concat([pd.read_csv(RUN/f'epsilon_{s["target_epsilon"]:g}'/'group_accuracy_losses.csv',dtype={'group':str}).assign(target_epsilon=s['target_epsilon']) for s in summaries],ignore_index=True)
sex=groups.loc[groups.attribute.eq('SEX'),['target_epsilon','group','n','accuracy_sgd','accuracy_dp','accuracy_loss_pp']].copy()
sex['group']=sex.group.map({'1':'Male','2':'Female'})
composition=json.loads((RUN/'composition.json').read_text())
log=f'''# DP-SGD pilot training log

**Completed November 16, 2026.** Three single-run development pilots; final test records were not loaded. The prediction target remains ESR==1. ESP is excluded. No LFR or equalized-odds intervention is applied yet.

## 1. What was trained

The same 77 → 64 → 32 → 1 MLP as the non-private baseline, with ReLU hidden layers, sigmoid scores, binary cross-entropy, SGD learning rate 0.05, no momentum, and threshold 0.5. All runs use 20 fixed training epochs, nominal batch size 512, 151,320 training records, and 30,264 development records. CPU execution uses four threads. The same initialization seed 285 and data-manifest fingerprint match the existing baseline.

The shared model factory is in [src/baseline.py](src/baseline.py); private training is in [src/dp.py](src/dp.py); settings are in [configs/dp_pilot.json](configs/dp_pilot.json). [Opacus PrivacyEngine documentation](https://opacus.ai/api/privacy_engine.html) describes the clipping, sampling and accounting interface used.

## 2. What changes under DP-SGD

1. Each step independently samples training people with Poisson inclusion probability q=1/296. Actual batch sizes vary. “Epoch” means 296 sampling steps, not exactly one visit to every person.
2. Opacus computes per-person gradients and clips their full-model L2 norm to C=1.
3. It adds Gaussian noise to the sum of clipped gradients. The noise multiplier below is the noise standard deviation relative to C; Opacus then normalizes the update by its expected batch size.
4. Noise is calibrated for each target epsilon with a fresh Rényi-DP accountant and fixed delta=1/(10×151320). Each accountant records 5,920 updates.
5. Development outcomes are treated as public auxiliary information. We keep the final epoch, without private-data early stopping. Raw training losses, individual gradients, and gradient statistics are not exported.

{table(privacy)}

The nominal batch size is 512, while q×N≈511.216 and Opacus's integer expected-batch normalizer is recorded in the table. These conventions come from its Poisson loader. Actual q, normalization and steps are logged rather than assuming batch_size/N exactly.

## 3. Accuracy compared with ordinary SGD

{table(display)}

Accuracy loss is ordinary-SGD accuracy minus DP accuracy, in percentage points. Positive means the DP model was less accurate. Balanced accuracy gives equal weight to the positive and negative classes.

## 4. Group losses and fairness

{table(sex.round(6))}

Full race and race-by-sex losses are in each run's group_accuracy_losses.csv. The comparison CSV also contains independence, separation and binary sufficiency gaps for sex and race. Components are saved in summary.json. Undefined rates stay undefined; rare groups can dominate max-minus-min gaps and require cautious interpretation. In this development split, full-race separation and sufficiency are undefined because a rare group's required denominators are absent; no categories were silently excluded to obtain a numerical gap.

In the epsilon≈1 pilot, the sex-group accuracy gap changes from {100*baseline['gaps']['SEX']['accuracy_gap']:.2f} to {100*summaries[0]['gaps']['SEX']['accuracy_gap']:.2f} percentage points, while the separation gap changes from {100*baseline['gaps']['SEX']['separation']:.2f} to {100*summaries[0]['gaps']['SEX']['separation']:.2f}. These illustrate different criteria moving in different directions, not a universal fairness improvement or deterioration.

These are one run at each budget with a fixed model initialization and independently randomized sampling/noise. They do not establish a statistically consistent epsilon trend or show whether noise or clipping caused a change. Ordinary SGD also uses shuffled fixed batches rather than Poisson batches. Additional seeds and a clipping-only/sampling-matched diagnostic would strengthen causal interpretation of the training changes.

## 5. Privacy interpretation

Epsilon/delta are accountant estimates for one training mechanism, conditional on fixed public reference/development/calibration data and preprocessing. Smaller epsilon means a stronger bound for the fixed delta. The public-auxiliary assumption is part of this benchmark design. The already released ordinary model and data audits do not provide DP for the training partition; the repository as a whole is not a private release.

For jointly releasing these three private runs, a conservative basic composition bound is epsilon={composition['basic_composition_epsilon']:.6f}, delta={composition['basic_composition_delta']:.9g}. Per-run budgets do not describe the entire sweep.

The pilots use secure_mode=False (research pseudorandom generators), not a cryptographically hardened production configuration. Model-initialization seed is recorded; separate sampling/noise seeds use local entropy and are not exported. A repeated run has the same settings but different random sampling/noise. The checkpoint stores model/accountant state without RNG secrets.

## 6. Files and reproduction

- [03_dp_sgd.ipynb](notebooks/03_dp_sgd.ipynb): read the settings, learning curves, group losses and fairness comparisons.
- results/dp_pilot/comparison.csv: overall and fairness comparisons with the baseline.
- Each epsilon folder: public development history, achieved budget, group metrics/losses, and ignored model checkpoint/row-level predictions.
- composition.json: conservative joint-release accounting for completed private runs.

```powershell
.\\.venv\\Scripts\\python.exe scripts\\train_dp.py
.\\.venv\\Scripts\\python.exe scripts\\report_dp.py
```

Completed runs are protected against accidental overwrite. For new replicates, use a new output directory in a copied configuration and record the run; additional releases compose. The report command reads saved artifacts and does not retrain. Next: inspect these pilots for the progress report, then plan repeated seeds and fairness interventions.
'''
(ROOT/'dp_training_log.md').write_text(log,encoding='utf-8')

md=nbf.v4.new_markdown_cell;code=nbf.v4.new_code_cell
nb=nbf.v4.new_notebook()
nb.cells=[md('''# 03 — DP-SGD: what changes when we add privacy?

Three pilots are completed at target epsilon 1, 4 and 8. This notebook reads saved results and never retrains by itself. The same employment target, 77 inputs, hidden widths 64/32, SGD learning rate, 20 epochs and development records match the baseline.

DP-SGD clips each person's gradient and adds Gaussian noise. Opacus tracks the resulting (epsilon, delta) bound. Smaller epsilon means stronger privacy at fixed delta. These are single-run development results; repeated runs are needed for conclusions.

See [the training log](../dp_training_log.md) for the full record. Test outcomes remain reserved.'''),
code('''from pathlib import Path
import json
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display
ROOT=next(p for p in [Path.cwd(),*Path.cwd().parents] if (p/'proposal.md').exists())
RUN=ROOT/'results/dp_pilot'
comparison=pd.read_csv(RUN/'comparison.csv')
summaries=[json.loads(p.read_text()) for p in sorted(RUN.glob('epsilon_*/summary.json'))]
display(pd.DataFrame([{k:s[k] for k in ['target_epsilon','epsilon','delta','noise_multiplier','max_grad_norm','sample_rate','expected_batch_size','steps']} for s in summaries]))'''),
md('''## 1. Read the privacy settings

target_epsilon is requested; epsilon is the accountant's achieved bound after training. Delta is fixed at approximately 6.61e-7. max_grad_norm=1 is the clipping norm, and noise_multiplier determines added noise relative to that norm. sample_rate is the per-person inclusion probability at each step. steps counts updates, including the accountant's updates.

Poisson sampling makes batch sizes variable. Each epoch is a fixed number of sampling steps. It does not guarantee visiting every person once. The reference scaler and encoder remain unchanged.'''),
code('''columns=['method','target_epsilon','epsilon','accuracy','balanced_accuracy','accuracy_loss_pp']
display(comparison[columns])
private=comparison.loc[comparison.method.eq('DP-SGD')].sort_values('epsilon')
baseline_accuracy=float(comparison.loc[comparison.method.eq('SGD'),'accuracy'].iloc[0])
fig,ax=plt.subplots(figsize=(6,3))
ax.plot(private.epsilon,100*private.accuracy,marker='o',label='DP-SGD')
ax.axhline(100*baseline_accuracy,color='gray',linestyle='--',label='Ordinary SGD')
ax.set(xlabel='Achieved epsilon (smaller = stronger privacy)',ylabel='Development accuracy (%)')
ax.legend();plt.tight_layout();plt.show()'''),
md('''## 2. Is the private model learning?

The table's accuracy values are fractions (0.75 means 75%). accuracy_loss_pp is baseline minus private accuracy in percentage points. Compare the learning curves below to see whether each model is improving or still learning at the fixed final epoch.

We deliberately do not log raw private-training losses. Development evaluation uses the designated public auxiliary partition.'''),
code('''fig,axes=plt.subplots(1,2,figsize=(10,3))
for s in summaries:
    h=pd.read_csv(RUN/f"epsilon_{s['target_epsilon']:g}"/'history.csv')
    axes[0].plot(h.epoch,h.development_accuracy,label=f"epsilon target {s['target_epsilon']:g}")
    axes[1].plot(h.epoch,h.epsilon,label=f"epsilon target {s['target_epsilon']:g}")
axes[0].axhline(baseline_accuracy,color='gray',linestyle='--',label='SGD final')
axes[0].set(xlabel='Epoch',ylabel='Development accuracy')
axes[1].set(xlabel='Epoch',ylabel='Privacy budget spent')
for ax in axes: ax.legend()
plt.tight_layout();plt.show()'''),
md('''## 3. Which groups lost accuracy?

These are model accuracy losses, not differences in label base rates. Positive loss means DP reduced that group's accuracy; negative means it improved it in this run. Sex codes are 1=male and 2=female. Race codes retain Census meanings from notebook 01/the processing log.

Always inspect counts and conditional denominators. Very small race/intersectional groups can have large unstable changes.'''),
code('''losses=[]
for s in summaries:
    frame=pd.read_csv(RUN/f"epsilon_{s['target_epsilon']:g}"/'group_accuracy_losses.csv',dtype={'group':str})
    frame['target_epsilon']=s['target_epsilon']
    losses.append(frame)
losses=pd.concat(losses,ignore_index=True)
columns=['target_epsilon','group','n','accuracy_sgd','accuracy_dp','accuracy_loss_pp','min_conditional_count']
display(losses.loc[losses.attribute.eq('SEX'),columns])
display(losses.loc[losses.attribute.eq('RAC1P'),columns])'''),
md('''## 4. What happened to fairness gaps?

Independence compares positive-prediction rates. Separation compares true-positive and false-positive rates. Binary sufficiency compares positive/negative predictive values. Smaller gaps mean closer agreement with that criterion; group accuracy losses answer a different question.

A blank/NaN gap means a required group rate is undefined. It is not a zero gap. Rare-group rates and three randomized pilot runs do not justify a universal conclusion about fairness.'''),
code('''for attribute in ['SEX','RAC1P']:
    columns=['method','target_epsilon']+[f'{attribute}_{name}' for name in ['accuracy_gap','independence','separation','sufficiency']]
    print(attribute)
    display(comparison[columns])
fig,axes=plt.subplots(1,2,figsize=(10,3))
for ax,attribute in zip(axes,['SEX','RAC1P']):
    for name in ['independence','separation','sufficiency']:
        ax.plot(private.epsilon,100*private[f'{attribute}_{name}'],marker='o',label=name)
    ax.set(title=attribute,xlabel='Achieved epsilon',ylabel='Group gap (percentage points)')
    ax.legend()
plt.tight_layout();plt.show()'''),
md('''## 5. Scope and your review notes

Each reported budget belongs to one private run, conditional on public auxiliary data. A joint release of several private models requires composition. The ordinary baseline and the repository's public-data audits are not DP releases. These experiments use research RNG settings, not cryptographically hardened deployment settings.

The MLP initialization seed is fixed; sampling/noise randomness is independent and not exported. Changes may reflect clipping, noise, sampling, and random variation. We need repeated seeds and can add a sampling-matched/clipping-only diagnostic to distinguish mechanisms.

- How large is the overall accuracy cost?
- Are sex-group losses similar? Which race-group estimates have enough support?
- Does a smaller gap accompany improved performance or worse performance for another group?
- Which findings can we cautiously report now, and which need repetitions?

Training entry point: `scripts/train_dp.py`; settings: `configs/dp_pilot.json`. Do not rerun completed pilots casually: new runs are new experiments and would add privacy cost if released.''')]
nb.metadata={'kernelspec':{'name':'python3','display_name':'Python (project .venv)','language':'python'},'language_info':{'name':'python'}}
for cell in nb.cells:
    if cell.cell_type=='code' and 'fig,' in cell.source:
        name = ('accuracy_vs_epsilon.png' if 'baseline_accuracy=float' in cell.source else
                'learning_and_privacy.png' if "h=pd.read_csv" in cell.source else 'fairness_gaps.png')
        cell.source=cell.source.replace('plt.tight_layout();plt.show()',
            "plt.tight_layout()\nFIG=ROOT/'figures/dp_pilot'\nFIG.mkdir(parents=True,exist_ok=True)\n"+
            f"fig.savefig(FIG/'{name}',dpi=180,bbox_inches='tight')\nplt.show()")
path=ROOT/'notebooks/03_dp_sgd.ipynb'
nbf.write(nb,path)
os.environ['JUPYTER_RUNTIME_DIR']=str(ROOT/'.jupyter/runtime')
os.environ['IPYTHONDIR']=str(ROOT/'.ipython')
NotebookClient(nb,timeout=180,kernel_name='python3',resources={'metadata':{'path':str(ROOT)}}).execute()
nbf.write(nb,path)
print('Wrote dp_training_log.md; built and executed notebooks/03_dp_sgd.ipynb')
