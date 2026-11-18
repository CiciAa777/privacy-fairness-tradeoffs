> **Historical V0 log.** The original results and this log are preserved under [V0/](V0/dp_training_log.md). The active five-seed, two-feature-policy rerun is tracked in [plan.md](plan.md) and `results/controlled_v2/`. Numerical results below describe only the original pilots.

# DP-SGD pilot training log

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

| target_epsilon | epsilon | delta | noise_multiplier | sample_rate | expected_batch_size | steps | seconds |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1.0 | 0.9991596010919738 | 6.608511763150938e-07 | 1.4111328125 | 0.0033783783783783786 | 511 | 5920 | 48.18898849992547 |
| 4.0 | 3.999411429217509 | 6.608511763150938e-07 | 0.740509033203125 | 0.0033783783783783786 | 511 | 5920 | 78.73706259997562 |
| 8.0 | 7.999650510019857 | 6.608511763150938e-07 | 0.5974578857421875 | 0.0033783783783783786 | 511 | 5920 | 67.61084580002353 |

The nominal batch size is 512, while q×N≈511.216 and Opacus's integer expected-batch normalizer is recorded in the table. These conventions come from its Poisson loader. Actual q, normalization and steps are logged rather than assuming batch_size/N exactly.

## 3. Accuracy compared with ordinary SGD

| method | target_epsilon | epsilon | accuracy | balanced_accuracy | accuracy_loss_pp |
| --- | --- | --- | --- | --- | --- |
| SGD | undefined | undefined | 77.08% | 75.73% | 0.000 |
| DP-SGD | 1.0 | 0.9991596010919738 | 75.82% | 73.67% | 1.259 |
| DP-SGD | 4.0 | 3.999411429217509 | 75.81% | 73.69% | 1.266 |
| DP-SGD | 8.0 | 7.999650510019857 | 75.78% | 73.63% | 1.299 |

Accuracy loss is ordinary-SGD accuracy minus DP accuracy, in percentage points. Positive means the DP model was less accurate. Balanced accuracy gives equal weight to the positive and negative classes.

## 4. Group losses and fairness

| target_epsilon | group | n | accuracy_sgd | accuracy_dp | accuracy_loss_pp |
| --- | --- | --- | --- | --- | --- |
| 1.0 | Male | 14826 | 0.804802 | 0.795832 | 0.897073 |
| 1.0 | Female | 15438 | 0.738114 | 0.722049 | 1.606426 |
| 4.0 | Male | 14826 | 0.804802 | 0.795494 | 0.930797 |
| 4.0 | Female | 15438 | 0.738114 | 0.722244 | 1.586993 |
| 8.0 | Male | 14826 | 0.804802 | 0.795697 | 0.910563 |
| 8.0 | Female | 15438 | 0.738114 | 0.721402 | 1.671201 |

Full race and race-by-sex losses are in each run's group_accuracy_losses.csv. The comparison CSV also contains independence, separation and binary sufficiency gaps for sex and race. Components are saved in summary.json. Undefined rates stay undefined; rare groups can dominate max-minus-min gaps and require cautious interpretation. In this development split, full-race separation and sufficiency are undefined because a rare group's required denominators are absent; no categories were silently excluded to obtain a numerical gap.

In the epsilon≈1 pilot, the sex-group accuracy gap changes from 6.67 to 7.38 percentage points, while the separation gap changes from 12.01 to 11.38. These illustrate different criteria moving in different directions, not a universal fairness improvement or deterioration.

These are one run at each budget with a fixed model initialization and independently randomized sampling/noise. They do not establish a statistically consistent epsilon trend or show whether noise or clipping caused a change. Ordinary SGD also uses shuffled fixed batches rather than Poisson batches. Additional seeds and a clipping-only/sampling-matched diagnostic would strengthen causal interpretation of the training changes.

## 5. Privacy interpretation

Epsilon/delta are accountant estimates for one training mechanism, conditional on fixed public reference/development/calibration data and preprocessing. Smaller epsilon means a stronger bound for the fixed delta. The public-auxiliary assumption is part of this benchmark design. The already released ordinary model and data audits do not provide DP for the training partition; the repository as a whole is not a private release.

For jointly releasing these three private runs, a conservative basic composition bound is epsilon=12.998222, delta=1.98255353e-06. Per-run budgets do not describe the entire sweep.

The pilots use secure_mode=False (research pseudorandom generators), not a cryptographically hardened production configuration. Model-initialization seed is recorded; separate sampling/noise seeds use local entropy and are not exported. A repeated run has the same settings but different random sampling/noise. The checkpoint stores model/accountant state without RNG secrets.

## 6. Files and reproduction

- [03_dp_sgd.ipynb](notebooks/03_dp_sgd.ipynb): read the settings, learning curves, group losses and fairness comparisons.
- results/dp_pilot/comparison.csv: overall and fairness comparisons with the baseline.
- Each epsilon folder: public development history, achieved budget, group metrics/losses, and ignored model checkpoint/row-level predictions.
- composition.json: conservative joint-release accounting for completed private runs.

```powershell
.\.venv\Scripts\python.exe scripts\train_dp.py
.\.venv\Scripts\python.exe scripts\report_dp.py
```

Completed runs are protected against accidental overwrite. For new replicates, use a new output directory in a copied configuration and record the run; additional releases compose. The report command reads saved artifacts and does not retrain. Next: inspect these pilots for the progress report, then plan repeated seeds and fairness interventions.
