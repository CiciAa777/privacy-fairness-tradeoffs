# Privacy–Fairness Tradeoffs

We study whether DP-SGD changes employment-prediction accuracy disproportionately across sex and race groups, and whether LFR or equalized-odds post-processing helps.

| Read this | For this purpose |
| --- | --- |
| [proposal.md](proposal.md) | Research questions, dataset/features/target, fairness definitions, related work, and privacy assumptions |
| [plan.md](plan.md) | Dated phases, today's tasks, milestones, and fallback priorities |
| [data_processing_log.md](data_processing_log.md) | What preparation actually did, observed column values/code meanings, current findings, and choices to review |
| [02_baseline.ipynb](notebooks/02_baseline.ipynb) | Training settings, loss curves, majority comparison, and development accuracy/fairness results |

**Progress report:** November 18, 2026. **Final report:** December 7, 2026.

Recommended workflow:

1. Agree on definitions in the proposal and check them against a data-audit notebook.
2. Implement reusable data preparation, training, and evaluation in Python modules; call them from notebooks and scripts.
3. Save configurations, split indices, and result tables so comparisons can be reproduced.
4. Update task checkboxes in the plan; record research-design changes in the proposal before running affected experiments.
5. Write the progress/final reports from saved results and figures.

## Inspect the prepared data

Open [01_data_audit.ipynb](notebooks/01_data_audit.ipynb). In the notebook kernel picker, choose the interpreter at `.venv\Scripts\python.exe`, then run cells from the top. The Markdown explanations and questions guide the review. The notebook's final Markdown cell is for your observations.

Preparation downloads Census data once and caches it under the ignored `data/raw/` directory:

```powershell
.\.venv\Scripts\python.exe scripts\prepare_data.py
```

Processed sparse matrices, aligned labels/protected metadata, split assignments, and the preparation manifest are under `data/processed/`. Descriptive audit tables are under `results/data_audit/`. Model inputs exclude sex, race, the target, and survey weights. Learned preprocessing uses the reference subset only.

The project environment was created with `python -m venv --system-site-packages .venv`, reusing existing scientific packages. `requirements-data.txt` lists direct dependencies; the data manifest records the scientific-library versions actually used. This environment is for data preparation; compatibility with the later PyTorch/Opacus stack still needs checking.

## Non-private baseline

ESP was dropped by user decision; the 14 remaining input variables become 77 encoded columns. The initial model is an MLP with hidden widths 64 and 32, trained for 20 epochs with ordinary SGD and binary cross-entropy. It predicts the existing ESR==1 target. This initial run uses development data for reporting; it never loads the test set.

```powershell
.\.venv\Scripts\python.exe scripts\train_baseline.py
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

Settings are in [configs/baseline.json](configs/baseline.json). Reusable training and metric functions are in `src/baseline.py` and `src/metrics.py`; the script is the entry point. Outputs under `results/baseline/` include loss history, development predictions, group metrics, summary/configuration, and the ignored model checkpoint. Re-running the same output directory overwrites its artifacts. The summary records PyTorch's version and a fingerprint of the data manifest.

Next: inspect notebook 02, then implement DP-SGD with the same inputs, target, architecture, and evaluation code. Several seeds/privacy budgets are needed before drawing privacy–fairness conclusions. LFR and EO follow after the DP pilots.
