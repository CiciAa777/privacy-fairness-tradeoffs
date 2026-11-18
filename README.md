# Privacy–Fairness Tradeoffs

We study whether DP-SGD changes employment-prediction accuracy disproportionately across sex and race groups, and whether LFR or equalized-odds post-processing helps.

| Read this | For this purpose |
| --- | --- |
| [proposal.md](proposal.md) | Research questions, dataset/features/target, fairness definitions, related work, and privacy assumptions |
| [plan.md](plan.md) | Dated phases, today's tasks, milestones, and fallback priorities |
| [data_processing_log.md](data_processing_log.md) | What preparation actually did, observed column values/code meanings, current findings, and choices to review |
| [02_baseline.ipynb](notebooks/02_baseline.ipynb) | Training settings, loss curves, majority comparison, and development accuracy/fairness results |
| [03_dp_sgd.ipynb](notebooks/03_dp_sgd.ipynb) | Completed DP-SGD pilots, privacy budgets, learning curves, and group comparisons |
| [dp_training_log.md](dp_training_log.md) | Exact DP training/accounting settings, preliminary results, and interpretation limits |
| [Fairness: start here](reports/fairness/README.md) | Desired rate directions, interpreted White/Black comparisons, and links to illustrated definitions and all result tables |

**Progress report:** November 18, 2026. **Final report:** December 7, 2026.

## Progress report for Overleaf

The LaTeX report by Xiyin Xu and Shannon Saad is ready as a standalone [Overleaf ZIP](reports/progress_report/progress_report_overleaf.zip), with a compiled [PDF preview](reports/progress_report/progress_report.pdf). In Overleaf, choose **New Project → Upload Project**, upload the ZIP, and compile `main.tex` using **pdfLaTeX**.

All report material is under `reports/`: [progress_report/](reports/progress_report/README.md) contains the current LaTeX source, PDF, Overleaf ZIP, and an `archive/` for the older extended draft; [fairness/](reports/fairness/README.md) combines the explanations, interpreted comparisons, complete tables, teaching images, and printable guide. Temporary build files go under ignored `tmp/`. There is no separate `output/` folder.

Rebuild the report PDF and ZIP with `python scripts/build_progress_report.py` when TeX Live/MiKTeX is installed. The original partner Markdown draft and experiment plots are preserved.

Recommended workflow:

1. Agree on definitions in the proposal and check them against a data-audit notebook.
2. Implement reusable data preparation, training, and evaluation in Python modules; call them from notebooks and scripts.
3. Save configurations, split indices, and result tables so comparisons can be reproduced.
4. Update task checkboxes in the plan; record research-design changes in the proposal before running affected experiments.
5. Write the progress/final reports from saved results and figures.

## Set up and get the data

A collaborator can clone this repository and run the preparation script to download and process the data locally. No manual CSV transfer is needed. Raw/processed data, environments, checkpoints, and row-level predictions are intentionally excluded from Git; aggregate results and executed notebooks are included.

Use Python 3.14 (tested here) and a new virtual environment. Run these commands in a Windows PowerShell terminal:

```powershell
git clone https://github.com/CiciAa777/privacy-fairness-tradeoffs.git
cd privacy-fairness-tradeoffs
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe scripts\prepare_data.py
```

On macOS/Linux, use `.venv/bin/python` instead of `.\.venv\Scripts\python.exe` in the commands after environment creation. Activation is optional because these commands invoke the environment's interpreter directly.

The first preparation run requires an internet connection to the Census Bureau. It downloads California's 2018 ACS 1-Year person data (the extracted raw CSV is about 267 MB). Later runs reuse the local cache. It applies the documented filters, drops ESP, creates the fixed splits, fits preprocessing on reference records, and writes the encoded matrices and descriptive audit tables. It does not train a model.

All dependencies are listed in the single [requirements.txt](requirements.txt). Exact library versions used for an experiment are recorded in its data manifest and model summary. The current machine originally reused system scientific packages; collaborators should use the isolated environment above.

## Inspect the prepared data

Open [01_data_audit.ipynb](notebooks/01_data_audit.ipynb). In the notebook kernel picker, choose the interpreter at `.venv\Scripts\python.exe`, then run cells from the top. The Markdown explanations and questions guide the review. The notebook's final Markdown cell is for your observations.

Preparation downloads Census data once and caches it under the ignored `data/raw/` directory:

```powershell
.\.venv\Scripts\python.exe scripts\prepare_data.py
```

Processed sparse matrices, aligned labels/protected metadata, split assignments, and the preparation manifest are under `data/processed/`. Descriptive audit tables are under `results/data_audit/`. Model inputs exclude sex, race, the target, and survey weights. Learned preprocessing uses the reference subset only.

To reproduce the baseline on newly prepared local data, run the non-private training command below. The data manifest includes library versions, so regenerate the baseline locally before comparing it with new private runs. Saved notebook outputs can be read immediately; rerunning their code requires the local data/results they reference.

## Non-private baseline

ESP was dropped by user decision; the 14 remaining input variables become 77 encoded columns. The initial model is an MLP with hidden widths 64 and 32, trained for 20 epochs with ordinary SGD and binary cross-entropy. It predicts the existing ESR==1 target. This initial run uses development data for reporting; it never loads the test set.

```powershell
.\.venv\Scripts\python.exe scripts\train_baseline.py
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

Settings are in [configs/baseline.json](configs/baseline.json). Reusable training and metric functions are in `src/baseline.py` and `src/metrics.py`; the script is the entry point. Outputs under `results/baseline/` include loss history, development predictions, group metrics, summary/configuration, and the ignored model checkpoint. Re-running the same output directory overwrites its artifacts. The summary records PyTorch's version and a fingerprint of the data manifest.

Inspect notebook 02 as the non-private reference for the completed DP-SGD pilots below. Several seeds/privacy budgets are needed before drawing privacy–fairness conclusions. LFR and EO follow after the DP pilots.

## DP-SGD pilots

The three 20-epoch development pilots at epsilon 1, 4, and 8 are completed using Opacus 1.6.0, RDP accounting, Poisson sampling, clipping norm 1, and delta=1/(10*n_train). See notebook 03 and the DP training log for results. The same model factory and metrics are shared with the ordinary baseline. Sampling/noise RNG seeds are not exported; a repeated run produces a new randomized experiment.

```powershell
.\.venv\Scripts\python.exe scripts\train_dp.py
.\.venv\Scripts\python.exe scripts\report_dp.py
```

The completed pilot summaries are already included in Git, so the training command above refuses to overwrite them, including on a fresh clone. For a new replicate, copy `configs/dp_pilot.json`, change its `output_dir` to an unused directory, then run `scripts/train_dp.py --config path/to/copied_config.json` after regenerating the local baseline. Account for additional releases. The report command currently reads the original `results/dp_pilot/` runs and rebuilds their notebook/log/PNG figures without retraining; it does not automatically report a new output directory. Individual predictions and checkpoints stay ignored by Git. Budgets apply per private run under the documented public-auxiliary assumption; the ordinary baseline and repository as a whole are not DP releases.

Next: inspect the pilot findings for the progress report, then repeat seeds and implement LFR/EO. Full-race gaps can be undefined for rare groups; retain the counts and prespecify supported comparisons rather than silently omitting categories.
