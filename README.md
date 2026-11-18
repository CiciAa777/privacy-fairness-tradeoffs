# Privacy-Fairness Tradeoffs

A Folktables study of DP-SGD, unequal group accuracy losses and fairness. **60 controlled runs completed; next priority: clipping-norm sensitivity and more Folktables tasks.**

Read only these four notes:

- [project_log.md](project_log.md): data processing, feature/label definitions, settings, metrics and decision history.
- [findings.md](findings.md): the three main findings, group losses, fairness and ablation tables.
- [plan.md](plan.md): completed progress and the next experiments through December 7.
- This README: where things live and how to run them.

## Files

| Folder | Contents |
| --- | --- |
| `src/` | Data/model/DP/metrics and aggregation implementation |
| `scripts/` | Prepare data, run/verify/summarize experiments, build figure assets |
| `configs/` | Experiment settings and a small historical privacy/baseline reference |
| `data/` | Ignored raw/prepared data, fixed split assignments; keep local |
| `results/controlled_v2/` | All current run records, aggregate CSV/JSON, audit checks, figure assets; local models/predictions remain ignored |
| `reports/progress_report/` | Current Overleaf ZIP, synchronized source, compiled PDF, figure/table and build tools |
| `tests/` | Numerical DP/metrics/aggregation checks |

V0 and obsolete notes/generators were removed from the current tree. Tracked history remains recoverable at `21d50cf`; ignored legacy artifacts have a separate local backup documented in the project log. Current experimental data/results were retained.

## Environment and commands

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

On a **fresh clone only**, prepare/download data with `scripts/prepare_data.py`, then run controlled `--stage prepare`. Do not overwrite existing prepared data or split assignments. The original cohort pipeline uses split seed285 and reference-only preprocessing. The current study settings are in `configs/controlled_v2.json`.

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
.\.venv\Scripts\python.exe scripts/run_controlled.py --stage verify
.\.venv\Scripts\python.exe scripts/run_controlled.py --stage summarize
.\.venv\Scripts\python.exe scripts/build_report_assets.py
```

The last three commands verify existing models/dev scores, regenerate numeric aggregates, and build figure/table assets respectively; they do not train or evaluate the final test set. They do not rewrite the maintained log/findings. See the plan before starting new work. The runner also supports `smoke`, `main`, and `controls`; completed matching runs are skipped. Changed protocol/source fingerprints are rejected. For stale `.running` locks, verify that the recorded process has ended before removing them. Never export private sampler/noise RNG state.

## Overleaf workflow

After downloading a ZIP, replace `reports/progress_report/ECE285_progress_report_overleaf.zip`, then:

```powershell
.\.venv\Scripts\python.exe reports/progress_report/build_report.py --import-overleaf
```

This imports the main source/figure/table and compiles a local PDF, preserving the downloaded ZIP unchanged. Without flags, the command only compiles local source. Use `--package` **only when intentionally replacing the ZIP with local edits**. To regenerate the report's compact figure/table from project results, run `reports/progress_report/build_figures.py` explicitly; this can replace those figure/table assets.

Current [report PDF](reports/progress_report/ECE285_progress_report.pdf), [LaTeX source](reports/progress_report/ECE285_progress_report.tex), and [Overleaf ZIP](reports/progress_report/ECE285_progress_report_overleaf.zip). The imported user version compiles to three pages; its edits are preserved. Compilation diagnostics are in `validation.json` next to the report.

Only individual model-training releases have DP bounds under fixed public auxiliary-data assumptions. Non-private baselines/audits and multiple releases require separate interpretation/composition; see the project log.
