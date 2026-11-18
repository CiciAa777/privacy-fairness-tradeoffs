# Privacy-Fairness Tradeoffs

The active experiment is the **controlled v2 baseline/DP-SGD rerun**. The original single-run pilots, reports, figures, notebooks, and settings are preserved in [V0/](V0/SETTINGS.txt).

**Completed:** 40 main runs plus 20 non-private controls, five seeds per setting. Start with [the report findings](results/controlled_v2/REPORT_FINDINGS.md) and [publication figures, captions, and tables](results/controlled_v2/report_assets/README.md). The [detailed findings](results/controlled_v2/FINDINGS.md), [group comparisons](results/controlled_v2/GROUP_COMPARISONS.md), and [aggregate conventions](results/controlled_v2/aggregate/README.md) retain the full analysis. Interventions and new sensitivity experiments remain planned.

Read [plan.md](plan.md) for the execution stages and changes, [proposal.md](proposal.md) for research definitions, and [the exact rerun request](V0/CONTROLLED_RERUN_REQUEST.txt) for requirements. The original progress-report source/PDF are in [V0/reports/progress_report/](V0/reports/progress_report/README.md); they remain unchanged.

## Controlled experiment

- California 2018 ACS Employment, original age/weight filters and fixed splits; Y=(ESR==1).
- Primary: include SEX/RAC1P as categorical predictors (87 encoded columns). Secondary: original sensitive-feature exclusion (77 columns).
- Both policies use the same reference-only preprocessing rules, labels, cohort, and original training/development assignments. Test outcomes remain reserved.
- MLP input->64->32->1; ReLU; SGD lr0.05, momentum0; 20 epochs; batch512; threshold0.5.
- Five initialization seeds: 42,123,285,2026,2027. SGD + DP epsilon1/4/8 = 40 main runs.
- Poisson unclipped and clipped C=1 noise-free controls = 20 extra non-private runs.
- No LFR/EO results yet. Report figures were generated afterward from the existing five-seed aggregates.

## Environment and data

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

On a fresh clone, download/prepare the original fixed cohort with `.\.venv\Scripts\python.exe scripts\prepare_data.py`. This preparation step creates the original person-level split assignments using seed285; do not rerun it over an existing study's prepared data. The controlled preparation reuses those assignments and does not modify them. Raw data and processed matrices stay local under ignored `data/`.

## Execute in order

```powershell
.\.venv\Scripts\python.exe scripts\run_controlled.py --stage prepare
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
.\.venv\Scripts\python.exe scripts\run_controlled.py --stage smoke
.\.venv\Scripts\python.exe scripts\run_controlled.py --stage main
.\.venv\Scripts\python.exe scripts\run_controlled.py --stage controls
.\.venv\Scripts\python.exe scripts\run_controlled.py --stage verify
.\.venv\Scripts\python.exe scripts\run_controlled.py --stage summarize
```

Smoke runs use disjoint subsets of the **public reference** partition, not private training or final test records. They check the pipeline without adding private-training releases. Settings are in [configs/controlled_v2.json](configs/controlled_v2.json).

Completed matching runs are skipped automatically. To resume interrupted main/control execution, rerun the same command. Partial runs restart from their initial seed; completed runs are never overwritten. A changed protocol/source fingerprint causes an error instead of silently mixing experiments. An interrupted process may leave a `.running` marker: verify its PID is no longer running before removing that marker and restarting. `--policies include_sensitive` and `--seeds 42` select subsets for execution; they do not change the configured study grid.

## Outputs and privacy scope

To regenerate the report assets from saved results, without training or overwriting the experimental aggregates:

```powershell
.\.venv\Scripts\python.exe scripts\build_report_assets.py
```

The builder independently verifies group rates, per-seed gaps, paired differences and mean/sample SD. It writes `results/controlled_v2/REPORT_FINDINGS.md` and `report_assets/` with two main figures, one optional figure (PDF/PNG/SVG), two summary tables (CSV/Markdown/LaTeX), captions and provenance. Historical `audit/validation.json` and `plot_plan.json` remain snapshots of the stage before figures were generated. Proposed clipping-norm/score diagnostics and EO/LFR have not been executed.

New outputs are under `results/controlled_v2/`: `audit/`, `runs/<policy>/<method>/seed_<seed>/`, `smoke/`, and later `aggregate/`. Cached matrices/metadata, checkpoints, and row-level development predictions remain ignored by Git. Published identity hashes describe fixed partition IDs/public manifest, not private training labels or feature values.

DP-SGD uses Poisson sampling, clipping C=1, RDP accounting, and delta=1/(10*n_train). Baseline and diagnostic controls are **not private**. Per-run individual-record privacy bounds assume fixed public reference/development/calibration data; releasing multiple private outputs requires composition. The whole repository, public descriptive audits, and non-private controls are not an end-to-end DP release. Research RNG mode is retained; sampling/noise RNG state is never exported.

Household overlap is audited, but the original person-level splits are retained. These are record-level, not household-level privacy claims. Repeated-seed variability measures training variation, not population-sampling uncertainty.

Legacy entry points and their exact source are preserved in `V0/scripts/` and `V0/src/` for historical inspection. Use the controlled runner for new experiments rather than regenerating the old report/pilots.
