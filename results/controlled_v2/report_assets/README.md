# Controlled-v2 report assets

Start with [REPORT_FINDINGS.md](../REPORT_FINDINGS.md).

For the two-page report, use Figure 1 (`figures/figure1_racial_impact.pdf`), Figure 2 (`figures/figure2_accuracy_ablations.pdf`), Table 1 (`tables/table1_sgd_vs_dp1.tex`) and Table 2 (`tables/table2_privacy_budgets.tex`). Figure 3 is optional supplementary material; it repeats Table 1. The two full-width figures total about 5.8 inches in height. The original progress report has not been edited.

- Figures are vector PDF/SVG and 350-dpi PNG, 6.5 inches wide, with sample SD error bars over five seeds. Small budget contrasts use a clearly labeled paired-difference panel.
- [Captions](captions.md) state populations, directions, units, uncertainty and ablation limitations.
- The two summary tables are available as Markdown, numeric CSV and Overleaf-ready LaTeX (requires `booktabs`). Numeric CSV means/SDs are percentages or pp as explicitly labeled, unlike raw aggregate fractions. LaTeX tables have no external-file dependency.
- Other files in `tables/` are machine-readable values behind figures and training contrasts, not extra summary tables to include.
- `source_provenance.json` records hashes of the existing aggregate CSVs and findings used. `validation.json` records independent numerical checks and source preservation. `visual_review.json` records the inspected asset hashes and PDF/LaTeX checks. Historical audit/plot-plan files remain unchanged snapshots of their pre-figure stage.

Rebuild from the repository root:

```powershell
.\.venv\Scripts\python.exe scripts\build_report_assets.py
```

The builder reads only existing result files. It never loads training/test matrices or invokes model training, score diagnostics, new clipping experiments, or fairness interventions. Original results, checkpoints, predictions and aggregates are byte-checked before and after generation. Source artifacts with changed protocols should be reviewed before regenerating a new report.
