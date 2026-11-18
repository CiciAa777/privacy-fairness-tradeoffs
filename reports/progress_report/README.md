# Overleaf progress report

Upload `output/overleaf/progress_report_overleaf.zip` from the repository root using **New Project → Upload Project** in Overleaf. Select `main.tex` as the main document and use **pdfLaTeX**. The ZIP contains the LaTeX source, bibliography, three result figures, and four optional teaching diagrams with their guide; compilation does not depend on repository-relative parent paths or the original data.

`main.tex` is the course submission: exactly two pages including references, US Letter, one-inch margins, 11pt text, and single line spacing. The title rules and compact headings are inspired by NeurIPS; the official NeurIPS template is not used because the course specifies a different format. Both authors, Xiyin Xu and Shannon Saad, are included.

The report covers proposal feedback, completed experiments, preliminary observations and limitations, changes since the proposal, and the schedule through December 7. The next steps explicitly include at least three training runs per setting, means/standard deviations/error bars, and repeated ACSIncome baseline/DP comparisons; only single-run ACSEmployment pilots are complete. One table and one group-loss figure appear in the submission. The other two result plots are included in the ZIP for optional use.

For accessible explanations, open [the visual metric guide](../../fairness_metric_guide.md) or [its printable four-page version](../../output/pdf/fairness_metric_guide.pdf). It has a separate image for each fairness metric with explanations underneath. These use invented counts and remain separate from the two-page submission. PNGs and the Markdown guide are also included in the ZIP for reference; they are not included by `main.tex`.

See [OVERLEAF_SYNC.md](OVERLEAF_SYNC.md) for manual updates, direct Overleaf Git integration, and GitHub synchronization. Rebuilding the ZIP or pushing this repository does not automatically update an Overleaf project created by a ZIP upload.

The partner's original Markdown draft is preserved. `extended_report.tex` preserves the earlier, longer report for reference; it is not the submission and is not included in the Overleaf ZIP. The two-page submission uses a compact inline bibliography, while `references.bib` retains the full bibliography for future expansion.

From the repository root, rebuild with:

```powershell
python scripts/build_progress_report.py
```

This requires `pdflatex` and `pdfinfo` on PATH (or a Windows TeX Live installation). BibTeX is needed only if the source is changed to use an external bibliography. The script refreshes the copied figures from `figures/dp_pilot`, compiles the report, checks that it has exactly two pages, and writes:

- `output/pdf/progress_report.pdf` for review.
- `output/overleaf/progress_report_overleaf.zip` for upload.

The build directory contains temporary LaTeX artifacts and is ignored by Git. To compile directly within this folder or on Overleaf, run pdfLaTeX twice.
