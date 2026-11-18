# Overleaf progress report

Upload [progress_report_overleaf.zip](progress_report_overleaf.zip) using **New Project → Upload Project** in Overleaf. Select `main.tex` as the main document and use **pdfLaTeX**. The ZIP contains the report source, bibliography, three result figures, and compilation/sync instructions. No experiment data are needed.

`main.tex` is the course submission: exactly two pages including references, US Letter, one-inch margins, 11pt text, and single line spacing. The title rules and compact headings are inspired by NeurIPS; the official NeurIPS template is not used because the course specifies a different format. Both authors, Xiyin Xu and Shannon Saad, are included.

The report covers proposal feedback, completed experiments, preliminary observations and limitations, changes since the proposal, and the schedule through December 7. The next steps explicitly include at least three training runs per setting, means/standard deviations/error bars, and repeated ACSIncome baseline/DP comparisons; only single-run ACSEmployment pilots are complete. One table and one group-loss figure appear in the submission. The other two result plots are included in the ZIP for optional use.

For accessible explanations and interpreted results, start with [fairness/README.md](../fairness/README.md). The same folder contains [the illustrated definitions](../fairness/GUIDE.md), [complete observed tables](../fairness/RESULTS.md), and [a printable four-page guide](../fairness/fairness_metric_guide.pdf). Teaching diagrams use invented counts; observed result tables are labeled separately. These remain outside the submission ZIP.

See [OVERLEAF_SYNC.md](OVERLEAF_SYNC.md) for manual updates, direct Overleaf Git integration, and GitHub synchronization. Rebuilding the ZIP or pushing this repository does not automatically update an Overleaf project created by a ZIP upload.

The partner's original Markdown draft is preserved. `archive/extended_report.tex` preserves the earlier, longer report for reference; it is not the submission and is not included in the Overleaf ZIP. The two-page submission uses a compact inline bibliography, while `references.bib` retains the full bibliography for future expansion.

Current files in this folder: `main.tex` is the editable submission, `progress_report.pdf` is the compiled preview, and `progress_report_overleaf.zip` is the upload package. All three stay together.

From the repository root, rebuild with:

```powershell
python scripts/build_progress_report.py
```

This requires `pdflatex` and `pdfinfo` on PATH (or a Windows TeX Live installation). BibTeX is needed only if the source is changed to use an external bibliography. The script refreshes the copied figures from `figures/dp_pilot`, compiles the report, checks that it has exactly two pages, and writes:

- `reports/progress_report/progress_report.pdf` for review.
- `reports/progress_report/progress_report_overleaf.zip` for upload.

Temporary LaTeX files go under ignored `tmp/progress_report_build/`, keeping this folder clear. To compile on Overleaf, run pdfLaTeX twice.
