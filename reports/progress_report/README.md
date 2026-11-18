# ECE 285 progress report - November 18, 2026

Current report: **ECE285_progress_report.pdf**. Authors: **Cici Xu and Shannon Saad**.

The report has exactly two US Letter pages, 11pt body/table/caption/reference text, one-inch margins, and single line spacing. It uses the standard article class with restrained NeurIPS-inspired title rules and headings. One compact racial-impact figure and one table fit the page limit; the clipping ablation is described numerically. The original report remains archived under `V0/reports/progress_report/`.

The six sections cover objectives, proposal feedback and independent refinements, a concise setup list, preliminary five-seed findings, remaining work through December 7, and changes from the proposal. Instructor feedback encouraging more Folktables tasks is included as supplied by the team. ACSIncome, LFR/EO, additional seeds and clipping-norm sensitivity are planned, not claimed as completed.

## Overleaf

Upload **ECE285_progress_report_overleaf.zip** as a new project, or upload its contents into the existing project. Select **ECE285_progress_report.tex** as the main document and **pdfLaTeX** as compiler. The ZIP contains the LaTeX source, one vector figure, the generated table, and this README. References are included in the main source; no BibTeX step or repository checkout is needed to compile it. The Python builders are repository tools and are not needed on Overleaf.

## Rebuild locally

From the repository root:

```powershell
.\.venv\Scripts\python.exe reports\progress_report\build_figures.py
.\.venv\Scripts\python.exe reports\progress_report\build_report.py
```

The first command checks frozen source values and rebuilds the compact figure (PDF/PNG/SVG) and table from `results/controlled_v2/`. The second compiles twice with pdfLaTeX, checks page count/formatting/overflow/unresolved references with Poppler, and creates the PDF and portable ZIP. TeX/auxiliary files remain under ignored `tmp/pdfs/progress_report/`. After rebuilding, render and inspect both pages before submission; `validation.json` records the inspected output hashes.

## Evidence and references

All experimental numbers come from `results/controlled_v2/aggregate/` and the verified `report_assets/tables/`; see `asset_provenance.json`. Accuracy losses are paired within policy/group/initialization. Fairness gaps are computed within each seed before averaging. SD is training-seed variation, not population uncertainty. The main experiment includes sex/race predictors; exclusion is a secondary sensitivity check. The report does not modify raw results or evaluate test data.

Citation metadata checked against primary sources:

- Bagdasaryan et al. (2019), [Differential Privacy Has Disparate Impact on Model Accuracy](https://proceedings.neurips.cc/paper/2019/hash/fc0de4e0396fff257ea362983c2dda5a-Abstract.html).
- Ding et al. (2021), [Retiring Adult: New Datasets for Fair Machine Learning](https://arxiv.org/abs/2108.04884).
- Hardt et al. (2016), [Equality of Opportunity in Supervised Learning](https://arxiv.org/abs/1610.02413).
- Zemel et al. (2013), [Learning Fair Representations](https://proceedings.mlr.press/v28/zemel13.html).
