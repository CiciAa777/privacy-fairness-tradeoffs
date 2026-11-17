# Overleaf progress report

Upload `output/overleaf/progress_report_overleaf.zip` from the repository root using **New Project → Upload Project** in Overleaf. Select `main.tex` as the main document and use **pdfLaTeX**. The ZIP contains the LaTeX source, bibliography and three local figures; it does not depend on repository-relative parent paths or the original data.

The source follows the partner's Markdown draft, with actual figure/table cross-references and citations. Statistical language was qualified where one run cannot establish run-to-run noise or rule out overfitting. The draft Markdown is preserved. The author names are Xiyin Xu and Shannon Saad, as provided for this report.

From the repository root, rebuild with:

```powershell
python scripts/build_progress_report.py
```

This requires `pdflatex` and `bibtex` on PATH (or a Windows TeX Live installation). It refreshes the copied figures from `figures/dp_pilot`, compiles the report, and writes:

- `output/pdf/progress_report.pdf` for review.
- `output/overleaf/progress_report_overleaf.zip` for upload.

The build directory contains temporary LaTeX artifacts and is ignored by Git. To compile directly within this folder: run pdfLaTeX, BibTeX, then pdfLaTeX twice.
