"""Compile the progress report and package a standalone Overleaf project."""
from pathlib import Path
import shutil
import subprocess
import zipfile
import re

ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/'reports/progress_report'
BUILD=SOURCE/'build'
BUILD.mkdir(exist_ok=True)
(SOURCE/'figures').mkdir(exist_ok=True)
FIGURES=['accuracy_vs_epsilon.png','group_loss_vs_epsilon.png','fairness_gaps.png']
for name in FIGURES:
    shutil.copy2(ROOT/'figures/dp_pilot'/name,SOURCE/'figures'/name)

def executable(name):
    candidate=shutil.which(name)
    if candidate: return candidate
    windows=Path('C:/texlive/2025/bin/windows')/(name+'.exe')
    if windows.exists(): return str(windows)
    raise RuntimeError(f'{name} not found; install TeX Live/MiKTeX or upload the source to Overleaf')

latex=executable('pdflatex')
def run(command,cwd):
    result=subprocess.run(command,cwd=cwd,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,encoding='utf-8',errors='replace')
    if result.returncode:
        print(result.stdout[-6000:])
        raise RuntimeError('LaTeX build failed; inspect reports/progress_report/build/main.log')

command=[latex,'-interaction=nonstopmode','-halt-on-error','-output-directory=build','main.tex']
run(command,SOURCE)
if '\\bibdata{' in (BUILD/'main.aux').read_text(encoding='utf-8'):
    shutil.copy2(SOURCE/'references.bib',BUILD/'references.bib')
    run([executable('bibtex'),'main'],BUILD)
run(command,SOURCE);run(command,SOURCE)
pdf_info=subprocess.check_output([executable('pdfinfo'),str(BUILD/'main.pdf')],text=True,encoding='utf-8',errors='replace')
page_count=re.search(r'^Pages:\s+(\d+)',pdf_info,re.MULTILINE)
if not page_count or int(page_count.group(1)) != 2:
    raise RuntimeError('The course submission must be exactly two pages; revise main.tex before packaging')
log=(BUILD/'main.log').read_text(encoding='utf-8',errors='replace')
if 'Overfull \\hbox' in log or 'Overfull \\vbox' in log or 'There were undefined references' in log:
    raise RuntimeError('Resolve layout overflow or undefined references before packaging')
outpdf=ROOT/'output/pdf';outpdf.mkdir(parents=True,exist_ok=True)
shutil.copy2(BUILD/'main.pdf',outpdf/'progress_report.pdf')
outzip=ROOT/'output/overleaf';outzip.mkdir(parents=True,exist_ok=True)
with zipfile.ZipFile(outzip/'progress_report_overleaf.zip','w',zipfile.ZIP_DEFLATED) as archive:
    for name in ['main.tex','references.bib','README.md']:
        archive.write(SOURCE/name,name)
    for name in FIGURES:
        archive.write(SOURCE/'figures'/name,'figures/'+name)
print('Built output/pdf/progress_report.pdf and output/overleaf/progress_report_overleaf.zip')
