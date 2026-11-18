"""Compile local report edits; import/package Overleaf sources only when requested.

Default compilation preserves the downloaded ZIP byte for byte.
"""
from pathlib import Path
import argparse
import hashlib
import json
import re
import shutil
import subprocess
import zipfile

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
NAME = 'ECE285_progress_report'
TEMP = ROOT/'tmp/pdfs/progress_report'
PORTABLE = [NAME+'.tex', 'figures/racial_impact.pdf', 'tables/sgd_dp1.tex']


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--import-overleaf', action='store_true', help='Import report source/assets from the downloaded ZIP')
    parser.add_argument('--package', action='store_true', help='Explicitly replace the ZIP with current local source/assets')
    args = parser.parse_args()
    archive = HERE/(NAME+'_overleaf.zip')
    before = hashlib.sha256(archive.read_bytes()).hexdigest() if archive.exists() else None
    if args.import_overleaf:
        with zipfile.ZipFile(archive) as package:
            for filename in PORTABLE:
                target = HERE/filename
                assert target.resolve().is_relative_to(HERE.resolve())
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(package.read(filename))
    TEMP.mkdir(parents=True, exist_ok=True)
    for _ in range(2):
        run = subprocess.run(['pdflatex', '-interaction=nonstopmode', '-halt-on-error',
            '-output-directory='+TEMP.as_posix(), NAME+'.tex'], cwd=HERE, capture_output=True, text=True)
        if run.returncode:
            raise RuntimeError(run.stdout[-4000:]+run.stderr)
    log = (TEMP/(NAME+'.log')).read_text(encoding='utf-8', errors='replace')
    for problem in ['Undefined control sequence', 'undefined references', 'undefined citations']:
        if problem in log:
            raise ValueError('Resolve LaTeX problem: '+problem)
    info = subprocess.check_output(['pdfinfo', str(TEMP/(NAME+'.pdf'))], text=True)
    pages = int(re.search(r'Pages:\s+(\d+)', info)[1])
    shutil.copyfile(TEMP/(NAME+'.pdf'), HERE/(NAME+'.pdf'))
    if args.package:
        with zipfile.ZipFile(archive, 'w', compression=zipfile.ZIP_DEFLATED) as package:
            for filename in PORTABLE:
                package.write(HERE/filename, filename)
    after = hashlib.sha256(archive.read_bytes()).hexdigest() if archive.exists() else None
    if not args.package:
        assert before==after, 'Compilation must not replace the downloaded Overleaf ZIP'
    source = (HERE/(NAME+'.tex')).read_text(encoding='utf-8')
    validation = dict(pages=pages, letterpaper='612 x 792 pts' in info, class_11pt='11pt' in source,
        one_inch_geometry='margin=1in' in source, single_spacing=r'\linespread{1}' in source,
        negative_title_offset=r'\vspace*{-0.4in}' in source, small_table_font=r'\small' in source,
        layout_note='User Overleaf formatting preserved; geometry alone does not verify actual text margins.',
        overfull_boxes='Overfull' in log, unresolved_references=False,
        downloaded_zip_preserved=not args.package, no_training_or_new_evaluation=True,
        visual_review='Pending inspection of newly compiled pages',
        output_sha256={filename: hashlib.sha256((HERE/filename).read_bytes()).hexdigest()
                       for filename in [NAME+'.tex', NAME+'.pdf', NAME+'_overleaf.zip']})
    (HERE/'validation.json').write_text(json.dumps(validation, indent=2)+'\n', encoding='utf-8')
    print(f'Compiled {pages} pages. Downloaded ZIP '+('explicitly repackaged.' if args.package else 'preserved unchanged.'))


if __name__=='__main__':
    main()
