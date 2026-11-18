"""Compile and check the two-page progress report; package portable Overleaf sources."""
from pathlib import Path
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


def main():
    TEMP.mkdir(parents=True, exist_ok=True)
    for _ in range(2):
        result = subprocess.run(['pdflatex', '-interaction=nonstopmode', '-halt-on-error',
            '-output-directory='+TEMP.as_posix(), NAME+'.tex'], cwd=HERE, capture_output=True, text=True)
        if result.returncode:
            raise RuntimeError(result.stdout[-4000:]+result.stderr)
    log = (TEMP/(NAME+'.log')).read_text(encoding='utf-8', errors='replace')
    for problem in ['Overfull', 'Undefined control sequence', 'undefined references', 'undefined citations']:
        if problem in log:
            raise ValueError('Resolve LaTeX problem: '+problem)
    info = subprocess.check_output(['pdfinfo', str(TEMP/(NAME+'.pdf'))], text=True)
    assert re.search(r'Pages:\s+2\b', info), info
    assert re.search(r'Page size:\s+612 x 792 pts', info), info
    shutil.copyfile(TEMP/(NAME+'.pdf'), HERE/(NAME+'.pdf'))
    text = subprocess.check_output(['pdftotext', '-layout', str(HERE/(NAME+'.pdf')), '-'], text=True, encoding='utf-8')
    for expected in ['Cici Xu', 'Shannon Saad', 'additional Folktables tasks', 'ACSIncome', 'December 7, 2026']:
        assert expected in text, expected
    assert text.count('Figure 1:')==1 and text.count('Table 1:')==1
    source = (HERE/(NAME+'.tex')).read_text(encoding='utf-8')
    for expected in ['11pt,letterpaper', 'margin=1in', r'\linespread{1}']:
        assert expected in source
    assert not any(token in source for token in [r'\small', r'\footnotesize', r'\scriptsize'])
    # Portable source bundle: no local absolute paths, bibliography service, or result files needed.
    portable = [NAME+'.tex', 'figures/racial_impact.pdf', 'tables/sgd_dp1.tex', 'README.md']
    with zipfile.ZipFile(HERE/(NAME+'_overleaf.zip'), 'w', compression=zipfile.ZIP_DEFLATED) as archive:
        for path in portable:
            archive.write(HERE/path, path)
    provenance = json.loads((HERE/'asset_provenance.json').read_text(encoding='utf-8'))
    for path, sha in provenance['source_sha256'].items():
        assert hashlib.sha256((ROOT/path).read_bytes()).hexdigest()==sha, path
    validation = dict(pages=2, page_size='US Letter (612 x 792 pt)', class_option='11pt',
        margins='1 inch, all sides', line_spread=1, figures=1, tables=1,
        authors=['Cici Xu', 'Shannon Saad'], feedback_included=True,
        feedback_source='User-confirmed positive feedback and request for additional Folktables tasks',
        additional_task_status='ACSIncome planned, not evaluated',
        latex_overflow_or_unresolved_references=False, source_aggregates_unchanged=True,
        new_training_or_test_evaluation=False, visual_review='Pending inspection of latest rendered pages',
        output_sha256={path: hashlib.sha256((HERE/path).read_bytes()).hexdigest() for path in
                       [NAME+'.tex', NAME+'.pdf', NAME+'_overleaf.zip', 'figures/racial_impact.pdf', 'tables/sgd_dp1.tex']})
    (HERE/'validation.json').write_text(json.dumps(validation, indent=2)+'\n', encoding='utf-8')
    print('Compiled exactly 2 US Letter pages; 11pt/1-inch/single-spacing settings verified. Overleaf ZIP created.')


if __name__=='__main__':
    main()
