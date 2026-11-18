r"""Draw educational metric examples, using invented counts and the project metrics.

No experiment data are read, and no models are trained.
Run: .\.venv\Scripts\python.exe scripts\build_fairness_guide.py
"""
from pathlib import Path
import os
import sys

ROOT = Path(__file__).resolve().parents[1]
cache = ROOT / 'tmp/matplotlib'
cache.mkdir(parents=True, exist_ok=True)
os.environ.setdefault('MPLCONFIGDIR', str(cache))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.ticker import PercentFormatter
import numpy as np

sys.path.insert(0, str(ROOT))
from src.metrics import binary_metrics

OUT = ROOT / 'reports/fairness/figures'
OUT.mkdir(parents=True, exist_ok=True)
PDF = ROOT / 'reports/fairness/fairness_metric_guide.pdf'
PDF.parent.mkdir(parents=True, exist_ok=True)

# The same two hypothetical confusion matrices are used in every image.
COUNTS = {'Group A': (40, 40, 10, 10), 'Group B': (30, 40, 20, 10)}
metrics = {}
for group, (tp, tn, fp, fn) in COUNTS.items():
    y = np.array([1] * tp + [0] * tn + [0] * fp + [1] * fn)
    pred = np.array([1] * tp + [0] * tn + [1] * fp + [0] * fn)
    metrics[group] = binary_metrics(y, pred)

SPECS = [
    dict(name='accuracy_parity', title='Accuracy parity',
         question='Are predictions equally often correct for each group?',
         rates=[('accuracy', 'Everyone in the group', 'Accuracy',
                 '(TP + TN) / everyone', [(80, 100), (70, 100)])],
         memory='Goal: equally high accuracy; accuracy gap toward zero.',
         explanation='A is correct for 80 of 100 people; B for 70 of 100. The accuracy gap is 10 percentage points.\n'
                     'This compares groups within ONE model. Accuracy loss instead compares SGD with DP-SGD.'),
    dict(name='independence', title='Independence / demographic parity',
         question='Does each group receive the same proportion of positive predictions?',
         rates=[('selection_rate', 'Everyone in the group', 'Predicted positive',
                 '(TP + FP) / everyone', [(50, 100), (50, 100)])],
         memory='Goal: equal positive-prediction rates; no target rate of 100%.',
         explanation='Both groups receive 50 positive predictions out of 100: the independence gap is 0 points.\n'
                     'We do not check whether those predictions are correct. Other fairness gaps can remain large.'),
    dict(name='separation', title='Separation / equalized odds',
         question='Within the same ACTUAL class, does the model behave similarly across groups?',
         rates=[('tpr', 'Actual positives only', 'TPR: positives found',
                 'TP / (TP + FN)', [(40, 50), (30, 40)]),
                ('fpr', 'Actual negatives only', 'FPR: false alarms',
                 'FP / (FP + TN)', [(10, 50), (20, 60)])],
         memory='Goal: equal TPR and FPR across groups; high TPR, low FPR.',
         explanation='TPR gap = 5 points; FPR gap = 13.3 points. Separation gap = the larger one = 13.3 points.\n'
                     'We compare both components across groups; we do not want a high false-positive rate.'),
    dict(name='sufficiency', title='Sufficiency for binary predictions',
         question='Within the same PREDICTED class, are predictions equally reliable across groups?',
         rates=[('ppv', 'Predicted positives only', 'PPV: positive reliability',
                 'TP / (TP + FP)', [(40, 50), (30, 50)]),
                ('npv', 'Predicted negatives only', 'NPV: negative reliability',
                 'TN / (TN + FN)', [(40, 50), (40, 50)])],
         memory='Goal: equal PPV and NPV across groups; high reliability for both.',
         explanation='PPV gap = 20 points; NPV gap = 0 points. Sufficiency gap = the larger one = 20 points.\n'
                     'We check BOTH positive and negative predictions. This is not a score-calibration test.'),
]

plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 11})
with PdfPages(PDF) as pdf:
    for spec in SPECS:
        fig = plt.figure(figsize=(11, 6.5), facecolor='white')
        fig.text(.06, .94, spec['title'], fontsize=22, weight='bold', color='#18334d')
        fig.text(.06, .885, spec['question'], fontsize=12)
        fig.text(.06, .835, 'ILLUSTRATION ONLY: invented counts, not project results. Positive means ESR = 1.',
                 fontsize=10, color='#60656b')
        count_text = 'Same example throughout: A (TP 40, TN 40, FP 10, FN 10); B (TP 30, TN 40, FP 20, FN 10).'
        fig.text(.06, .79, count_text, fontsize=9.5, color='#60656b')

        n = len(spec['rates'])
        gaps = []
        for index, (key, subset, label, formula, fractions) in enumerate(spec['rates']):
            y0, height = ((.40, .24) if n == 1 else (.52 - index * .22, .125))
            ax = fig.add_axes([.34, y0, .55, height])
            values = [metrics[g][key] for g in COUNTS]
            for value, (numerator, denominator) in zip(values, fractions):
                assert np.isclose(value, numerator / denominator)
            ax.barh([1, 0], values, color=['#2877ad', '#dd8543'], height=.48)
            ax.set_yticks([1, 0], list(COUNTS))
            ax.set_xlim(0, 1.10)
            ax.set_xticks([0, .25, .5, .75, 1])
            ax.xaxis.set_major_formatter(PercentFormatter(1))
            ax.grid(axis='x', alpha=.16)
            ax.set_axisbelow(True)
            for spine in ax.spines.values():
                spine.set_visible(False)
            ax.tick_params(length=0)
            for yvalue, value, (numerator, denominator) in zip([1, 0], values, fractions):
                ax.text(value + .012, yvalue, f'{100*value:.1f}%', va='center', weight='bold')
                ax.text(.012, yvalue, f'{numerator}/{denominator}', va='center', color='white', weight='bold')
            fig.text(.06, y0 + height * .78, subset, fontsize=11, weight='bold')
            fig.text(.06, y0 + height * .48, label, fontsize=10)
            fig.text(.06, y0 + height * .17, formula, fontsize=10, color='#60656b')
            gaps.append(max(values) - min(values))

        gap = max(gaps)
        fig.text(.06, .225, f'Our gap = {100*gap:.1f} percentage points. Smaller gap = closer parity.',
                 fontsize=13, weight='bold', color='#18334d')
        fig.text(.06, .16, spec['memory'], fontsize=12, weight='bold')
        fig.text(.06, .095, spec['explanation'], fontsize=10.5, linespacing=1.5)
        fig.text(.06, .025, 'TP/TN: correct positive/negative. FP/FN: incorrect positive/negative. '
                 'Definitions match src/metrics.py.', fontsize=9, color='#60656b')
        fig.savefig(OUT / (spec['name'] + '.png'), dpi=180)
        pdf.savefig(fig)
        plt.close(fig)

print('Built four illustrative PNGs and fairness_metric_guide.pdf under reports/fairness/')
