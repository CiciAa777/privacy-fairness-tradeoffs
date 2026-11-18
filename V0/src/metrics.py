"""Shared binary evaluation; undefined conditional rates remain NaN."""
import numpy as np
import pandas as pd
from itertools import product


def binary_metrics(y, pred):
    y, pred = np.asarray(y), np.asarray(pred)
    if y.shape != pred.shape or not np.isin(y, [0, 1]).all() or not np.isin(pred, [0, 1]).all():
        raise ValueError('Aligned binary labels/predictions required')
    tp, tn = int(((y == 1) & (pred == 1)).sum()), int(((y == 0) & (pred == 0)).sum())
    fp, fn = int(((y == 0) & (pred == 1)).sum()), int(((y == 1) & (pred == 0)).sum())
    def rate(a, b): return a / b if b else float('nan')
    tpr, fpr = rate(tp, tp+fn), rate(fp, fp+tn)
    return dict(n=len(y), tp=tp, tn=tn, fp=fp, fn=fn,
                accuracy=rate(tp+tn, len(y)), balanced_accuracy=(tpr+1-fpr)/2,
                positive_rate=rate(tp+fn, len(y)), selection_rate=rate(tp+fp, len(y)),
                tpr=tpr, fpr=fpr, ppv=rate(tp, tp+fp), npv=rate(tn, tn+fn),
                min_conditional_count=min(tp+fn, fp+tn, tp+fp, tn+fn))


def evaluate(metadata, scores, threshold=.5):
    scores = np.asarray(scores)
    if scores.shape != (len(metadata),) or not np.isfinite(scores).all():
        raise ValueError('Expected one finite score per metadata row')
    if ((scores < 0) | (scores > 1)).any():
        raise ValueError('Scores must be probabilities')
    y, pred = metadata.Y.to_numpy(), (scores >= threshold).astype(int)
    rows = []
    for attribute, values in [('SEX', [1,2]), ('RAC1P', list(range(1,10)))]:
        for value in values:
            mask = metadata[attribute].eq(value).to_numpy()
            rows.append(dict(attribute=attribute, group=str(value), **binary_metrics(y[mask], pred[mask])))
    for race, sex in product(range(1,10), [1,2]):
        mask = (metadata.RAC1P.eq(race) & metadata.SEX.eq(sex)).to_numpy()
        rows.append(dict(attribute='RAC1P_SEX', group=f'{race}_{sex}', **binary_metrics(y[mask], pred[mask])))
    groups = pd.DataFrame(rows)
    gaps = {}
    for attribute, frame in groups.groupby('attribute'):
        def gap(column):
            v = frame[column]
            return float(v.max()-v.min()) if v.notna().all() else None
        def maximum(a,b): return max(a,b) if a is not None and b is not None else None
        gaps[attribute] = dict(accuracy_gap=gap('accuracy'), independence=gap('selection_rate'),
                               tpr_gap=gap('tpr'), fpr_gap=gap('fpr'),
                               separation=maximum(gap('tpr'), gap('fpr')),
                               ppv_gap=gap('ppv'), npv_gap=gap('npv'),
                               sufficiency=maximum(gap('ppv'), gap('npv')))
    return binary_metrics(y,pred), groups, gaps
