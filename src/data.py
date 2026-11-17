"""Reproducible ACS preparation; learned preprocessing uses public reference only."""
from pathlib import Path
import json
import platform
from importlib.metadata import version

import joblib
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from folktables import ACSDataSource

ROOT = Path(__file__).resolve().parents[1]
FEATURES = ['AGEP', 'SCHL', 'MAR', 'SEX', 'DIS', 'ESP', 'MIG', 'CIT',
            'MIL', 'ANC', 'NATIVITY', 'RELP', 'DEAR', 'DEYE', 'DREM',
            'RAC1P', 'GCL']
PROTECTED = ['SEX', 'RAC1P']
DROPPED = ['ESP']  # User decision: structurally missing for 98.58% of eligible people.
INPUTS = [c for c in FEATURES if c not in PROTECTED + DROPPED]
CATEGORICAL = [c for c in INPUTS if c != 'AGEP']
SPLITS = [('reference', .10), ('train', .50), ('development', .10),
          ('calibration', .10), ('test', .20)]


def model_frame(frame):
    """Use string categories so structural N/A is not treated as a magnitude."""
    result = frame[INPUTS].copy()
    for column in CATEGORICAL:
        result[column] = (pd.to_numeric(result[column], errors='raise')
                          .astype('Int64').astype('string').fillna('NOT_APPLICABLE'))
    return result


def group_summary(frame, groups):
    summary = (frame.groupby(groups, dropna=False, observed=True)['Y']
            .agg(n='size', positives='sum', positive_rate='mean')
            .assign(negatives=lambda x: x.n - x.positives))
    levels = {'SEX': [1, 2], 'RAC1P': list(range(1, 10)),
              'split': sorted(frame['split'].unique())}
    index = (pd.Index(levels[groups[0]], name=groups[0]) if len(groups) == 1
             else pd.MultiIndex.from_product([levels[g] for g in groups], names=groups))
    summary = summary.reindex(index)
    for column in ['n', 'positives', 'negatives']:
        summary[column] = summary[column].fillna(0).astype(int)
    return summary.reset_index()


def prepare(seed=285):
    raw_dir = ROOT / 'data' / 'raw'
    out = ROOT / 'data' / 'processed'
    audit = ROOT / 'results' / 'data_audit'
    for folder in (raw_dir, out, audit):
        folder.mkdir(parents=True, exist_ok=True)
    print('Loading California 2018 ACS person data (cached after first download)...', flush=True)
    raw = ACSDataSource(survey_year='2018', horizon='1-Year', survey='person',
                        root_dir=str(raw_dir)).get_data(states=['CA'], download=True)
    missing = sorted(set(FEATURES + ['ESR', 'PWGTP', 'SERIALNO', 'SPORDER']) - set(raw))
    if missing:
        raise ValueError(f'Missing required columns: {missing}')
    assert not raw.duplicated(['SERIALNO', 'SPORDER']).any(), 'Duplicate person identifiers'
    eligibility = raw.AGEP.gt(16) & raw.AGEP.lt(90) & raw.PWGTP.ge(1)
    cohort = raw.loc[eligibility, FEATURES + ['ESR', 'PWGTP']].copy()
    assert cohort.ESR.isin([1, 2, 3, 4, 5, 6]).all(), 'Unexpected/missing target code'
    assert cohort.SEX.isin([1, 2]).all()
    assert cohort.RAC1P.isin(range(1, 10)).all()
    assert cohort.AGEP.notna().all()
    cohort.insert(0, 'row_id', cohort.index.to_numpy())
    cohort['Y'] = cohort.ESR.eq(1).astype('int8')
    # Assignment is independent of labels/groups; identical saved rows across methods.
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(cohort))
    assignment = np.empty(len(cohort), dtype=object)
    start = 0
    for index, (name, share) in enumerate(SPLITS):
        stop = len(cohort) if index == len(SPLITS) - 1 else start + int(len(cohort) * share)
        assignment[order[start:stop]] = name
        start = stop
    cohort['split'] = assignment
    assert cohort.row_id.is_unique and cohort.split.notna().all()

    pd.DataFrame({'feature': FEATURES,
                  'raw_missing': raw[FEATURES].isna().sum().to_numpy(),
                  'cohort_missing': cohort[FEATURES].isna().sum().to_numpy(),
                  'cohort_missing_fraction': cohort[FEATURES].isna().mean().to_numpy()
                  }).to_csv(audit / 'missingness.csv', index=False)
    # Descriptive audit covers reference/train/development/calibration only.
    # Test labels remain saved locally for final evaluation, but are not summarized.
    visible = cohort.loc[cohort.split.ne('test')]
    for name, columns in [('sex', ['SEX']), ('race', ['RAC1P']),
                          ('race_sex', ['RAC1P', 'SEX']),
                          ('split_groups', ['split', 'RAC1P', 'SEX'])]:
        group_summary(visible, columns).to_csv(audit / f'{name}.csv', index=False)
    visible.groupby(['split', 'ESR']).size().rename('n').reset_index().to_csv(
        audit / 'target_codes.csv', index=False)

    X = model_frame(cohort)
    transformer = ColumnTransformer([
        ('age', StandardScaler(), ['AGEP']),
        ('categories', OneHotEncoder(handle_unknown='ignore', dtype=np.float32), CATEGORICAL)
    ], sparse_threshold=1.0)
    transformer.fit(X.loc[cohort.split.eq('reference')])
    encoded = sparse.csr_matrix(transformer.transform(X), dtype=np.float32)
    assert np.isfinite(encoded.data).all()
    assert not set(PROTECTED + ['ESR', 'Y', 'PWGTP', 'row_id', 'split']) & set(INPUTS)
    feature_names = transformer.get_feature_names_out().tolist()
    for name, _ in SPLITS:
        mask = cohort.split.eq(name).to_numpy()
        assert cohort.loc[mask, 'Y'].nunique() == 2, f'Only one label in {name}'
        sparse.save_npz(out / f'X_{name}.npz', encoded[mask])
        cohort.loc[mask, ['row_id', 'Y', 'SEX', 'RAC1P']].to_csv(
            out / f'metadata_{name}.csv', index=False)
    cohort.drop(columns=DROPPED).to_csv(out / 'cohort.csv', index=False)
    joblib.dump(transformer, out / 'preprocessor.joblib')
    manifest = {
        'state': 'CA', 'year': '2018', 'horizon': '1-Year', 'survey': 'person',
        'task': 'explicit ACSEmploymentFiltered definition',
        'filters': ['AGEP > 16', 'AGEP < 90', 'PWGTP >= 1'],
        'target': 'Y = 1 if ESR == 1, else 0; ESR must be valid 1..6',
        'raw_rows': len(raw), 'eligible_rows': len(cohort),
        'excluded_rows': int((~eligibility).sum()), 'duplicate_person_ids': 0,
        'seed': seed, 'split_counts': cohort.split.value_counts().to_dict(),
        'source_features': FEATURES, 'model_inputs': INPUTS,
        'dropped_features': DROPPED,
        'drop_reason': 'ESP removed by user decision; retain all eligible people and missingness audit',
        'encoded_features': feature_names, 'encoded_dimension': len(feature_names),
        'preprocessor_fitted_on': 'reference', 'weighted_training_or_metrics': False,
        'privacy_scope': 'public auxiliary sets; only model-training records designated private',
        'python': platform.python_version(),
        'versions': {name: version(name) for name in ['folktables', 'numpy', 'pandas', 'scipy', 'scikit-learn']}
    }
    (out / 'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    print(json.dumps({k: manifest[k] for k in ['raw_rows', 'eligible_rows', 'excluded_rows',
                                             'split_counts', 'encoded_dimension']}, indent=2), flush=True)
    return manifest
