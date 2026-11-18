"""Reference-fitted feature policies using the original saved split assignments."""
from pathlib import Path
import hashlib
import json
from itertools import combinations

import joblib
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .data import ROOT, FEATURES, INPUTS

POLICIES = {
    'include_sensitive': [name for name in FEATURES if name != 'ESP'],
    'exclude_sensitive': list(INPUTS),
}
VISIBLE = ['reference', 'train', 'development']


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def json_write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')


def feature_frame(frame, inputs):
    if set(inputs) & {'ESR', 'ESP', 'Y', 'PWGTP', 'row_id', 'split'}:
        raise ValueError('Target/identifier/excluded-feature leakage')
    result = frame[inputs].copy()
    for column in inputs:
        if column != 'AGEP':
            result[column] = (pd.to_numeric(result[column], errors='raise').astype('Int64')
                              .astype('string').fillna('NOT_APPLICABLE'))
    return result


def prepare_controlled(config):
    output = ROOT / config['output_dir']
    audit = output/'audit'
    audit.mkdir(parents=True, exist_ok=True)
    data = ROOT/'data/processed'
    original = json.loads((data/'manifest.json').read_text(encoding='utf-8'))
    if original['target'] != 'Y = 1 if ESR == 1, else 0; ESR must be valid 1..6':
        raise ValueError('Original target changed')
    if original['filters'] != ['AGEP > 16', 'AGEP < 90', 'PWGTP >= 1']:
        raise ValueError('Original cohort filters changed')
    # Explicitly exclude Y/ESR when reading all-cohort features: no test outcomes
    # or test metadata/matrix files are loaded. Only fixed assignments/IDs are audited.
    inputs = list(POLICIES['include_sensitive'])
    cohort = pd.read_csv(data/'cohort.csv', usecols=['row_id', 'split']+inputs)
    assert cohort.row_id.is_unique
    assignments = cohort[['row_id', 'split']].copy()
    assert assignments.split.value_counts().to_dict() == original['split_counts']
    frame = cohort.loc[cohort.split.isin(VISIBLE)].set_index('row_id')
    split_meta = {name: pd.read_csv(data/f'metadata_{name}.csv') for name in VISIBLE}
    for name, meta in split_meta.items():
        assert meta.row_id.is_unique
        assert frame.loc[meta.row_id, 'split'].eq(name).all()
        for protected in ['SEX', 'RAC1P']:
            assert np.array_equal(frame.loc[meta.row_id, protected], meta[protected])
    ids = {name: set(meta.row_id) for name, meta in split_meta.items()}
    assert all(ids[a].isdisjoint(ids[b]) for a, b in combinations(VISIBLE, 2))
    raw_path = ROOT/'data/raw/2018/1-Year/psam_p06.csv'
    if not raw_path.exists():
        raise FileNotFoundError('Original cached California CSV is needed for household/ESR auditing')
    # Household overlap uses identifiers only, including test assignments, not labels.
    households = pd.read_csv(raw_path, usecols=['SERIALNO', 'SPORDER'], dtype={'SERIALNO': str})
    assert not households.duplicated(['SERIALNO', 'SPORDER']).any()
    assert assignments.row_id.max() < len(households)
    assigned = assignments.copy()
    assigned['household'] = households.loc[assigned.row_id, 'SERIALNO'].to_numpy()
    sets = {name: set(g.household) for name, g in assigned.groupby('split')}
    overlap = []
    for a, b in combinations(sorted(sets), 2):
        common = len(sets[a] & sets[b])
        overlap.append(dict(split_a=a, split_b=b, shared_households=common,
                            households_a=len(sets[a]), households_b=len(sets[b]),
                            fraction_a_shared=common/len(sets[a]), fraction_b_shared=common/len(sets[b])))
    pd.DataFrame(overlap).to_csv(audit/'household_overlap.csv', index=False)
    visible_ids = set(frame.index)
    # skiprows removes non-visible rows before ESR parsing, including every test row.
    raw_esr = pd.read_csv(raw_path, usecols=['ESR'],
                         skiprows=lambda line: line > 0 and (line-1) not in visible_ids)
    ordered_ids = sorted(visible_ids)
    raw_esr.index = ordered_ids
    assert raw_esr.ESR.isin(range(1, 7)).all()
    for name, meta in split_meta.items():
        meta['ESR'] = raw_esr.loc[meta.row_id, 'ESR'].to_numpy().astype(int)
        assert np.array_equal(meta.Y, meta.ESR.eq(1).astype(int))
    split_hashes = {name: digest(meta.row_id.tolist()) for name, meta in split_meta.items()}
    identity = dict(original_manifest_sha256=sha(data/'manifest.json'),
                    fixed_assignments_sha256=digest(assignments.to_dict('list')),
                    split_row_ids_sha256=split_hashes)
    signature = digest(identity)
    cache = output/'cache'
    cache.mkdir(parents=True, exist_ok=True)
    # Raw-data fingerprints stay local, like row-level predictions. Published
    # identity hashes cover fixed partition IDs/public manifest only, not labels.
    json_write(cache/'local_validation.json', dict(cohort=sha(data/'cohort.csv'),
               metadata={name: sha(data/f'metadata_{name}.csv') for name in VISIBLE}))
    feature_audit = {}
    for policy in config['feature_policies']:
        selected = POLICIES[policy]
        categorical = [column for column in selected if column != 'AGEP']
        transformer = ColumnTransformer([
            ('age', StandardScaler(), ['AGEP']),
            ('categories', OneHotEncoder(handle_unknown='ignore', dtype=np.float32), categorical)
        ], sparse_threshold=1.0)
        transformed_frame = feature_frame(frame, selected)
        transformer.fit(transformed_frame.loc[split_meta['reference'].row_id])
        policy_cache = cache/policy
        policy_cache.mkdir(parents=True, exist_ok=True)
        names = transformer.get_feature_names_out().tolist()
        for name, meta in split_meta.items():
            encoded = sparse.csr_matrix(transformer.transform(transformed_frame.loc[meta.row_id]), dtype=np.float32)
            assert np.isfinite(encoded.data).all()
            if policy == 'exclude_sensitive' and name in ['train', 'development']:
                old = sparse.load_npz(data/f'X_{name}.npz')
                assert old.shape == encoded.shape and (old != encoded).nnz == 0, 'V0 encoding changed'
            np.save(policy_cache/f'X_{name}.npy', encoded.toarray())
            meta.to_csv(policy_cache/f'metadata_{name}.csv', index=False)
        joblib.dump(transformer, policy_cache/'preprocessor.joblib')
        record = dict(feature_policy=policy, model_inputs=selected, encoded_dimension=len(names),
                      encoded_features=names, preprocessing_fitted_on='reference',
                      data_identity=identity, data_signature=signature,
                      feature_signature=digest(dict(policy=policy, names=names, identity=identity)),
                      original_split_counts=original['split_counts'], target='ESR == 1')
        json_write(policy_cache/'manifest.json', record)
        feature_audit[policy] = record
    json_write(audit/'feature_policies.json', feature_audit)
    development = split_meta['development']
    esr_counts = []
    for attribute, levels in [('SEX', [1, 2]), ('RAC1P', list(range(1, 10)))]:
        for value in levels:
            for esr in range(1, 7):
                esr_counts.append(dict(attribute=attribute, group=value, ESR=esr,
                                       n=int((development[attribute].eq(value) & development.ESR.eq(esr)).sum())))
    pd.DataFrame(esr_counts).to_csv(audit/'development_esr_distributions.csv', index=False)
    json_write(audit/'audit.json', dict(
        data_identity=identity, data_signature=signature,
        splits_changed=False, test_outcomes_loaded=False, household_split=False,
        public_auxiliary_assumption=True, target_unchanged=True,
        household_overlap_exists=any(row['shared_households'] for row in overlap),
        fixes=['Primary policy adds categorical SEX/RAC1P; original exclusion retained as secondary',
               'Diagnostic Poisson controls use DP expected-batch normalization',
               'Five paired initializations replace single-run comparisons',
               'Existing person splits retained despite household overlap; record-level DP only'],
        limitations=['Household members may span training and auxiliary/test splits',
                     'These are individual-record, not household-level DP bounds',
                     'Public descriptive audits and non-private controls are not DP releases']
    ))
    return feature_audit


def load_policy(config, policy, smoke=False):
    cache = ROOT/config['output_dir']/'cache'/policy
    manifest = json.loads((cache/'manifest.json').read_text(encoding='utf-8'))
    current = json.loads((ROOT/config['output_dir']/'audit/feature_policies.json').read_text(encoding='utf-8'))[policy]
    assert manifest == current
    local = json.loads((cache.parent/'local_validation.json').read_text(encoding='utf-8'))
    assert sha(ROOT/'data/processed/cohort.csv') == local['cohort'], 'Source cohort changed'
    for name, checksum in local['metadata'].items():
        assert sha(ROOT/'data/processed'/f'metadata_{name}.csv') == checksum, 'Split metadata changed'
    if smoke:
        x = np.load(cache/'X_reference.npy')
        metadata = pd.read_csv(cache/'metadata_reference.csv')
        return x[:512], metadata.iloc[:512].copy(), x[512:640], metadata.iloc[512:640].copy(), manifest
    return (np.load(cache/'X_train.npy'), pd.read_csv(cache/'metadata_train.csv'),
            np.load(cache/'X_development.npy'), pd.read_csv(cache/'metadata_development.csv'), manifest)
