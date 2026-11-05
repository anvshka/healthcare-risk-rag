"""Stage 2: retrospective admission features with training-only fitted state."""
import argparse
import json
import math
from pathlib import Path
import random
import re
import sys
import pandas as pd
from .data_loading import load_tables
from .preprocessing import preprocess_tables

CATEGORICAL = ['gender', 'admission_type', 'insurance']
BASE_NUMERIC = ['age', 'length_of_stay_days', 'num_icu_stays',
                'num_diagnoses', 'num_prescriptions']
REQUIRED = {
    'patients': {'subject_id', 'dob', 'gender'},
    'admissions': {'subject_id', 'hadm_id', 'admittime', 'dischtime',
                   'admission_type', 'insurance', 'hospital_expire_flag'},
    'icustays': {'subject_id', 'hadm_id', 'icustay_id'},
    'diagnoses_icd': {'subject_id', 'hadm_id'},
    'prescriptions': {'subject_id', 'hadm_id'},
    'labevents': {'subject_id', 'hadm_id', 'itemid', 'valuenum'},
    'd_labitems': {'itemid', 'label'},
}


def prepare_tables(raw: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    """Validate and clean raw tables without full-dataset numeric imputation."""
    for name, columns in REQUIRED.items():
        if name not in raw:
            raise ValueError(f'Missing table: {name}')
        actual = {str(c).strip().lower() for c in raw[name]}
        if columns - actual:
            raise ValueError(f'{name}: missing {sorted(columns - actual)}; actual headers: {sorted(actual)}')
    normalized = {}
    for name, original in raw.items():
        frame = original.copy(deep=True)
        frame.columns = frame.columns.str.strip().str.lower()
        for column in frame:
            if column.endswith('_id') or column == 'itemid':
                frame[column] = frame[column].astype('string').str.strip().str.upper().replace('', pd.NA)
        normalized[name] = frame
    tables, _ = preprocess_tables(normalized, impute_labs=False, prune_columns=False)
    for name, key in [('patients', 'subject_id'), ('admissions', 'hadm_id'), ('d_labitems', 'itemid')]:
        if tables[name][key].isna().any() or tables[name][key].duplicated().any():
            raise ValueError(f'{name}.{key} must be nonmissing and unique')
    admissions = tables['admissions']
    if admissions['subject_id'].isna().any():
        raise ValueError('admissions.subject_id must not be missing')
    if not admissions['subject_id'].isin(tables['patients']['subject_id']).all():
        raise ValueError('An admission references a missing patient')
    label = pd.to_numeric(admissions['hospital_expire_flag'], errors='raise')
    if label.isna().any() or not label.isin([0, 1]).all():
        raise ValueError('hospital_expire_flag must contain only 0 and 1')
    admissions['hospital_expire_flag'] = label.astype(int)
    owner = admissions.set_index('hadm_id')['subject_id']
    for name in ['icustays', 'diagnoses_icd', 'prescriptions', 'labevents']:
        events = tables[name]
        # Unassigned/outside-cohort events cannot be attributed to an admission.
        valid = events['hadm_id'].isin(owner.index)
        subset = events.loc[valid].copy()
        expected = subset['hadm_id'].map(owner)
        if subset['subject_id'].isna().any() or not subset['subject_id'].eq(expected).all():
            raise ValueError(f'{name}: subject_id/hadm_id ownership mismatch')
        tables[name] = subset
    if tables['icustays']['icustay_id'].isna().any():
        raise ValueError('icustays.icustay_id must not be missing')
    if tables['icustays']['icustay_id'].duplicated().any():
        raise ValueError('icustays.icustay_id must be unique')
    return tables


def patient_split(subject_ids: pd.Series, test_size: float = 0.2,
                  random_state: int = 42) -> tuple[list[str], list[str]]:
    """Create a deterministic patient-disjoint Stage 2 development holdout.

    Stage 3 will use GroupShuffleSplit and fit a new builder within each fold.
    No labels or event measurements influence this provisional split.
    """
    if not 0 < test_size < 1:
        raise ValueError('test_size must be between 0 and 1')
    if subject_ids.isna().any():
        raise ValueError('Cannot split missing subject IDs')
    subjects = sorted(set(subject_ids.astype(str)))
    if len(subjects) < 2:
        raise ValueError('At least two patients are needed for a train/test split')
    random.Random(random_state).shuffle(subjects)
    n_test = min(len(subjects) - 1, max(1, math.ceil(len(subjects) * test_size)))
    return sorted(subjects[n_test:]), sorted(subjects[:n_test])


class AdmissionFeatureBuilder:
    """Fit top labs, medians and population z-scores using training patients only.

    Identifiers and outcomes are metadata, never predictors. Categorical values
    remain strings for train-only encoding in Stage 3. Full-stay features make
    this a retrospective benchmark, not a prospective admission-time score.
    """

    def __init__(self, top_n_labs: int = 10) -> None:
        if top_n_labs < 1:
            raise ValueError('top_n_labs must be positive')
        self.top_n_labs = top_n_labs
        self.fitted_ = False

    def _require_fit(self) -> None:
        if not self.fitted_:
            raise ValueError('Call fit with training subject IDs before transform')

    def fit(self, raw: dict[str, pd.DataFrame], train_subject_ids: list[str]) -> 'AdmissionFeatureBuilder':
        """Learn state from designated training patients, rejecting invalid IDs."""
        self.fitted_ = False
        tables = prepare_tables(raw)
        admissions = tables['admissions']
        subjects = {str(v).strip().upper() for v in train_subject_ids}
        available = set(admissions['subject_id'].astype(str))
        if not subjects or not subjects <= available:
            raise ValueError('Training IDs must be a nonempty subset of admission patient IDs')
        self.train_subject_ids_ = sorted(subjects)
        train = admissions['subject_id'].isin(subjects)
        train_hadm = set(admissions.loc[train, 'hadm_id'])
        labs = tables['labevents']
        labs = labs[labs['hadm_id'].isin(train_hadm)].copy()
        labs['valuenum'] = labs['valuenum'].replace([float('inf'), -float('inf')], float('nan'))
        numeric_items = set(labs.loc[labs['valuenum'].notna(), 'itemid'].dropna())
        counts = labs[labs['itemid'].isin(numeric_items)].groupby('itemid').size()
        # Tie-breaking is deterministic and independent of source file ordering.
        ordered = sorted(counts.index, key=lambda item: (-int(counts[item]), str(item)))
        self.lab_items_ = list(ordered[:self.top_n_labs])
        medians = labs.groupby('itemid')['valuenum'].median()
        self.lab_medians_ = {str(item): float(medians[item]) for item in self.lab_items_}
        dictionary = tables['d_labitems'].set_index('itemid')['label']
        self.lab_names_ = {}
        for item in self.lab_items_:
            label = dictionary.get(item, 'UNLABELLED')
            label = 'UNLABELLED' if pd.isna(label) else str(label)
            slug = re.sub(r'[^a-z0-9]+', '_', label.lower()).strip('_') or 'unlabelled'
            item_slug = re.sub(r'[^a-z0-9]+', '_', str(item).lower()).strip('_')
            self.lab_names_[str(item)] = f'lab_{item_slug}_{slug}'
        self.numeric_columns_ = BASE_NUMERIC + [
            f'{self.lab_names_[str(item)]}_{stat}' for item in self.lab_items_
            for stat in ['mean', 'min', 'max']]
        self.feature_columns_ = self.numeric_columns_ + CATEGORICAL
        if len(set(self.feature_columns_)) != len(self.feature_columns_):
            raise ValueError('Lab labels create duplicate feature names')
        features = self._build(tables).loc[train.to_numpy(), self.numeric_columns_]
        self.numeric_medians_ = features.median().fillna(0.0)
        filled = features.fillna(self.numeric_medians_).astype(float)
        self.means_ = filled.mean()
        self.scales_ = filled.std(ddof=0).replace(0.0, 1.0).fillna(1.0)
        self.fitted_ = True
        return self

    def _build(self, tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Aggregate using allowlisted fields and previously selected lab items."""
        adm = tables['admissions'].reset_index(drop=True)
        result = adm[['age', 'admission_type', 'insurance']].copy()
        result['gender'] = adm['subject_id'].map(tables['patients'].set_index('subject_id')['gender'])
        los = (adm['dischtime'] - adm['admittime']).dt.total_seconds() / 86400
        if (los.dropna() < 0).any():
            raise ValueError('An admission has dischtime before admittime')
        result['length_of_stay_days'] = los
        for table, feature in [('icustays', 'num_icu_stays'),
                               ('diagnoses_icd', 'num_diagnoses'),
                               ('prescriptions', 'num_prescriptions')]:
            counts = tables[table].groupby('hadm_id').size()
            result[feature] = adm['hadm_id'].map(counts).fillna(0).astype(int)
        labs = tables['labevents'].copy()
        labs['valuenum'] = labs['valuenum'].replace([float('inf'), -float('inf')], float('nan'))
        labs = labs[labs['itemid'].isin(self.lab_items_)]
        labs['valuenum'] = labs['valuenum'].fillna(labs['itemid'].map(self.lab_medians_))
        grouped = labs.groupby(['hadm_id', 'itemid'])['valuenum'].agg(['mean', 'min', 'max'])
        for item in self.lab_items_:
            if item in grouped.index.get_level_values('itemid'):
                item_stats = grouped.xs(item, level='itemid')
            else:
                item_stats = pd.DataFrame(columns=['mean', 'min', 'max'])
            for stat in ['mean', 'min', 'max']:
                col = f'{self.lab_names_[str(item)]}_{stat}'
                result[col] = adm['hadm_id'].map(item_stats[stat]).astype(float)
        for col in CATEGORICAL:
            result[col] = result[col].astype('string').fillna('UNKNOWN')
        return result[self.feature_columns_]

    def transform(self, raw: dict[str, pd.DataFrame], *, standardize: bool = True) -> pd.DataFrame:
        """Return one predictor row per admission without refitting any state."""
        self._require_fit()
        result = self._build(prepare_tables(raw))
        if standardize:
            numeric = result[self.numeric_columns_].fillna(self.numeric_medians_).astype(float)
            result[self.numeric_columns_] = (numeric - self.means_) / self.scales_
        return result

    def state(self) -> dict:
        """Return JSON-compatible fitted parameters for auditing and reuse."""
        self._require_fit()
        return {'version': 1, 'top_n_labs': self.top_n_labs,
                'train_subject_ids': self.train_subject_ids_, 'lab_items': self.lab_items_,
                'lab_names': self.lab_names_, 'lab_medians': self.lab_medians_,
                'numeric_columns': self.numeric_columns_, 'feature_columns': self.feature_columns_,
                'numeric_medians': self.numeric_medians_.to_dict(),
                'means': self.means_.to_dict(), 'scales': self.scales_.to_dict()}

    @classmethod
    def from_state(cls, state: dict) -> 'AdmissionFeatureBuilder':
        """Restore a trusted JSON state saved by this version of the builder."""
        if state.get('version') != 1:
            raise ValueError('Unsupported feature-state version')
        builder = cls(state['top_n_labs'])
        for key in ['train_subject_ids', 'lab_items', 'lab_names', 'lab_medians',
                    'numeric_columns', 'feature_columns']:
            setattr(builder, f'{key}_', state[key])
        for key in ['numeric_medians', 'means', 'scales']:
            setattr(builder, f'{key}_', pd.Series(state[key], dtype=float))
        builder.fitted_ = True
        return builder


def main(argv: list[str] | None = None) -> int:
    """Build Stage 2 artifacts using raw CSVs and a patient-disjoint holdout."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--raw-dir', type=Path, default=Path('data/raw'))
    parser.add_argument('--output-dir', type=Path, default=Path('data/processed/stage2'))
    parser.add_argument('--report', type=Path, default=Path('reports/feature_report.md'))
    parser.add_argument('--test-size', type=float, default=0.2)
    parser.add_argument('--top-n-labs', type=int, default=10)
    args = parser.parse_args(argv)
    try:
        raw = load_tables(args.raw_dir)
        tables = prepare_tables(raw)
        admissions = tables['admissions'].reset_index(drop=True)
        train_ids, test_ids = patient_split(admissions['subject_id'], args.test_size)
        builder = AdmissionFeatureBuilder(args.top_n_labs).fit(raw, train_ids)
        values = builder.transform(raw, standardize=False)
        standardized = builder.transform(raw)
        metadata = admissions[['subject_id', 'hadm_id', 'hospital_expire_flag']].copy()
        mask = metadata['subject_id'].isin(train_ids)
        metadata['split'] = mask.map({True: 'train', False: 'test'})
        features = pd.concat([metadata[['subject_id', 'hadm_id']], values], axis=1)
        state_json = json.dumps(builder.state(), indent=2, allow_nan=False)
    except (ValueError, FileNotFoundError, pd.errors.EmptyDataError) as exc:
        print(str(exc), file=sys.stderr)
        return 1
    report = ['# Stage 2 feature report', '',
        f'Admissions: {len(admissions)}; unique patients: {metadata.subject_id.nunique()}.',
        f'Train: {int(mask.sum())} admissions / {len(train_ids)} patients.',
        f'Test: {int((~mask).sum())} admissions / {len(test_ids)} patients.',
        f'Patient overlap: {len(set(train_ids) & set(test_ids))}.',
        f'Predictors: {len(builder.feature_columns_)} ({len(builder.numeric_columns_)} numeric, 3 categorical).',
        '', 'Top lab items selected on training admissions only:', '',
        '| itemid | Feature prefix | Training numeric median |', '|---|---|---:|']
    for item in builder.lab_items_:
        report.append(f'| {item} | {builder.lab_names_[str(item)]} | {builder.lab_medians_[str(item)]:.4g} |')
    report += ['', 'Raw inputs are reloaded; Stage 1 full-data lab medians are not used.',
        'Lab frequency counts include all rows for items with at least one numeric training observation.',
        'Unassigned/outside-cohort events are excluded; ownership mismatches fail validation.',
        'Numeric lab medians, admission-level missing-value medians, means and population standard deviations are fitted on training patients only.',
        'Constant numeric scales use 1; a feature with no training observations uses 0 as a documented fallback.',
        'Categorical fields remain strings and use UNKNOWN for missing values; encoding belongs to Stage 3.',
        'Identifiers and hospital_expire_flag are metadata, not predictors; discharge_location and death-related fields are excluded entirely.',
        'This is a retrospective full-stay benchmark: LOS and whole-stay counts/labs are unavailable at admission time. Do not interpret it as early risk prediction.',
        'This provisional deterministic split uses Python random.Random(42). Stage 3 must use GroupShuffleSplit and refit all learned state within each CV fold.', '']
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    features.to_csv(args.output_dir / 'admission_features.csv', index=False)
    metadata.to_csv(args.output_dir / 'admission_index.csv', index=False)
    for split, selected in [('train', mask), ('test', ~mask)]:
        standardized.loc[selected].to_csv(args.output_dir / f'{split}_features.csv', index=False)
        metadata.loc[selected, ['subject_id', 'hadm_id', 'hospital_expire_flag']].to_csv(
            args.output_dir / f'{split}_index.csv', index=False)
    (args.output_dir / 'feature_state.json').write_text(state_json + '\n', encoding='utf-8')
    args.report.write_text('\n'.join(report), encoding='utf-8')
    print('\n'.join(report))
    print(f'Saved Stage 2 artifacts to {args.output_dir}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
