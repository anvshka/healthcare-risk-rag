"""Unit-test fixtures only; no substitute patient dataset is generated."""
import copy
import json
import pandas as pd
from pandas.testing import assert_frame_equal
import pytest
from src.features import AdmissionFeatureBuilder, BASE_NUMERIC, CATEGORICAL, main, patient_split, prepare_tables


@pytest.fixture
def tables():
    """Minimal values chosen to expose grouping, scaling and leakage errors."""
    return {
        'patients': pd.DataFrame({'subject_id': ['a', 'b', 'c'],
            'dob': ['2000-01-01'] * 3, 'gender': ['f', 'm', None],
            'dod': [None] * 3, 'expire_flag': [1, 1, 1]}),
        'admissions': pd.DataFrame({'subject_id': ['a', 'a', 'b', 'c'],
            'hadm_id': ['a1', 'a2', 'b1', 'c1'],
            'admittime': ['2050-01-01'] * 4,
            'dischtime': ['2050-01-03', '2050-01-05', '2050-01-07', '2050-04-11'],
            'admission_type': ['emergency', 'urgent', 'elective', 'emergency'],
            'insurance': ['x', 'y', None, 'z'], 'hospital_expire_flag': [0, 1, 0, 1],
            'discharge_location': ['HOME', 'DEAD/EXPIRED', 'HOME', 'DEAD/EXPIRED']}),
        'icustays': pd.DataFrame({'subject_id': ['a', 'a'], 'hadm_id': ['a1', 'a1'],
            'icustay_id': ['i1', 'i2']}),
        'diagnoses_icd': pd.DataFrame({'row_id': ['d1', 'd2', 'd3'],
            'subject_id': ['a', 'a', 'b'], 'hadm_id': ['a1', 'a1', 'b1']}),
        'prescriptions': pd.DataFrame({'row_id': ['p1', 'p2'],
            'subject_id': ['a', 'c'], 'hadm_id': ['a2', 'c1']}),
        'labevents': pd.DataFrame({'row_id': ['l1', 'l2', 'l3', 'l4', 'l5', 'l6', 'l7'],
            'subject_id': ['a', 'a', 'a', 'b', 'c', 'c', 'c'],
            'hadm_id': ['a1', 'a1', 'a2', 'b1', 'c1', 'c1', 'c1'],
            'itemid': ['x', 'x', 'x', 'x', 'x', 'y', 'y'],
            'valuenum': [2.0, 6.0, None, 10.0, 1000.0, 9999.0, 9999.0]}),
        'd_labitems': pd.DataFrame({'itemid': ['x', 'y'], 'label': ['Test lab', 'Holdout lab']})}


def test_one_row_per_admission_counts_and_lab_aggregates(tables):
    builder = AdmissionFeatureBuilder().fit(tables, ['a', 'b'])
    frame = builder.transform(tables, standardize=False)
    assert len(frame) == 4
    assert frame['num_icu_stays'].tolist() == [2, 0, 0, 0]
    assert frame['num_diagnoses'].tolist() == [2, 0, 1, 0]
    assert frame['num_prescriptions'].tolist() == [0, 1, 0, 1]
    assert frame['length_of_stay_days'].tolist() == [2.0, 4.0, 6.0, 100.0]
    assert frame.loc[0, 'lab_x_test_lab_mean'] == 4
    assert frame.loc[0, 'lab_x_test_lab_min'] == 2
    assert frame.loc[0, 'lab_x_test_lab_max'] == 6
    assert frame.loc[1, 'lab_x_test_lab_mean'] == 6  # training item median
    assert frame.loc[2, 'insurance'] == 'UNKNOWN'
    assert frame.loc[3, 'gender'] == 'UNKNOWN'


def test_train_only_selection_imputation_and_population_scaling(tables):
    builder = AdmissionFeatureBuilder().fit(tables, ['a', 'b'])
    assert builder.lab_items_ == ['X']
    assert builder.lab_medians_ == {'X': 6.0}
    assert builder.means_['length_of_stay_days'] == 4.0
    assert builder.scales_['length_of_stay_days'] == pytest.approx((8/3)**0.5)
    output = builder.transform(tables)
    assert output.loc[:2, builder.numeric_columns_].mean().abs().max() < 1e-12
    for col in builder.numeric_columns_:
        if output.loc[:2, col].std(ddof=0) > 0:
            assert output.loc[:2, col].std(ddof=0) == pytest.approx(1.0)
    assert output.loc[3, 'length_of_stay_days'] > 50
    assert output[builder.numeric_columns_].notna().all().all()
    assert output['age'].eq(0).all()  # zero-variance scale is 1


def test_holdout_changes_do_not_change_fitted_state_or_training_rows(tables):
    builder = AdmissionFeatureBuilder().fit(tables, ['a', 'b'])
    changed = copy.deepcopy(tables)
    changed['labevents'].loc[changed['labevents']['subject_id'] == 'c', 'valuenum'] = -1e9
    changed['admissions'].loc[3, 'dischtime'] = '2150-01-01'
    other = AdmissionFeatureBuilder().fit(changed, ['a', 'b'])
    assert builder.state() == other.state()
    assert_frame_equal(builder.transform(tables).iloc[:3], other.transform(changed).iloc[:3])


def test_no_target_identifiers_or_death_columns_in_predictors(tables):
    builder = AdmissionFeatureBuilder().fit(tables, ['a', 'b'])
    predictors = builder.transform(tables)
    forbidden = {'subject_id', 'hadm_id', 'hospital_expire_flag', 'expire_flag',
                 'dod', 'deathtime', 'discharge_location'}
    assert not forbidden.intersection(predictors.columns)
    assert set(predictors) == set(BASE_NUMERIC + CATEGORICAL +
        ['lab_x_test_lab_mean', 'lab_x_test_lab_min', 'lab_x_test_lab_max'])
    changed = copy.deepcopy(tables)
    changed['admissions']['hospital_expire_flag'] = [1, 0, 1, 0]
    changed['admissions']['discharge_location'] = 'DEAD/EXPIRED'
    assert_frame_equal(predictors, builder.transform(changed))


def test_split_is_reproducible_and_patient_disjoint(tables):
    ids = tables['admissions']['subject_id']
    train, test = patient_split(ids)
    assert (train, test) == patient_split(ids.sample(frac=1, random_state=5))
    assert not set(train) & set(test)
    assert set(train) | set(test) == {'a', 'b', 'c'}
    assert len(set(ids[ids.isin(train)]) & set(ids[ids.isin(test)])) == 0


@pytest.mark.parametrize('size', [0, 1, -1, 2])
def test_invalid_split(size):
    with pytest.raises(ValueError):
        patient_split(pd.Series(['a', 'b']), size)


def test_one_patient_split_rejected():
    with pytest.raises(ValueError, match='At least two'):
        patient_split(pd.Series(['a', 'a']))


def test_state_roundtrip_and_no_mutation(tables):
    original = copy.deepcopy(tables)
    builder = AdmissionFeatureBuilder().fit(tables, ['a', 'b'])
    state = json.loads(json.dumps(builder.state(), allow_nan=False))
    restored = AdmissionFeatureBuilder.from_state(state)
    assert_frame_equal(builder.transform(tables), restored.transform(tables))
    for name in tables:
        assert_frame_equal(tables[name], original[name])


def test_no_numeric_labs_and_absent_events(tables):
    tables['labevents']['valuenum'] = float('nan')
    tables['icustays'] = tables['icustays'].iloc[:0]
    builder = AdmissionFeatureBuilder().fit(tables, ['a', 'b'])
    assert builder.lab_items_ == []
    assert builder.transform(tables).shape == (4, 8)
    assert builder.transform(tables, standardize=False)['num_icu_stays'].eq(0).all()


def test_missing_admission_lab_uses_training_aggregate_median(tables):
    tables['labevents'] = tables['labevents'].query("hadm_id != 'a2'").copy()
    builder = AdmissionFeatureBuilder().fit(tables, ['a', 'b'])
    assert pd.isna(builder.transform(tables, standardize=False).loc[1, 'lab_x_test_lab_mean'])
    assert builder.numeric_medians_['lab_x_test_lab_mean'] == 7
    assert pd.notna(builder.transform(tables).loc[1, 'lab_x_test_lab_mean'])


def test_event_ownership_conflict_rejected(tables):
    tables['labevents'].loc[0, 'subject_id'] = 'c'
    with pytest.raises(ValueError, match='ownership mismatch'):
        prepare_tables(tables)


def test_unassigned_lab_not_joined_by_subject_alone(tables):
    tables['labevents'].loc[0, 'hadm_id'] = None
    result = AdmissionFeatureBuilder().fit(tables, ['a', 'b']).transform(tables, standardize=False)
    assert result.loc[0, 'lab_x_test_lab_mean'] == 6


def test_duplicate_admission_rejected(tables):
    tables['admissions'].loc[1, 'hadm_id'] = 'a1'
    with pytest.raises(ValueError, match='nonmissing and unique'):
        prepare_tables(tables)


def test_missing_required_header_rejected(tables):
    tables['patients'] = tables['patients'].drop(columns='gender')
    with pytest.raises(ValueError, match='actual headers'):
        prepare_tables(tables)


def test_negative_los_rejected(tables):
    tables['admissions'].loc[0, 'dischtime'] = '2049-01-01'
    with pytest.raises(ValueError, match='dischtime before'):
        AdmissionFeatureBuilder().fit(tables, ['a', 'b'])


def test_invalid_labels_rejected(tables):
    tables['admissions'].loc[0, 'hospital_expire_flag'] = 2
    with pytest.raises(ValueError, match='only 0 and 1'):
        prepare_tables(tables)


def test_unfitted_transform_rejected(tables):
    with pytest.raises(ValueError, match='Call fit'):
        AdmissionFeatureBuilder().transform(tables)


def test_empty_train_ids_rejected(tables):
    with pytest.raises(ValueError, match='nonempty subset'):
        AdmissionFeatureBuilder().fit(tables, [])


def test_deterministic_lab_frequency_tie_break(tables):
    tables['labevents'].loc[1, 'itemid'] = 'y'
    tables['labevents'].loc[2, 'itemid'] = 'y'
    builder = AdmissionFeatureBuilder(top_n_labs=1).fit(tables, ['a', 'b'])
    assert builder.lab_items_ == ['X']


def test_cli_missing_input_no_outputs(tmp_path, capsys):
    output = tmp_path / 'out'
    assert main(['--raw-dir', str(tmp_path / 'missing'), '--output-dir', str(output)]) == 1
    assert 'Missing MIMIC-III' in capsys.readouterr().err
    assert not output.exists()


def test_cli_success_and_artifact_alignment(tables, tmp_path, capsys):
    raw = tmp_path / 'raw'
    raw.mkdir()
    for name, frame in tables.items():
        frame.to_csv(raw / f'{name}.csv', index=False)
    output, report = tmp_path / 'out', tmp_path / 'report.md'
    assert main(['--raw-dir', str(raw), '--output-dir', str(output), '--report', str(report)]) == 0
    index = pd.read_csv(output / 'admission_index.csv')
    unscaled = pd.read_csv(output / 'admission_features.csv')
    assert index['hadm_id'].is_unique and len(index) == 4
    assert unscaled['hadm_id'].tolist() == index['hadm_id'].tolist()
    train = pd.read_csv(output / 'train_index.csv')
    test = pd.read_csv(output / 'test_index.csv')
    assert not set(train.subject_id) & set(test.subject_id)
    assert len(train) == len(pd.read_csv(output / 'train_features.csv'))
    assert len(test) == len(pd.read_csv(output / 'test_features.csv'))
    assert 'retrospective' in report.read_text(encoding='utf-8')
    assert 'Patient overlap: 0' in capsys.readouterr().out

def test_numeric_and_string_identifier_inputs_match(tables):
    for frame in tables.values():
        if 'subject_id' in frame:
            frame['subject_id'] = frame['subject_id'].map({'a': 1, 'b': 2, 'c': 3})
    tables['patients']['subject_id'] = tables['patients']['subject_id'].astype(str)
    builder = AdmissionFeatureBuilder().fit(tables, ['1', '2'])
    output = builder.transform(tables, standardize=False)
    assert output['age'].eq(50).all()
    assert output['num_icu_stays'].tolist() == [2, 0, 0, 0]


def test_infinite_lab_values_are_not_fitted_as_observations(tables):
    tables['labevents'].loc[0, 'valuenum'] = float('inf')
    builder = AdmissionFeatureBuilder().fit(tables, ['a', 'b'])
    assert builder.lab_medians_['X'] == 8.0
    state = json.dumps(builder.state(), allow_nan=False)
    assert 'Infinity' not in state


def test_transform_does_not_modify_fitted_state(tables):
    builder = AdmissionFeatureBuilder().fit(tables, ['a', 'b'])
    before = json.dumps(builder.state(), sort_keys=True)
    builder.transform(tables)
    assert json.dumps(builder.state(), sort_keys=True) == before
