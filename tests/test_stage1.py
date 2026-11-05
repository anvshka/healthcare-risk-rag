"""Small isolated test fixtures only; never project/demo patient datasets."""
from pathlib import Path
import pandas as pd
from pandas.testing import assert_frame_equal
import pytest
from src.data_loading import TABLE_KEYS, load_tables
from src.preprocessing import DATETIME_COLUMNS, main, preprocess_tables


def test_missing_files_actionable_and_no_outputs(tmp_path, capsys):
    output = tmp_path / 'processed'
    report = tmp_path / 'report.md'
    assert main(['--raw-dir', str(tmp_path / 'raw'), '--output-dir', str(output),
                 '--report', str(report)]) == 1
    message = capsys.readouterr().err
    assert 'PATIENTS.csv' in message and 'Place your authorized' in message
    assert not output.exists() and not report.exists()


def write_headers(path: Path) -> None:
    """Write schema-only files, containing no patient records."""
    for table, keys in TABLE_KEYS.items():
        pd.DataFrame(columns=[key.upper() for key in keys]).to_csv(
            path / f'{table.upper()}.csv', index=False)


def test_loader_case_gzip_and_leading_zero_codes(tmp_path):
    write_headers(tmp_path)
    path = tmp_path / 'DIAGNOSES_ICD.csv'
    path.unlink()
    pd.DataFrame({' SUBJECT_ID ': ['001'], 'HADM_ID': ['002'], 'ICD9_CODE': ['0010']}).to_csv(
        tmp_path / 'diagnoses_icd.csv.gz', index=False)
    result = load_tables(tmp_path)
    assert len(result) == 7
    row = result['diagnoses_icd'].iloc[0]
    assert row['subject_id'] == '001' and row['icd9_code'] == '0010'
    assert all(c == c.lower() for frame in result.values() for c in frame)


def test_loader_actual_headers_in_error(tmp_path):
    write_headers(tmp_path)
    (tmp_path / 'PATIENTS.csv').write_text('unexpected\n', encoding='utf-8')
    with pytest.raises(ValueError, match='actual headers'):
        load_tables(tmp_path)


def test_loader_ambiguous_file(tmp_path):
    write_headers(tmp_path)
    pd.DataFrame(columns=['subject_id']).to_csv(tmp_path / 'patients.csv.gz', index=False)
    with pytest.raises(ValueError, match='Multiple files'):
        load_tables(tmp_path)


def test_duplicate_normalized_headers(tmp_path):
    write_headers(tmp_path)
    (tmp_path / 'PATIENTS.csv').write_text('SUBJECT_ID,subject_id\n', encoding='utf-8')
    with pytest.raises(ValueError, match='duplicate column'):
        load_tables(tmp_path)


@pytest.mark.parametrize('column', sorted(DATETIME_COLUMNS))
def test_dates_parsed(column):
    cleaned, _ = preprocess_tables({'example': pd.DataFrame({
        column: ['2100-01-01', '2101-01-01']})})
    assert pd.api.types.is_datetime64_any_dtype(cleaned['example'][column])


def test_invalid_datetime_rejected():
    with pytest.raises(ValueError, match='admissions.admittime'):
        preprocess_tables({'admissions': pd.DataFrame({'admittime': ['invalid']})})


def test_age_calendar_cap_missing_and_negative():
    patients = pd.DataFrame({'subject_id': list('abcde'),
        'dob': ['2000-06-02', '1800-01-01', None, '2200-01-01', '2000-06-01']})
    admissions = pd.DataFrame({'subject_id': list('abcde'), 'hadm_id': list('vwxyz'),
        'admittime': ['2050-06-01', '2101-01-01', '2101-01-01', '2101-01-01', '2050-06-01'],
        'hospital_expire_flag': [0] * 5})
    cleaned, _ = preprocess_tables({'patients': patients, 'admissions': admissions})
    age = cleaned['admissions']['age']
    assert age.iloc[0] == 49 and age.iloc[1] == 90 and age.iloc[4] == 50
    assert pd.isna(age.iloc[2]) and pd.isna(age.iloc[3])
    assert 'hospital_expire_flag' in cleaned['admissions']


def test_conflicting_birth_dates_rejected():
    with pytest.raises(ValueError, match='conflicting birth dates'):
        preprocess_tables({
            'patients': pd.DataFrame({'subject_id': ['a', 'a'], 'dob': ['2000-01-01', '2001-01-01']}),
            'admissions': pd.DataFrame({'subject_id': ['a'], 'admittime': ['2100-01-01']})})


def test_text_missing_threshold_constants_and_duplicates():
    frame = pd.DataFrame({'subject_id': ['a', 'b', 'c', 'd', 'e', 'e'],
        'gender': [' f ', 'm', None, ' f', 'm ', 'm '],
        'sparse': [None, None, None, None, 'x', 'x'],
        'exactly_60': ['x', 'y', None, None, None, None],
        'constant': ['same'] * 6})
    original = frame.copy(deep=True)
    cleaned, report = preprocess_tables({'patients': frame})
    result = cleaned['patients']
    assert len(result) == 5
    assert result['gender'].tolist() == ['F', 'M', 'UNKNOWN', 'F', 'M']
    assert 'sparse' not in result and 'constant' not in result
    assert 'exactly_60' in result
    assert '| patients | 6 | 5 |' in report
    assert_frame_equal(frame, original)


def test_item_medians_no_cross_item_fallback_and_protected_values():
    labs = pd.DataFrame({'subject_id': list('abcdefg'), 'hadm_id': list('abcdefg'),
        'itemid': ['x', 'x', 'x', 'y', 'y', 'z', None],
        'valuenum': [1.0, 9.0, None, 100.0, None, None, None]})
    cleaned, _ = preprocess_tables({'labevents': labs})
    result = cleaned['labevents']
    assert result['valuenum'].iloc[2] == 5.0
    assert result['valuenum'].iloc[4] == 100.0
    assert pd.isna(result['valuenum'].iloc[5]) and pd.isna(result['valuenum'].iloc[6])
    assert pd.isna(result['itemid'].iloc[6])
    empty = pd.DataFrame({'itemid': ['x', 'y'], 'valuenum': [None, None]})
    cleaned, _ = preprocess_tables({'labevents': empty})
    assert 'valuenum' in cleaned['labevents']
    assert cleaned['labevents']['valuenum'].isna().all()


def test_nonnumeric_lab_value_rejected():
    with pytest.raises(ValueError, match='non-numeric'):
        preprocess_tables({'labevents': pd.DataFrame({'itemid': ['x'], 'valuenum': ['bad']})})


def test_idempotent():
    tables = {
        'patients': pd.DataFrame({'subject_id': ['a', 'b', 'c'],
            'dob': ['2000-01-01', '2001-02-01', None], 'gender': ['f', 'm', None]}),
        'admissions': pd.DataFrame({'subject_id': ['a', 'b', 'c'], 'hadm_id': ['x', 'y', 'z'],
            'admittime': ['2050-01-01', '2050-01-01', '2050-01-01'], 'hospital_expire_flag': [0, 0, 0]}),
        'labevents': pd.DataFrame({'itemid': ['x', 'x', 'y'], 'valuenum': [1.0, None, None]})}
    first, _ = preprocess_tables(tables)
    second, _ = preprocess_tables(first)
    for name in first:
        assert_frame_equal(first[name], second[name])


def test_cli_schema_only_success_report_and_outputs(tmp_path, capsys):
    raw = tmp_path / 'raw'
    raw.mkdir()
    write_headers(raw)
    out, report = tmp_path / 'out', tmp_path / 'report.md'
    assert main(['--raw-dir', str(raw), '--output-dir', str(out), '--report', str(report)]) == 0
    assert len(list(out.glob('*.csv'))) == 7
    assert '# Stage 1 preprocessing report' in report.read_text(encoding='utf-8')
    assert '| patients | 0 | 0 |' in capsys.readouterr().out

def test_exact_duplicates_do_not_weight_lab_median():
    labs = pd.DataFrame({'row_id': ['a', 'a', 'b', 'c', 'd'],
                         'itemid': ['x'] * 5, 'valuenum': [1.0, 1.0, 5.0, 9.0, None]})
    cleaned, _ = preprocess_tables({'labevents': labs})
    result = cleaned['labevents']
    assert len(result) == 4
    assert result.loc[result['row_id'] == 'D', 'valuenum'].iloc[0] == 5.0


def test_idempotence_after_column_pruning_collapses_rows():
    # Removing constants may collapse records; pruning must reach a fixed point.
    frame = pd.DataFrame({'subject_id': ['a'] * 5,
        'x': ['a', 'a', 'b', 'b', 'c'], 'discard': [None] * 5,
        'admittime': ['2100-01-01', '2100-01-01', None, None, None]})
    first, _ = preprocess_tables({'patients': frame})
    second, _ = preprocess_tables(first)
    assert_frame_equal(first['patients'], second['patients'])


def test_empty_dated_tables_do_not_crash():
    tables = {
        'patients': pd.DataFrame(columns=['subject_id', 'dob']),
        'admissions': pd.DataFrame(columns=['subject_id', 'hadm_id', 'admittime', 'hospital_expire_flag'])}
    cleaned, report = preprocess_tables(tables)
    assert 'age' in cleaned['admissions']
    assert '0.00' in report
