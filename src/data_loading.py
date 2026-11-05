"""Load user-supplied MIMIC-III CSVs without downloading data."""
from pathlib import Path
import pandas as pd

TABLE_KEYS: dict[str, tuple[str, ...]] = {
    'patients': ('subject_id',),
    'admissions': ('subject_id', 'hadm_id', 'hospital_expire_flag'),
    'icustays': ('subject_id', 'hadm_id', 'icustay_id'),
    'diagnoses_icd': ('subject_id', 'hadm_id'),
    'labevents': ('subject_id', 'hadm_id', 'itemid', 'valuenum'),
    'd_labitems': ('itemid',),
    'prescriptions': ('subject_id', 'hadm_id'),
}


def load_tables(raw_dir: Path) -> dict[str, pd.DataFrame]:
    """Read seven CSV/CSV.GZ tables and normalize actual column headers."""
    raw_dir = Path(raw_dir)
    files = list(raw_dir.iterdir()) if raw_dir.is_dir() else []
    paths: dict[str, Path] = {}
    missing = []
    for table in TABLE_KEYS:
        matches = [p for p in files if p.is_file() and p.name.lower()
                   in (f'{table}.csv', f'{table}.csv.gz')]
        if len(matches) > 1:
            raise ValueError(f'Multiple files for {table}: {[p.name for p in matches]}')
        if not matches:
            missing.append(f'{table.upper()}.csv (or .csv.gz)')
        else:
            paths[table] = matches[0]
    if missing:
        raise FileNotFoundError(
            f'Missing MIMIC-III files in {raw_dir.resolve()}:\n  '
            + '\n  '.join(missing)
            + '\nPlace your authorized MIMIC-III Clinical Database DEMO CSVs there '
            'and rerun. No data is downloaded or generated.')
    tables = {}
    for table, path in paths.items():
        headers = pd.read_csv(path, nrows=0).columns
        normalized = headers.str.strip().str.lower()
        if normalized.duplicated().any():
            raise ValueError(f'{path.name}: duplicate column names after normalization')
        strings = {name: 'string' for name, lower in zip(headers, normalized)
                   if lower.endswith('_id') or lower in ('itemid', 'icd9_code', 'ndc', 'gsn')}
        frame = pd.read_csv(path, dtype=strings, low_memory=False)
        frame.columns = normalized
        absent = set(TABLE_KEYS[table]) - set(frame.columns)
        if absent:
            raise ValueError(f'{path.name}: missing columns {sorted(absent)}; '
                             f'actual headers: {list(frame.columns)}')
        tables[table] = frame
    return tables
