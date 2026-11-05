"""Stage 1 cleaning CLI; no feature engineering or modeling."""
import argparse
from pathlib import Path
import sys
import pandas as pd
from pandas.api.types import is_object_dtype, is_string_dtype
from .data_loading import TABLE_KEYS, load_tables

DATETIME_COLUMNS = {
    'admittime', 'dischtime', 'deathtime', 'intime', 'outtime', 'dob',
    'dod', 'dod_hosp', 'dod_ssn', 'charttime', 'startdate', 'enddate',
}


def _stats(frame: pd.DataFrame) -> tuple[int, int, float]:
    """Return row/column counts and cell missing percentage."""
    return len(frame), len(frame.columns), (
        float(frame.isna().sum().sum() / frame.size * 100) if frame.size else 0.0)


def preprocess_tables(
    tables: dict[str, pd.DataFrame], *, impute_labs: bool = True,
    prune_columns: bool = True,
) -> tuple[dict[str, pd.DataFrame], str]:
    """Clean copies and return an audit report. Never substitute across lab items.

    Keys, target, valuenum and age survive column pruning. Full-data medians
    are descriptive only; later modeling must fit imputation on training data.
    Stage 2 disables lab imputation and pruning to learn them on training data.
    """
    cleaned = {}
    before = {name: _stats(df) for name, df in tables.items()}
    for name, original in tables.items():
        frame = original.copy(deep=True)
        frame.columns = frame.columns.str.strip().str.lower()
        if frame.columns.duplicated().any():
            raise ValueError(f'{name}: duplicate normalized column names')
        for col in frame.columns:
            if col in DATETIME_COLUMNS:
                nonblank = frame[col].replace(r'^\s*$', pd.NA, regex=True)
                try:
                    frame[col] = pd.to_datetime(nonblank, format='mixed', errors='raise')
                except (ValueError, TypeError) as exc:
                    raise ValueError(f'{name}.{col}: invalid datetime: {exc}') from exc
            elif is_object_dtype(frame[col]) or is_string_dtype(frame[col]):
                frame[col] = frame[col].astype('string').str.strip().str.upper().replace('', pd.NA)
        if name == 'labevents' and 'valuenum' in frame:
            try:
                frame['valuenum'] = pd.to_numeric(frame['valuenum'], errors='raise')
            except (ValueError, TypeError) as exc:
                raise ValueError('labevents.valuenum contains non-numeric values') from exc
        cleaned[name] = frame
    patients = cleaned.get('patients', pd.DataFrame())
    admissions = cleaned.get('admissions', pd.DataFrame())
    age_note = 'Age not recomputed: dob or admittime unavailable.'
    if {'subject_id', 'dob'} <= set(patients) and 'admittime' in admissions:
        unique = patients[['subject_id', 'dob']].drop_duplicates()
        if unique['subject_id'].duplicated().any():
            raise ValueError('patients: conflicting birth dates for a subject_id')
        dob = unique.set_index('subject_id')['dob'].reindex(
            admissions['subject_id']).set_axis(admissions.index)
        adm = admissions['admittime']
        pending = (adm.dt.month < dob.dt.month) | (
            (adm.dt.month == dob.dt.month) & (adm.dt.day < dob.dt.day))
        # Calendar arithmetic avoids subtraction overflow for shifted ages >300.
        age = (adm.dt.year - dob.dt.year - pending.astype(int)).astype('Float64')
        admissions['age'] = age.mask(adm.isna() | dob.isna() | (age < 0)).clip(upper=90)
        age_note = 'Age in completed years; ages >90 capped at 90; negative ages left missing.'
    lines = ['# Stage 1 preprocessing report', '', age_note, '',
             '| Table | Rows before | Rows after | Columns before | Columns after | Missing before (%) | Missing after (%) |',
             '|---|---:|---:|---:|---:|---:|---:|']
    details = []
    for name, frame in cleaned.items():
        frame = frame.drop_duplicates().reset_index(drop=True)
        protected = set(TABLE_KEYS.get(name, ())) | {'age'}
        sparse = [c for c in frame if prune_columns and c not in protected and len(frame)
                  and frame[c].isna().mean() > 0.60]
        frame = frame.drop(columns=sparse)
        if impute_labs and name == 'labevents' and {'itemid', 'valuenum'} <= set(frame):
            medians = frame.groupby('itemid', dropna=True)['valuenum'].transform('median')
            frame['valuenum'] = frame['valuenum'].fillna(medians)
        for col in frame:
            if col not in protected and col not in DATETIME_COLUMNS and (
                is_object_dtype(frame[col]) or is_string_dtype(frame[col])):
                frame[col] = frame[col].fillna('UNKNOWN')
        constants = [c for c in frame if prune_columns and c not in protected and len(frame)
                     and frame[c].nunique(dropna=False) <= 1]
        frame = frame.drop(columns=constants).drop_duplicates().reset_index(drop=True)
        # Dropping columns can collapse rows and change missingness fractions.
        # Reach a fixed point so a second call cannot prune additional columns.
        while True:
            extra_sparse = [c for c in frame if prune_columns and c not in protected and len(frame)
                            and frame[c].isna().mean() > 0.60]
            extra_constants = [c for c in frame if prune_columns and c not in protected and len(frame)
                               and frame[c].nunique(dropna=False) <= 1]
            if not extra_sparse and not extra_constants:
                break
            sparse.extend(c for c in extra_sparse if c not in sparse)
            constants.extend(c for c in extra_constants if c not in constants)
            frame = frame.drop(columns=list(set(extra_sparse + extra_constants)))
            frame = frame.drop_duplicates().reset_index(drop=True)
        cleaned[name] = frame
        b, a = before[name], _stats(frame)
        lines.append(f'| {name} | {b[0]} | {a[0]} | {b[1]} | {a[1]} | {b[2]:.2f} | {a[2]:.2f} |')
        details.append(f'- **{name}**: >60% missing removed: {sparse}; constants removed: {constants}.')
    lines += ['', *details, '', 'Keys, hospital_expire_flag, valuenum and age survive sparse/constant pruning.',
              'Categorical missing values use UNKNOWN; other numeric/datetime missing values remain missing.',
              'Lab items without observed numeric values stay missing; missing itemid is never imputed.',
              'Full-data medians are for inspection only. Later modeling must fit imputation on training patients.',
              'Duplicate removal repeats after transformations for idempotence.', '']
    return cleaned, '\n'.join(lines)


def main(argv: list[str] | None = None) -> int:
    """Run Stage 1, returning 1 with an actionable message on input errors."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--raw-dir', type=Path, default=Path('data/raw'))
    parser.add_argument('--output-dir', type=Path, default=Path('data/processed'))
    parser.add_argument('--report', type=Path, default=Path('reports/preprocessing_report.md'))
    args = parser.parse_args(argv)
    try:
        tables, report = preprocess_tables(load_tables(args.raw_dir))
    except (FileNotFoundError, ValueError, pd.errors.EmptyDataError) as exc:
        print(str(exc), file=sys.stderr)
        return 1
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    for name, frame in tables.items():
        frame.to_csv(args.output_dir / f'{name}.csv', index=False)
    args.report.write_text(report, encoding='utf-8')
    print(report)
    print(f'Saved cleaned CSVs to {args.output_dir}; report to {args.report}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
