"""Secondary outcome: retrospective risk of a hospital stay lasting at least seven days.

This module deliberately leaves the mortality model unchanged. Its separate
soft-voting model makes a second outcome visible in the app. Length of stay is
the outcome, so ``length_of_stay_days`` is excluded from every predictor.
"""
import argparse
import json
from pathlib import Path
import pickle
import sys
from typing import Any

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBClassifier

from .data_loading import load_tables
from .features import AdmissionFeatureBuilder, CATEGORICAL, prepare_tables
from .train import RANDOM_STATE, evaluate_predictions

OUTCOME_NAME = 'prolonged_stay_7d'
OUTCOME_LABEL = 'Hospital stay of 7 days or more'
EXCLUDED_FEATURE = 'length_of_stay_days'


def outcome_metadata(raw: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Return admission IDs, patient groups and the derived binary outcome."""
    admissions = prepare_tables(raw)['admissions'].reset_index(drop=True)
    days = (admissions['dischtime'] - admissions['admittime']).dt.total_seconds() / 86400
    if days.isna().any() or (days < 0).any():
        raise ValueError('Admissions need valid non-negative admit/discharge times')
    return pd.DataFrame({'subject_id': admissions['subject_id'].astype(str),
                         'hadm_id': admissions['hadm_id'].astype(str),
                         OUTCOME_NAME: (days >= 7).astype(int)})


def _split(metadata: pd.DataFrame, test_size: float) -> tuple[list[int], list[int]]:
    splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=RANDOM_STATE)
    train, test = next(splitter.split(metadata, metadata[OUTCOME_NAME], metadata['subject_id']))
    if metadata.iloc[train][OUTCOME_NAME].nunique() != 2:
        raise ValueError('The fixed training split has one prolonged-stay class; more data are required')
    return train.tolist(), test.tolist()


def _models(y: pd.Series, trees: int) -> dict[str, Any]:
    positives, negatives = int(y.sum()), int((y == 0).sum())
    if not positives or not negatives:
        raise ValueError('Both prolonged-stay classes are required for training')
    return {
        'logistic_regression': LogisticRegression(class_weight='balanced', max_iter=2000, random_state=RANDOM_STATE),
        'random_forest': RandomForestClassifier(n_estimators=trees, max_depth=5, min_samples_leaf=2,
                                                  class_weight='balanced_subsample', n_jobs=1, random_state=RANDOM_STATE),
        'xgboost': XGBClassifier(n_estimators=trees, max_depth=2, learning_rate=0.05, subsample=0.8,
                                 colsample_bytree=0.8, reg_lambda=1.0, scale_pos_weight=negatives / positives,
                                 objective='binary:logistic', eval_metric='logloss', tree_method='hist',
                                 n_jobs=1, random_state=RANDOM_STATE),
    }


def train_secondary_model(raw: dict[str, pd.DataFrame], model_path: Path, report_path: Path,
                          *, trees: int = 200, test_size: float = 0.2) -> dict[str, Any]:
    """Fit three base learners and their equal-weight soft-voting ensemble."""
    metadata = outcome_metadata(raw)
    train, test = _split(metadata, test_size)
    train_ids = metadata.iloc[train]['subject_id'].unique().tolist()
    builder = AdmissionFeatureBuilder().fit(raw, train_ids)
    features = builder.transform(raw).drop(columns=[EXCLUDED_FEATURE])
    numeric = [name for name in builder.numeric_columns_ if name != EXCLUDED_FEATURE]
    encoder = ColumnTransformer([
        ('numeric', 'passthrough', numeric),
        ('categorical', OneHotEncoder(handle_unknown='ignore', sparse_output=False), CATEGORICAL),
    ], remainder='drop', verbose_feature_names_out=False)
    x_train = encoder.fit_transform(features.iloc[train])
    x_test = encoder.transform(features.iloc[test])
    y_train, y_test = metadata.iloc[train][OUTCOME_NAME], metadata.iloc[test][OUTCOME_NAME]
    models = _models(y_train, trees)
    probabilities: dict[str, Any] = {}
    for name, model in models.items():
        model.fit(x_train, y_train)
        probabilities[name] = model.predict_proba(x_test)[:, 1]
    probabilities['soft_voting_ensemble'] = sum(probabilities.values()) / len(probabilities)
    metrics = {name: evaluate_predictions(y_test, scores) for name, scores in probabilities.items()}
    bundle = {
        'version': 1, 'outcome_name': OUTCOME_NAME, 'outcome_label': OUTCOME_LABEL,
        'threshold_days': 7, 'excluded_feature': EXCLUDED_FEATURE,
        'feature_state': builder.state(), 'numeric_columns': numeric,
        'encoder': encoder, 'models': models, 'threshold': 0.5,
        'train_subject_ids': sorted(train_ids),
        'interpretation': 'retrospective whole-stay benchmark; not a clinical prediction tool',
    }
    model_path.parent.mkdir(parents=True, exist_ok=True)
    with model_path.open('wb') as stream:
        pickle.dump(bundle, stream, protocol=pickle.HIGHEST_PROTOCOL)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    lines = ['# Secondary outcome: prolonged hospital-stay risk', '',
             f'Outcome: **{OUTCOME_LABEL}**.',
             'The target is derived from ADMISSIONS admit/discharge times. The target-defining length-of-stay feature is excluded from predictors.',
             'This is a separate retrospective soft-voting ensemble; the mortality model and its artifacts are unchanged.', '',
             f'Admissions: {len(metadata)}; patients: {metadata.subject_id.nunique()}.',
             f'Training admissions: {len(train)}; held-out admissions: {len(test)}; patient overlap: 0.', '',
             '## Held-out metrics', '',
             '| Model | ROC-AUC | PR-AUC (AP) | Precision | Recall | F1 |',
             '|---|---:|---:|---:|---:|---:|']
    for name, values in metrics.items():
        number = lambda value: 'undefined' if value is None else f'{value:.3f}'
        lines.append('| ' + name.replace('_', ' ') + ' | ' + ' | '.join(number(values[key]) for key in ['roc_auc', 'pr_auc_ap', 'precision', 'recall', 'f1']) + ' |')
    lines += ['', 'The ensemble is the arithmetic mean of the three class-1 probabilities. Whole-stay event features can still cause temporal leakage, so this is educational only and not an admission-time or clinical decision score.', '']
    report_path.write_text('\n'.join(lines), encoding='utf-8')
    return {'metrics': metrics, 'train_admissions': len(train), 'test_admissions': len(test),
            'model_path': str(model_path), 'report_path': str(report_path)}


def load_secondary_bundle(path: Path) -> dict[str, Any]:
    """Load a locally generated trusted bundle and validate its basic schema."""
    with path.open('rb') as stream:
        bundle = pickle.load(stream)
    required = {'version', 'outcome_name', 'feature_state', 'encoder', 'models', 'excluded_feature'}
    if not isinstance(bundle, dict) or bundle.get('version') != 1 or required - set(bundle):
        raise ValueError('Unsupported prolonged-stay model bundle')
    return bundle


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--raw-dir', type=Path, default=Path('data/raw'))
    parser.add_argument('--model-path', type=Path, default=Path('models/prolonged_stay_ensemble.pkl'))
    parser.add_argument('--report-path', type=Path, default=Path('reports/prolonged_stay_model.md'))
    parser.add_argument('--trees', type=int, default=200)
    args = parser.parse_args(argv)
    try:
        result = train_secondary_model(load_tables(args.raw_dir), args.model_path, args.report_path, trees=args.trees)
        print(json.dumps(result, indent=2, default=str))
    except (ValueError, FileNotFoundError, pd.errors.EmptyDataError) as exc:
        print(str(exc), file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
