"""Stage 3: patient-grouped evaluation of retrospective mortality models."""
import argparse
from dataclasses import dataclass
import importlib.metadata
import json
import math
import os
from pathlib import Path
import pickle
import sys
from typing import Any
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, confusion_matrix, f1_score,
                             precision_recall_curve, precision_score, recall_score,
                             roc_auc_score, roc_curve)
from sklearn.model_selection import GroupShuffleSplit, StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBClassifier
from .data_loading import load_tables
from .features import AdmissionFeatureBuilder, CATEGORICAL, prepare_tables

RANDOM_STATE = 42
METRICS = ['roc_auc', 'pr_auc_ap', 'precision', 'recall', 'f1']
MODEL_NAMES = ['logistic_regression', 'random_forest', 'xgboost']


def evaluate_predictions(y_true: pd.Series | np.ndarray,
                         probabilities: np.ndarray, threshold: float = 0.5) -> dict[str, Any]:
    """Report AP (step-wise PR area), ROC-AUC and fixed-threshold metrics.

    Undefined ranking metrics are None for a single-class evaluation partition.
    Confusion matrix axes are actual/predicted, with labels [0, 1].
    """
    y = np.asarray(y_true)
    scores = np.asarray(probabilities, dtype=float)
    if y.ndim != 1 or len(y) == 0 or scores.shape != y.shape:
        raise ValueError('Nonempty, aligned one-dimensional outcomes and scores required')
    if not np.isin(y, [0, 1]).all() or not np.isfinite(scores).all() or ((scores < 0) | (scores > 1)).any():
        raise ValueError('Binary outcomes and finite probabilities in [0,1] required')
    if not 0 < threshold < 1:
        raise ValueError('Threshold must be between 0 and 1')
    predicted = (scores >= threshold).astype(int)
    both = len(np.unique(y)) == 2
    return {
        'roc_auc': float(roc_auc_score(y, scores)) if both else None,
        'pr_auc_ap': float(average_precision_score(y, scores)) if both else None,
        'precision': float(precision_score(y, predicted, zero_division=0)),
        'recall': float(recall_score(y, predicted, zero_division=0)),
        'f1': float(f1_score(y, predicted, zero_division=0)),
        'confusion_matrix': confusion_matrix(y, predicted, labels=[0, 1]).tolist(),
        'n': len(y), 'positive': int(y.sum()), 'threshold': threshold,
        'ranking_metrics_defined': both,
    }


def grouped_holdout(metadata: pd.DataFrame, test_size: float = 0.2) -> tuple[np.ndarray, np.ndarray]:
    """One fixed GroupShuffleSplit; never retry seeds to optimize label balance."""
    if not 0 < test_size < 1:
        raise ValueError('test_size must be between 0 and 1')
    if metadata['subject_id'].isna().any() or metadata.subject_id.nunique() < 2:
        raise ValueError('Holdout requires at least two nonmissing patient groups')
    splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=RANDOM_STATE)
    train, test = next(splitter.split(metadata, groups=metadata['subject_id']))
    if metadata.iloc[train].hospital_expire_flag.nunique() != 2:
        raise ValueError('The fixed training split has one outcome class; more data are required')
    return train, test


def grouped_cv(metadata: pd.DataFrame, train_indices: np.ndarray,
               n_splits: int = 5) -> list[tuple[np.ndarray, np.ndarray]]:
    """Create stratified grouped folds inside the training partition only.

    Never reduce the requested fold count or reuse the test set to repair an
    infeasible split. Both labels must exist in every fit/validation partition.
    """
    if n_splits < 2:
        raise ValueError('At least two cross-validation folds are required')
    train = metadata.iloc[train_indices]
    labels = train['hospital_expire_flag']
    if labels.nunique() != 2 or train.subject_id.nunique() < n_splits:
        raise ValueError(f'Insufficient training patients/classes for {n_splits}-fold grouped CV')
    for label in [0, 1]:
        if train.loc[labels == label, 'subject_id'].nunique() < n_splits:
            raise ValueError(f'Class {label} occurs in fewer than {n_splits} training patient groups')
    cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    folds = []
    for fit_local, valid_local in cv.split(train, labels, train['subject_id']):
        fit, valid = train_indices[fit_local], train_indices[valid_local]
        if any(metadata.iloc[idx].hospital_expire_flag.nunique() != 2 for idx in [fit, valid]):
            raise ValueError('Stratified grouped CV produced a single-class fold; more data are required')
        if set(metadata.iloc[fit].subject_id) & set(metadata.iloc[valid].subject_id):
            raise ValueError('Patient overlap in cross-validation')
        folds.append((fit, valid))
    return folds


def make_models(y_train: pd.Series, trees: int = 200) -> dict[str, Any]:
    """Fixed model settings; class balance uses only current training labels."""
    positives = int((y_train == 1).sum())
    negatives = int((y_train == 0).sum())
    if not positives or not negatives or trees < 1:
        raise ValueError('Training requires both classes and at least one tree')
    return {
        'logistic_regression': LogisticRegression(class_weight='balanced',
            max_iter=2000, random_state=RANDOM_STATE),
        'random_forest': RandomForestClassifier(n_estimators=trees, max_depth=5,
            min_samples_leaf=2, class_weight='balanced_subsample', n_jobs=1,
            random_state=RANDOM_STATE),
        'xgboost': XGBClassifier(n_estimators=trees, max_depth=2, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
            scale_pos_weight=negatives / positives, objective='binary:logistic',
            eval_metric='logloss', tree_method='hist', n_jobs=1,
            random_state=RANDOM_STATE),
    }


@dataclass
class PartitionResult:
    """Fitted objects and predictions for one independent evaluation partition."""
    builder: AdmissionFeatureBuilder
    pipelines: dict[str, Pipeline]
    probabilities: dict[str, np.ndarray]
    metrics: dict[str, dict[str, Any]]
    encoded_feature_names: list[str]


def fit_partition(raw: dict[str, pd.DataFrame], metadata: pd.DataFrame,
                  fit_indices: np.ndarray, valid_indices: np.ndarray,
                  trees: int = 200) -> PartitionResult:
    """Refit all learned feature/encoding state for this partition only."""
    expected = prepare_tables(raw)['admissions'][['subject_id', 'hadm_id', 'hospital_expire_flag']].reset_index(drop=True)
    if len(expected) != len(metadata) or not expected.astype(str).equals(metadata.reset_index(drop=True).astype(str)):
        raise ValueError('Metadata and raw admission rows are not aligned')
    train_ids = metadata.iloc[fit_indices]['subject_id'].unique().tolist()
    valid_ids = set(metadata.iloc[valid_indices]['subject_id'])
    if set(train_ids) & valid_ids:
        raise ValueError('Training and evaluation patients must not overlap')
    builder = AdmissionFeatureBuilder().fit(raw, train_ids)
    features = builder.transform(raw)
    if len(features) != len(metadata):
        raise ValueError('Metadata and feature rows are not aligned')
    train_x, valid_x = features.iloc[fit_indices], features.iloc[valid_indices]
    train_y = metadata.iloc[fit_indices]['hospital_expire_flag']
    valid_y = metadata.iloc[valid_indices]['hospital_expire_flag']
    pipelines, probabilities, metrics = {}, {}, {}
    for name, model in make_models(train_y, trees).items():
        encoder = ColumnTransformer([
            ('numeric', 'passthrough', builder.numeric_columns_),
            ('categorical', OneHotEncoder(handle_unknown='ignore', sparse_output=False), CATEGORICAL),
        ], remainder='drop', verbose_feature_names_out=False)
        pipeline = Pipeline([('encoding', encoder), ('model', model)])
        pipeline.fit(train_x, train_y)
        probability = pipeline.predict_proba(valid_x)[:, 1]
        pipelines[name], probabilities[name] = pipeline, probability
        metrics[name] = evaluate_predictions(valid_y, probability)
    encoded = pipelines['logistic_regression'].named_steps['encoding'].get_feature_names_out().tolist()
    return PartitionResult(builder, pipelines, probabilities, metrics, encoded)


def summarize_cv(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Summarize fold metrics with sample SD, not a confidence interval."""
    frame = pd.DataFrame(rows)
    summary = {}
    for model in MODEL_NAMES:
        selected = frame.loc[frame.model == model]
        if selected.empty:
            raise ValueError(f'Missing CV results for {model}')
        summary[model] = {}
        for metric in METRICS:
            values = selected[metric].dropna().astype(float)
            summary[model][metric] = {
                'mean': float(values.mean()) if len(values) else None,
                'std': float(values.std(ddof=1)) if len(values) > 1 else None,
                'folds': len(values)}
    return summary


def select_model(cv_summary: dict[str, dict[str, Any]]) -> str:
    """Choose using mean training-CV AP, with ROC-AUC and name as tie-breakers."""
    def key(name: str) -> tuple[float, float, str]:
        ap, auc = cv_summary[name]['pr_auc_ap']['mean'], cv_summary[name]['roc_auc']['mean']
        if ap is None or auc is None:
            raise ValueError('Defined CV ranking metrics are required for model selection')
        return -ap, -auc, name
    return sorted(cv_summary, key=key)[0]


def save_plots(predictions: pd.DataFrame, output_dir: Path) -> None:
    """Render held-out ROC, PR and confusion plots using a noninteractive backend."""
    output_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault('MPLCONFIGDIR', str((output_dir / '.matplotlib').resolve()))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    output_dir.mkdir(parents=True, exist_ok=True)
    for kind in ['roc', 'precision_recall']:
        fig, ax = plt.subplots(figsize=(7, 5), layout='constrained')
        for name, selected in predictions.groupby('model', sort=False):
            y, probability = selected['y_true'], selected['probability']
            if y.nunique() != 2:
                continue
            if kind == 'roc':
                x, y_curve, _ = roc_curve(y, probability)
                label = f'{name} (AUC={roc_auc_score(y, probability):.3f})'
            else:
                y_curve, x, _ = precision_recall_curve(y, probability)
                label = f'{name} (AP={average_precision_score(y, probability):.3f})'
            ax.step(x, y_curve, where='post', label=label)
        if kind == 'roc':
            ax.plot([0, 1], [0, 1], '--', color='gray', label='Chance')
            ax.set(xlabel='False positive rate', ylabel='True positive rate')
        else:
            baseline = float(predictions.drop_duplicates('hadm_id')['y_true'].mean())
            ax.axhline(baseline, linestyle='--', color='gray', label=f'Prevalence={baseline:.3f}')
            ax.set(xlabel='Recall', ylabel='Precision')
        ax.set(xlim=(0, 1), ylim=(0, 1.03), title='Held-out patients: retrospective benchmark')
        ax.legend(fontsize=8, loc='best')
        ax.grid(alpha=0.2)
        fig.savefig(output_dir / f'{kind}_curves.png', dpi=160)
        plt.close(fig)
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.5), layout='constrained')
    for ax, name in zip(axes, MODEL_NAMES):
        selected = predictions.loc[predictions.model == name]
        matrix = confusion_matrix(selected.y_true, selected.predicted, labels=[0, 1])
        ax.imshow(matrix, cmap='Blues', vmin=0)
        for i in range(2):
            for j in range(2):
                ax.text(j, i, str(matrix[i, j]), ha='center', va='center',
                        color='white' if matrix[i, j] > matrix.max()/2 else 'black')
        ax.set(xticks=[0, 1], yticks=[0, 1], xlabel='Predicted label',
               ylabel='Actual label', title=name.replace('_', ' '))
    fig.suptitle('Held-out confusion matrices at threshold 0.5')
    fig.savefig(output_dir / 'confusion_matrices.png', dpi=160)
    plt.close(fig)


def prediction_rows(metadata: pd.DataFrame, indices: np.ndarray,
                    result: PartitionResult, fold: int | None = None) -> list[dict[str, Any]]:
    """Attach explicit IDs to probabilities for an auditable row alignment."""
    rows = []
    for name, probabilities in result.probabilities.items():
        selected = metadata.iloc[indices].reset_index(drop=True)
        for index, record in selected.iterrows():
            probability = float(probabilities[index])
            rows.append({'model': name, 'subject_id': str(record.subject_id),
                'hadm_id': str(record.hadm_id), 'y_true': int(record.hospital_expire_flag),
                'probability': probability, 'predicted': int(probability >= 0.5), 'fold': fold})
    return rows


def write_report(results: dict[str, Any], path: Path) -> str:
    """Write model comparison and explain the limits of this small benchmark."""
    def number(value: float | None) -> str:
        return 'undefined' if value is None else f'{value:.3f}'
    lines = ['# Stage 3 model comparison', '',
        'Retrospective whole-stay demonstration; not a clinically validated early-risk predictor.', '',
        f"Patients: {results['patients']}; admissions: {results['admissions']}.",
        f"Training: {results['train_patients']} patients / {results['train_admissions']} admissions.",
        f"Holdout: {results['test_patients']} patients / {results['test_admissions']} admissions.",
        f"Held-out mortality labels: {results['holdout_positive']}; patient overlap: 0.",
        f"Selected model: **{results['selected_model']}**, chosen by mean training-only CV average precision.", '',
        '## Held-out results', '',
        '| Model | ROC-AUC | PR-AUC (AP) | Precision | Recall | F1 | Confusion [[TN, FP], [FN, TP]] |',
        '|---|---:|---:|---:|---:|---:|---|']
    for name in MODEL_NAMES:
        row = results['holdout'][name]
        lines.append('| ' + name + ' | ' + ' | '.join(number(row[k]) for k in METRICS)
                     + f" | {row['confusion_matrix']} |")
    lines += ['', f"## Stratified {results['n_splits']}-fold grouped CV on training patients", '',
        'Mean +/- sample standard deviation across folds; these are not confidence intervals.', '',
        '| Model | ROC-AUC | PR-AUC (AP) | Precision | Recall | F1 |',
        '|---|---:|---:|---:|---:|---:|']
    for name in MODEL_NAMES:
        values = results['cv_summary'][name]
        cells = [f"{number(values[k]['mean'])} +/- {number(values[k]['std'])}" for k in METRICS]
        lines.append('| ' + name + ' | ' + ' | '.join(cells) + ' |')
    lines += ['', '## Evaluation procedure', '',
        '- The holdout is a single GroupShuffleSplit(random_state=42); no seed search is used.',
        '- StratifiedGroupKFold(shuffle=True, random_state=42) operates only within training patients. Fold label balance is approximate under group constraints.',
        '- Every fold refits top-lab selection, lab medians, admission-level imputation, scaling and one-hot encoding using its fitting patients only.',
        '- Logistic regression uses balanced class weights; random forest uses balanced_subsample; XGBoost scale_pos_weight is computed from each fitting partition.',
        '- Model settings are fixed before evaluation. The classification threshold is 0.5 for all models; no test-set tuning, early stopping or model selection is performed.',
        '- PR-AUC is reported as average precision (step-wise integration), not trapezoidal PR area. Zero-division precision/recall/F1 are reported as zero.',
        '- Single-class evaluation partitions have undefined ROC-AUC/AP; the CLI rejects infeasible five-fold training CV rather than silently changing the protocol.',
        '- Final models are fitted to training patients only. The selected model is not refitted on the holdout after reporting.', '',
        '## Files', '',
        '- roc_curves.png, precision_recall_curves.png, confusion_matrices.png: held-out plots.',
        '- model_metrics.json: machine-readable metrics, parameters, versions and CV-only model choice.',
        '- data/processed/stage3/: split/fold assignments, fold metrics, out-of-fold and holdout predictions, preprocessing audits.',
        '- models/: fitted training-only model bundles and selected_model.json. Load only trusted pickle bundles using the recorded package versions.', '',
        '## Limitations', '',
        'The demo contains only 100 patients. Results are indicative and sensitive to which patients land in each partition. Whole-stay LOS, counts and labs are unavailable at admission time and can encode outcome-related events; excluding explicit death fields does not remove that temporal limitation. This is an educational retrospective benchmark, not medical advice, a calibrated individual risk estimate or evidence of clinical validity. Future work requires a prediction-time window, calibration and validation on full MIMIC-III/IV.', '']
    if results['holdout_positive'] in [0, results['test_admissions']]:
        lines += ['The fixed holdout has only one outcome class; ranking metrics and curves are undefined.', '']
    content = '\n'.join(lines)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding='utf-8')
    return content


def auditable_parameters(model: Any) -> dict[str, Any]:
    """Represent estimator defaults as strict JSON, including XGBoost NaN."""
    return {key: ('NaN' if isinstance(value, float) and math.isnan(value) else value)
            for key, value in model.get_params().items()}


def run_training(raw: dict[str, pd.DataFrame], output_dir: Path, report_dir: Path,
                 model_dir: Path, *, trees: int = 200, test_size: float = 0.2,
                 n_splits: int = 5) -> dict[str, Any]:
    """Evaluate fixed candidate models, save fitted bundles and audit artifacts."""
    tables = prepare_tables(raw)
    metadata = tables['admissions'][['subject_id', 'hadm_id', 'hospital_expire_flag']].reset_index(drop=True)
    train, test = grouped_holdout(metadata, test_size)
    folds = grouped_cv(metadata, train, n_splits)
    fold_rows, cv_predictions, audits = [], [], []
    assignment = metadata.copy()
    assignment['split'] = 'holdout'
    assignment.loc[train, 'split'] = 'train'
    assignment['cv_fold'] = pd.Series(pd.NA, index=assignment.index, dtype='Int64')
    for fold, (fit, valid) in enumerate(folds, start=1):
        result = fit_partition(raw, metadata, fit, valid, trees)
        assignment.loc[valid, 'cv_fold'] = fold
        for name in MODEL_NAMES:
            fold_rows.append({'fold': fold, 'model': name, **result.metrics[name]})
        cv_predictions.extend(prediction_rows(metadata, valid, result, fold))
        audits.append({'fold': fold, 'feature_state': result.builder.state(),
            'validation_subject_ids': sorted(set(metadata.iloc[valid].subject_id)),
            'encoded_feature_names': result.encoded_feature_names,
            'model_parameters': {name: auditable_parameters(pipe.named_steps['model'])
                                 for name, pipe in result.pipelines.items()}})
        print(f'Completed grouped CV fold {fold}/{n_splits}', flush=True)
    summary = summarize_cv(fold_rows)
    selected = select_model(summary)  # Freeze choice before evaluating holdout.
    final = fit_partition(raw, metadata, train, test, trees)
    holdout = pd.DataFrame(prediction_rows(metadata, test, final))
    versions = {name: importlib.metadata.version(name) for name in
                ['pandas', 'numpy', 'scikit-learn', 'xgboost', 'matplotlib']}
    results = {
        'random_state': RANDOM_STATE, 'n_splits': n_splits, 'trees': trees,
        'patients': int(metadata.subject_id.nunique()), 'admissions': len(metadata),
        'train_patients': int(metadata.iloc[train].subject_id.nunique()),
        'test_patients': int(metadata.iloc[test].subject_id.nunique()),
        'train_admissions': len(train), 'test_admissions': len(test),
        'holdout_positive': int(metadata.iloc[test].hospital_expire_flag.sum()),
        'selected_model': selected, 'cv_summary': summary, 'holdout': final.metrics,
        'versions': versions,
        'model_parameters': {name: auditable_parameters(pipe.named_steps['model'])
                             for name, pipe in final.pipelines.items()},
    }
    # Validate serialization before writing artifacts.
    json.dumps(results, allow_nan=False)
    json.dumps(audits, allow_nan=False)
    for directory in [output_dir, report_dir, model_dir]:
        directory.mkdir(parents=True, exist_ok=True)
    assignment.to_csv(output_dir / 'split_assignments.csv', index=False)
    pd.DataFrame(fold_rows).to_csv(output_dir / 'cv_fold_metrics.csv', index=False)
    pd.DataFrame(cv_predictions).to_csv(output_dir / 'cv_predictions.csv', index=False)
    holdout.to_csv(output_dir / 'holdout_predictions.csv', index=False)
    (output_dir / 'cv_preprocessing_audit.json').write_text(json.dumps(audits, indent=2, allow_nan=False), encoding='utf-8')
    (output_dir / 'feature_state.json').write_text(json.dumps(final.builder.state(), indent=2, allow_nan=False), encoding='utf-8')
    (report_dir / 'model_metrics.json').write_text(json.dumps(results, indent=2, allow_nan=False), encoding='utf-8')
    for name, pipeline in final.pipelines.items():
        bundle = {'version': 1, 'model_name': name, 'feature_state': final.builder.state(),
            'pipeline': pipeline, 'encoded_feature_names': final.encoded_feature_names,
            'threshold': 0.5, 'versions': versions, 'interpretation': 'retrospective whole-stay benchmark'}
        with (model_dir / f'{name}.pkl').open('wb') as stream:
            pickle.dump(bundle, stream, protocol=pickle.HIGHEST_PROTOCOL)
    final.pipelines['xgboost'].named_steps['model'].save_model(model_dir / 'xgboost.json')
    (model_dir / 'selected_model.json').write_text(json.dumps({
        'selected_model': selected, 'bundle': f'{selected}.pkl',
        'selection_rule': 'highest mean training-CV average precision; ROC-AUC then name tie-breakers'}, indent=2), encoding='utf-8')
    save_plots(holdout, report_dir)
    print(write_report(results, report_dir / 'model_comparison.md'))
    return results


def main(argv: list[str] | None = None) -> int:
    """Run Stage 3 without starting SHAP, RAG or app stages."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--raw-dir', type=Path, default=Path('data/raw'))
    parser.add_argument('--output-dir', type=Path, default=Path('data/processed/stage3'))
    parser.add_argument('--report-dir', type=Path, default=Path('reports'))
    parser.add_argument('--model-dir', type=Path, default=Path('models'))
    parser.add_argument('--trees', type=int, default=200)
    parser.add_argument('--test-size', type=float, default=0.2)
    args = parser.parse_args(argv)
    try:
        run_training(load_tables(args.raw_dir), args.output_dir, args.report_dir,
                     args.model_dir, trees=args.trees, test_size=args.test_size)
    except (ValueError, FileNotFoundError, pd.errors.EmptyDataError) as exc:
        print(str(exc), file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
