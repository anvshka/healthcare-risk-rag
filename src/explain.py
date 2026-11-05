"""Stage 4: auditable SHAP explanations of the saved mortality model."""
import argparse
from dataclasses import dataclass
import importlib.metadata
import json
import os
from pathlib import Path
import pickle
import sys
from typing import Any
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from .data_loading import load_tables
from .features import AdmissionFeatureBuilder, CATEGORICAL, prepare_tables

DISCLAIMER = ('Educational retrospective decision-support demo, not medical advice. '
              'SHAP describes model behavior, not clinical causation or calibrated patient risk.')


@dataclass
class ExplanationBatch:
    """Class-1 probability contributions aligned to admission row positions."""
    indices: np.ndarray
    encoded_values: np.ndarray
    grouped_values: pd.DataFrame
    base_values: np.ndarray
    probabilities: np.ndarray


def binary_shap_arrays(explanation: Any, n_rows: int, n_features: int) -> tuple[np.ndarray, np.ndarray]:
    """Normalize binary-class and scalar-positive-output SHAP array formats."""
    values = np.asarray(explanation.values, dtype=float)
    base = np.asarray(explanation.base_values, dtype=float)
    if values.shape == (n_rows, n_features, 2):
        values = values[:, :, 1]
        if base.shape == (n_rows, 2):
            base = base[:, 1]
        elif base.shape == (2,):
            base = np.repeat(base[1], n_rows)
        else:
            raise ValueError(f'Unexpected binary SHAP baseline shape: {base.shape}')
    elif values.shape != (n_rows, n_features):
        raise ValueError(f'Unexpected SHAP value shape: {values.shape}')
    if base.ndim == 0 or base.size == 1:
        base = np.repeat(float(base.reshape(-1)[0]), n_rows)
    if base.shape != (n_rows,) or not np.isfinite(values).all() or not np.isfinite(base).all():
        raise ValueError('SHAP output contains invalid baselines or nonfinite contributions')
    return values, base


class RiskExplainer:
    """Explain class 1 using frozen preprocessing and the saved model.

    Random Forest uses exact Tree SHAP and its fitted training-path statistics.
    Other classifiers use probability-space permutation SHAP with a training-only
    background (approximate allocation, checked exact prediction reconstruction).
    """

    def __init__(self, bundle: dict[str, Any], raw: dict[str, pd.DataFrame],
                 max_background: int = 100) -> None:
        if max_background < 1:
            raise ValueError('max_background must be positive')
        if bundle.get('version') != 1:
            raise ValueError('Unsupported saved model bundle version; rerun Stage 3')
        self.bundle = bundle
        self.builder = AdmissionFeatureBuilder.from_state(bundle['feature_state'])
        self.pipeline = bundle['pipeline']
        self.encoder = self.pipeline.named_steps['encoding']
        self.model = self.pipeline.named_steps['model']
        if list(self.model.classes_) != [0, 1]:
            raise ValueError('Expected binary model classes [0,1]')
        self.metadata = prepare_tables(raw)['admissions'][['subject_id', 'hadm_id']].reset_index(drop=True)
        self.features = self.builder.transform(raw)
        self.original_features = self.builder.transform(raw, standardize=False)
        self.encoded = np.asarray(self.encoder.transform(self.features), dtype=float)
        self.encoded_names = self.encoder.get_feature_names_out().tolist()
        if self.encoded_names != bundle['encoded_feature_names']:
            raise ValueError('Saved encoded feature names do not match the fitted encoder')
        if not np.isfinite(self.encoded).all():
            raise ValueError('Nonfinite model inputs cannot be explained')
        train_ids = set(self.builder.train_subject_ids_)
        if not train_ids <= set(self.metadata.subject_id):
            raise ValueError('Raw data must include the saved model training patients')
        self.training_mask = self.metadata.subject_id.isin(train_ids).to_numpy()
        self.heldout_indices = np.flatnonzero(~self.training_mask)
        train_positions = pd.Series(np.flatnonzero(self.training_mask))
        self.background_indices = train_positions.sample(n=min(max_background, len(train_positions)),
                                                         random_state=42).to_numpy()
        self.mapping: dict[str, list[int]] = {}
        numeric = self.builder.numeric_columns_
        if self.encoded_names[:len(numeric)] != numeric:
            raise ValueError('Unexpected numeric encoder ordering')
        for position, name in enumerate(numeric):
            self.mapping[name] = [position]
        offset = len(numeric)
        onehot = self.encoder.named_transformers_['categorical']
        if onehot.drop is not None:
            raise ValueError('Category grouping expects the Stage 3 non-dropping one-hot encoder')
        for name, categories in zip(CATEGORICAL, onehot.categories_):
            self.mapping[name] = list(range(offset, offset + len(categories)))
            offset += len(categories)
        if offset != len(self.encoded_names):
            raise ValueError('Categorical encoder columns are not completely mapped')
        import shap
        if isinstance(self.model, RandomForestClassifier):
            self.method = 'Tree SHAP: tree_path_dependent; Random Forest class-1 probability'
            self.baseline_definition = ('Expected class-1 output under the fitted training-tree path statistics; '
                                        'class weighting can make this differ from observed mortality prevalence.')
            self.explainer = shap.TreeExplainer(self.model, feature_perturbation='tree_path_dependent',
                                                model_output='raw')
        else:
            self.method = 'Permutation SHAP: class-1 probability; training-only background; 10 permutations'
            self.baseline_definition = 'Mean model probability over a fixed sample of up to 100 training admissions.'
            self.explainer = None

    def explain_indices(self, indices: np.ndarray | list[int]) -> ExplanationBatch:
        """Explain admission positions and verify class-1 probability additivity."""
        indices = np.asarray(indices)
        if indices.ndim != 1 or indices.dtype.kind not in 'iu' or len(indices) == 0:
            raise ValueError('Nonempty one-dimensional integer admission positions required')
        if (indices < 0).any() or (indices >= len(self.metadata)).any() or len(set(indices)) != len(indices):
            raise ValueError('Admission positions must be unique and in range')
        matrix = self.encoded[indices]
        if self.explainer is not None:
            result = self.explainer(matrix, check_additivity=True)
        else:
            import shap
            explainer = shap.Explainer(lambda data: self.model.predict_proba(data)[:, 1],
                self.encoded[self.background_indices], algorithm='permutation', seed=42)
            result = explainer(matrix, max_evals=10 * (2 * matrix.shape[1] + 1), silent=True)
        values, base = binary_shap_arrays(result, len(indices), matrix.shape[1])
        probabilities = self.pipeline.predict_proba(self.features.iloc[indices])[:, 1]
        if not np.allclose(base + values.sum(axis=1), probabilities, rtol=1e-6, atol=1e-6):
            raise ValueError('SHAP contributions do not reconstruct saved-model class-1 probabilities')
        grouped = pd.DataFrame({name: values[:, columns].sum(axis=1)
                                for name, columns in self.mapping.items()})
        if not np.allclose(grouped.sum(axis=1).to_numpy(), values.sum(axis=1), atol=1e-12):
            raise ValueError('Grouping one-hot contributions changed their sum')
        return ExplanationBatch(indices, values, grouped, base, probabilities)

    def top_factors(self, batch: ExplanationBatch, row: int, top_k: int = 5) -> list[dict[str, Any]]:
        """Return original-unit feature/value/direction/magnitude records.

        Numeric values are engineered summaries after training-derived imputation;
        value_imputed identifies admission-level filling, not lab-event-level filling.
        Category contributions sum all dummies for that field, including zero dummies.
        """
        if top_k < 1 or not 0 <= row < len(batch.indices):
            raise ValueError('Positive top_k and valid explanation row are required')
        contributions = batch.grouped_values.iloc[row]
        ordered = sorted(contributions.index, key=lambda name: (-abs(contributions[name]), name))
        index = int(batch.indices[row])
        factors = []
        for name in ordered[:top_k]:
            contribution = float(contributions[name])
            value = self.original_features.iloc[index][name]
            imputed = bool(pd.isna(value))
            if name in self.builder.numeric_columns_:
                value = float(self.features.iloc[index][name] * self.builder.scales_[name]
                              + self.builder.means_[name])
            else:
                value = str(value)
            factors.append({'feature': name, 'value': value,
                'direction': 'raises' if contribution > 0 else 'lowers' if contribution < 0 else 'neutral',
                'magnitude': abs(contribution), 'shap_value': contribution,
                'percentage_points': contribution * 100, 'value_imputed': imputed,
                'contribution_unit': 'probability'})
        return factors

    def records(self, batch: ExplanationBatch, top_k: int = 5) -> list[dict[str, Any]]:
        """Return JSON-ready explanations without target labels or clinical claims."""
        records = []
        for row, index in enumerate(batch.indices):
            identity = self.metadata.iloc[index]
            factors = self.top_factors(batch, row, top_k)
            total = float(batch.grouped_values.iloc[row].sum())
            records.append({'subject_id': str(identity.subject_id), 'hadm_id': str(identity.hadm_id),
                'model': self.bundle['model_name'], 'scope': 'training' if self.training_mask[index] else 'heldout',
                'probability': float(batch.probabilities[row]), 'baseline_probability': float(batch.base_values[row]),
                'all_contributions_sum': total,
                'other_contributions_sum': total - sum(f['shap_value'] for f in factors),
                'reconstruction_error': float(abs(batch.base_values[row] + total - batch.probabilities[row])),
                'baseline_definition': self.baseline_definition, 'method': self.method,
                'top_factors': factors, 'disclaimer': DISCLAIMER})
        return records

    def explain_admission(self, hadm_id: str, top_k: int = 5) -> dict[str, Any]:
        """Explain a specific admission, disambiguating repeated patient admissions."""
        matches = np.flatnonzero(self.metadata.hadm_id.eq(str(hadm_id).strip().upper()))
        if len(matches) != 1:
            raise ValueError(f'Unknown or ambiguous hadm_id: {hadm_id}')
        return self.records(self.explain_indices(matches), top_k)[0]


def load_selected_bundle(model_dir: Path) -> dict[str, Any]:
    """Load the trusted local Stage 3 selected bundle; never load arbitrary uploads."""
    manifest_path = model_dir / 'selected_model.json'
    if not manifest_path.is_file():
        raise FileNotFoundError(f'Missing {manifest_path}; run python -m src.train first')
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    name = manifest.get('selected_model')
    if name not in {'logistic_regression', 'random_forest', 'xgboost'} or manifest.get('bundle') != f'{name}.pkl':
        raise ValueError('Invalid Stage 3 selected-model manifest')
    path = model_dir / manifest['bundle']
    if not path.is_file():
        raise FileNotFoundError(f'Missing {path}; rerun Stage 3 training')
    with path.open('rb') as stream:
        bundle = pickle.load(stream)
    if bundle.get('model_name') != name:
        raise ValueError('Manifest and saved model bundle disagree')
    return bundle


def save_plots(explainer: RiskExplainer, batch: ExplanationBatch,
               example: dict[str, Any], report_dir: Path) -> None:
    """Save a held-out beeswarm, grouped importance and one admission's top factors."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import shap
    view = shap.Explanation(values=batch.encoded_values, base_values=batch.base_values,
        data=explainer.encoded[batch.indices], feature_names=explainer.encoded_names)
    rng_state = np.random.get_state()
    try:
        np.random.seed(42)
        shap.plots.beeswarm(view, max_display=15, show=False, plot_size=(11, 8))
    finally:
        np.random.set_state(rng_state)
    plt.title(f"SHAP summary: {len(batch.indices)} held-out admissions", pad=18)
    plt.xlabel('SHAP contribution to class-1 probability')
    plt.gcf().savefig(report_dir / 'shap_summary.png', dpi=160, bbox_inches='tight')
    plt.close()
    importance = batch.grouped_values.abs().mean().sort_values(ascending=False).head(15).sort_values()
    fig, ax = plt.subplots(figsize=(11, 7), layout='constrained')
    ax.barh(importance.index, importance.values, color='#347ba9')
    ax.set(xlabel='Mean absolute grouped contribution (probability)',
           title='Global importance across held-out admissions')
    fig.savefig(report_dir / 'shap_global_importance.png', dpi=160)
    plt.close(fig)
    factors = example['top_factors']
    labels = [f"{factor['feature']} = {factor['value']:.4g}" if isinstance(factor['value'], float)
              else f"{factor['feature']} = {factor['value']}" for factor in factors]
    values = [factor['percentage_points'] for factor in factors]
    labels.append('Other features combined')
    values.append(example['other_contributions_sum'] * 100)
    fig, ax = plt.subplots(figsize=(11, 5.5), layout='constrained')
    ax.barh(labels[::-1], values[::-1], color=['#cb4b4b' if v > 0 else '#347ba9' for v in values[::-1]])
    ax.axvline(0, color='#444444', linewidth=0.8)
    ax.set(xlabel='Contribution in percentage points (relative to model baseline)',
        title=f"Admission {example['hadm_id']}: model output {example['probability']:.1%}; baseline {example['baseline_probability']:.1%}")
    fig.savefig(report_dir / 'shap_patient_top5.png', dpi=160)
    plt.close(fig)


def run_explanations(raw: dict[str, pd.DataFrame], bundle: dict[str, Any],
                     output_dir: Path, report_dir: Path,
                     hadm_id: str | None = None) -> dict[str, Any]:
    """Explain all holdout admissions and an optional specific admission."""
    report_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault('MPLCONFIGDIR', str((report_dir / '.matplotlib').resolve()))
    explainer = RiskExplainer(bundle, raw)
    if len(explainer.heldout_indices) == 0:
        raise ValueError('No held-out patients are present; supply the original Stage 3 cohort')
    batch = explainer.explain_indices(explainer.heldout_indices)
    records = explainer.records(batch)
    if hadm_id is None:
        example = records[0]  # Fixed admission order, not a chosen favorable explanation.
    else:
        matches = [r for r in records if r['hadm_id'] == str(hadm_id).strip().upper()]
        example = matches[0] if matches else explainer.explain_admission(hadm_id)
    max_error = max(record['reconstruction_error'] for record in records)
    importance = batch.grouped_values.abs().mean().sort_values(ascending=False)
    report = ['# Stage 4 explainability report', '',
        f"Model: {bundle['model_name']}; explained held-out admissions: {len(records)}.",
        f'Method: {explainer.method}.', f'Baseline: {explainer.baseline_definition}',
        f'Maximum probability reconstruction error: {max_error:.3g}.', '',
        '## Global importance', '', '| Feature | Mean absolute grouped SHAP contribution |', '|---|---:|']
    for name, value in importance.head(10).items():
        report.append(f'| {name} | {value:.6f} |')
    report += ['', 'The global summary and importance use held-out admissions only. No model, imputer, scaler or encoder is refitted.',
        'The Random Forest baseline uses only fitted training-tree statistics, including class weights; it is not the observed mortality prevalence.',
        'The beeswarm shows encoded features: numeric feature values are standardized and categorical feature values are one-hot indicators.',
        'Readable top factors and grouped importance sum the signed dummy contributions for each categorical field; this is an aggregation of encoded SHAP values, not a separately recomputed conditional group Shapley value.',
        'Numeric values in top factors use original engineered units after training-derived imputation. value_imputed flags admission-level missing-feature filling; event-level lab median filling may also contribute to a lab summary.',
        'Positive SHAP values raise the model output relative to its baseline; negative values lower it. They are not clinical causes, treatment effects or evidence that changing a value will change mortality.',
        'Magnitudes are probability contributions; multiply by 100 for percentage points. The top five alone need not reconstruct the prediction; other_contributions_sum accounts for the rest.',
        'The global ranking averages absolute values after categorical grouping, so opposing dummy contributions can cancel before averaging.',
        'The optional example is the first held-out admission unless --hadm-id is supplied; the example is not selected to make the explanation look favorable.', '', DISCLAIMER, '',
        'Whole-stay features remain retrospective and unavailable at admission time. Correlated labs can share or redistribute importance under SHAP assumptions. No medical-literature explanation is generated in Stage 4.', '',
        'Method reference: https://shap.readthedocs.io/en/latest/generated/shap.TreeExplainer.html', '']
    output_dir.mkdir(parents=True, exist_ok=True)
    identity = explainer.metadata.iloc[batch.indices].reset_index(drop=True)
    pd.concat([identity, batch.grouped_values], axis=1).to_csv(output_dir / 'grouped_shap_values.csv', index=False)
    pd.concat([identity, pd.DataFrame(batch.encoded_values, columns=explainer.encoded_names)], axis=1).to_csv(output_dir / 'encoded_shap_values.csv', index=False)
    pd.DataFrame({'feature': importance.index, 'mean_absolute_shap': importance.values}).to_csv(output_dir / 'global_importance.csv', index=False)
    (output_dir / 'admission_explanations.json').write_text(json.dumps(records, indent=2, allow_nan=False), encoding='utf-8')
    (output_dir / 'example_explanation.json').write_text(json.dumps(example, indent=2, allow_nan=False), encoding='utf-8')
    metadata = {'model': bundle['model_name'], 'method': explainer.method,
        'baseline_definition': explainer.baseline_definition, 'shap_version': importlib.metadata.version('shap'),
        'training_subject_ids': explainer.builder.train_subject_ids_,
        'reference_training_admissions': len(np.flatnonzero(explainer.training_mask)),
        'heldout_admissions': len(records), 'max_reconstruction_error': max_error,
        'encoded_feature_count': len(explainer.encoded_names), 'grouped_feature_count': len(explainer.mapping),
        'output_unit': 'class-1 probability', 'disclaimer': DISCLAIMER}
    (output_dir / 'explanation_metadata.json').write_text(json.dumps(metadata, indent=2, allow_nan=False), encoding='utf-8')
    save_plots(explainer, batch, example, report_dir)
    (report_dir / 'explainability_report.md').write_text('\n'.join(report), encoding='utf-8')
    print(f"Explained {len(records)} held-out admissions using {bundle['model_name']}.")
    print(f'Maximum probability reconstruction error: {max_error:.3g}')
    print(f'Saved explanation data to {output_dir} and plots/report to {report_dir}')
    return metadata


def main(argv: list[str] | None = None) -> int:
    """Run only Stage 4 using the trusted local selected-model bundle."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--raw-dir', type=Path, default=Path('data/raw'))
    parser.add_argument('--model-dir', type=Path, default=Path('models'))
    parser.add_argument('--output-dir', type=Path, default=Path('data/processed/stage4'))
    parser.add_argument('--report-dir', type=Path, default=Path('reports'))
    parser.add_argument('--hadm-id', help='Admission ID for the example plot; defaults to first held-out admission')
    args = parser.parse_args(argv)
    try:
        raw = load_tables(args.raw_dir)
        bundle = load_selected_bundle(args.model_dir)
        run_explanations(raw, bundle, args.output_dir, args.report_dir, args.hadm_id)
    except (FileNotFoundError, ValueError, KeyError, pd.errors.EmptyDataError) as exc:
        print(str(exc), file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
