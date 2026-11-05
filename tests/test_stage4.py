"""SHAP tests use isolated fixtures, never a replacement clinical dataset."""
import copy
import json
from types import SimpleNamespace
import numpy as np
import pytest
from test_stage2 import tables
from test_stage3 import metadata, fitted
from src.explain import (RiskExplainer, binary_shap_arrays, load_selected_bundle,
                         main, run_explanations)


@pytest.fixture
def bundle(fitted):
    # Use the frozen model and feature state from the Stage 3 unit fixture.
    pipe = fitted.pipelines['random_forest']
    return {'version': 1, 'model_name': 'random_forest', 'pipeline': pipe,
        'feature_state': fitted.builder.state(), 'encoded_feature_names': fitted.encoded_feature_names}


@pytest.fixture
def explainer(bundle, tables):
    return RiskExplainer(bundle, tables)


def test_binary_class_one_shape_is_selected():
    values = np.array([[[1.0, -1.0], [2.0, -2.0]], [[3.0, -3.0], [4.0, -4.0]]])
    result, base = binary_shap_arrays(SimpleNamespace(values=values, base_values=np.array([[0.2, 0.8], [0.2, 0.8]])), 2, 2)
    np.testing.assert_array_equal(result, [[-1, -2], [-3, -4]])
    np.testing.assert_array_equal(base, [0.8, 0.8])


def test_scalar_positive_probability_shape():
    result, base = binary_shap_arrays(SimpleNamespace(values=np.ones((2, 3)), base_values=0.5), 2, 3)
    assert result.shape == (2, 3)
    np.testing.assert_array_equal(base, [0.5, 0.5])


@pytest.mark.parametrize('shape', [(2, 4), (2, 3, 3)])
def test_unexpected_shap_shape_rejected(shape):
    with pytest.raises(ValueError, match='Unexpected SHAP'):
        binary_shap_arrays(SimpleNamespace(values=np.ones(shape), base_values=0.5), 2, 3)


def test_nonfinite_values_rejected():
    with pytest.raises(ValueError, match='nonfinite'):
        binary_shap_arrays(SimpleNamespace(values=np.full((1, 2), np.nan), base_values=0.5), 1, 2)


def test_additivity_grouping_and_train_only_reference(explainer):
    batch = explainer.explain_indices([0, 1, 2, 3])
    np.testing.assert_allclose(batch.base_values + batch.encoded_values.sum(axis=1), batch.probabilities, atol=1e-6)
    np.testing.assert_allclose(batch.grouped_values.sum(axis=1), batch.encoded_values.sum(axis=1), atol=1e-12)
    assert not set(explainer.background_indices) & set(explainer.heldout_indices)
    assert explainer.heldout_indices.tolist() == [3]
    assert 'tree_path_dependent' in explainer.method
    assert set(batch.grouped_values) == set(explainer.builder.feature_columns_)


def test_categorical_mapping_covers_each_dummy_once(explainer):
    positions = [position for group in explainer.mapping.values() for position in group]
    assert sorted(positions) == list(range(len(explainer.encoded_names)))
    for name in ['gender', 'insurance', 'admission_type']:
        assert all(explainer.encoded_names[index].startswith(name + '_') for index in explainer.mapping[name])


def test_top_five_format_values_and_sorting(explainer):
    record = explainer.explain_admission('c1')
    assert record['hadm_id'] == 'C1' and record['scope'] == 'heldout'
    factors = record['top_factors']
    assert len(factors) == 5 and len({f['feature'] for f in factors}) == 5
    assert [f['magnitude'] for f in factors] == sorted([f['magnitude'] for f in factors], reverse=True)
    for factor in factors:
        assert {'feature', 'value', 'direction', 'magnitude'} <= set(factor)
        assert factor['direction'] in {'raises', 'lowers', 'neutral'}
        assert factor['magnitude'] == abs(factor['shap_value'])
        assert factor['percentage_points'] == factor['shap_value'] * 100
    assert record['baseline_probability'] + sum(f['shap_value'] for f in factors) + record['other_contributions_sum'] == pytest.approx(record['probability'])
    assert 'not medical advice' in record['disclaimer']
    json.dumps(record, allow_nan=False)


def test_top_factor_signs_and_original_units(explainer):
    batch = explainer.explain_indices([3])
    batch.grouped_values.loc[0, :] = 0.0
    batch.grouped_values.loc[0, 'length_of_stay_days'] = 0.25
    batch.grouped_values.loc[0, 'age'] = -0.1
    factors = explainer.top_factors(batch, 0)
    assert factors[0]['feature'] == 'length_of_stay_days'
    assert factors[0]['value'] == pytest.approx(100.0)
    assert factors[0]['direction'] == 'raises'
    assert factors[1]['value'] == pytest.approx(50.0)
    assert factors[1]['direction'] == 'lowers'


def test_imputed_aggregate_is_marked(bundle, tables):
    altered = copy.deepcopy(tables)
    altered['labevents'] = altered['labevents'].query("hadm_id != 'c1'").copy()
    explainer = RiskExplainer(bundle, altered)
    batch = explainer.explain_indices([3])
    batch.grouped_values.loc[0, :] = 0.0
    batch.grouped_values.loc[0, 'lab_x_test_lab_mean'] = 0.2
    factor = explainer.top_factors(batch, 0)[0]
    assert factor['value_imputed'] is True
    assert factor['value'] == pytest.approx(explainer.builder.numeric_medians_['lab_x_test_lab_mean'])


def test_unknown_admission_and_invalid_indices_rejected(explainer):
    with pytest.raises(ValueError, match='Unknown'):
        explainer.explain_admission('missing')
    for indices in [[], [0, 0], [-1], [10], [1.5]]:
        with pytest.raises(ValueError):
            explainer.explain_indices(indices)


def test_no_predictor_refitting_or_state_changes(explainer):
    before = json.dumps(explainer.builder.state(), sort_keys=True)
    first = explainer.explain_admission('C1')
    second = explainer.explain_admission('C1')
    assert first == second
    assert json.dumps(explainer.builder.state(), sort_keys=True) == before


def test_incorrect_feature_names_fail(bundle, tables):
    bundle['encoded_feature_names'] = ['incorrect']
    with pytest.raises(ValueError, match='feature names'):
        RiskExplainer(bundle, tables)


def test_missing_training_patients_rejected(bundle, tables):
    altered = copy.deepcopy(bundle)
    altered['feature_state']['train_subject_ids'].append('MISSING')
    with pytest.raises(ValueError, match='training patients'):
        RiskExplainer(altered, tables)


@pytest.mark.parametrize('model_name', ['logistic_regression', 'xgboost'])
def test_probability_fallback(model_name, fitted, tables):
    bundle = {'version': 1, 'model_name': model_name,
        'pipeline': fitted.pipelines[model_name],
        'feature_state': fitted.builder.state(), 'encoded_feature_names': fitted.encoded_feature_names}
    explainer = RiskExplainer(bundle, tables)
    record = explainer.explain_admission('C1')
    assert 'Permutation' in explainer.method
    assert record['reconstruction_error'] < 1e-6


def test_missing_model_and_manifest_paths(tmp_path):
    with pytest.raises(FileNotFoundError, match='src.train'):
        load_selected_bundle(tmp_path)
    (tmp_path / 'selected_model.json').write_text(json.dumps({'selected_model': 'random_forest', 'bundle': '../bad.pkl'}))
    with pytest.raises(ValueError, match='manifest'):
        load_selected_bundle(tmp_path)


def test_cli_missing_data_no_outputs(tmp_path, capsys):
    out = tmp_path / 'out'
    assert main(['--raw-dir', str(tmp_path / 'missing'), '--output-dir', str(out)]) == 1
    assert 'Missing MIMIC-III' in capsys.readouterr().err
    assert not out.exists()


def test_explanation_pipeline_outputs(bundle, tables, tmp_path):
    result = run_explanations(tables, bundle, tmp_path / 'out', tmp_path / 'reports')
    assert result['heldout_admissions'] == 1
    assert result['max_reconstruction_error'] < 1e-6
    records = json.loads((tmp_path / 'out/admission_explanations.json').read_text())
    assert len(records) == 1 and records[0]['hadm_id'] == 'C1'
    for file in ['shap_summary.png', 'shap_global_importance.png', 'shap_patient_top5.png']:
        assert (tmp_path / 'reports' / file).read_bytes().startswith(b'\x89PNG')


def test_failed_probability_reconstruction_is_rejected(explainer):
    def incorrect(matrix, check_additivity):
        return SimpleNamespace(values=np.zeros((len(matrix), matrix.shape[1])),
                               base_values=np.full(len(matrix), 0.99999))
    explainer.explainer = incorrect
    with pytest.raises(ValueError, match='do not reconstruct'):
        explainer.explain_indices([3])
