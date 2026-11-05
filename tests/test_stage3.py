"""Stage 3 tests use small isolated fixtures, not substitute datasets."""
import copy
import json
import pickle
import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal
from test_stage2 import tables
from src.features import AdmissionFeatureBuilder, prepare_tables
from src.train import (MODEL_NAMES, METRICS, auditable_parameters, evaluate_predictions,
                       fit_partition, grouped_cv, grouped_holdout, main, make_models,
                       prediction_rows, save_plots, select_model, summarize_cv)


@pytest.fixture
def metadata(tables):
    return prepare_tables(tables)['admissions'][['subject_id', 'hadm_id', 'hospital_expire_flag']].reset_index(drop=True)


@pytest.fixture
def fitted(tables, metadata):
    return fit_partition(tables, metadata, np.array([0, 1, 2]), np.array([3]), trees=4)


def test_known_metrics_and_confusion():
    metrics = evaluate_predictions(np.array([0, 0, 1, 1]), np.array([0.1, 0.6, 0.8, 0.4]))
    assert metrics['roc_auc'] == pytest.approx(0.75)
    assert metrics['pr_auc_ap'] == pytest.approx(5/6)
    assert metrics['precision'] == metrics['recall'] == metrics['f1'] == 0.5
    assert metrics['confusion_matrix'] == [[1, 1], [1, 1]]


@pytest.mark.parametrize('labels', [[0, 0], [1, 1]])
def test_single_class_ranking_metrics_explicitly_undefined(labels):
    metrics = evaluate_predictions(np.array(labels), np.array([0.2, 0.8]))
    assert metrics['roc_auc'] is None and metrics['pr_auc_ap'] is None
    assert len(metrics['confusion_matrix']) == 2
    json.dumps(metrics, allow_nan=False)


@pytest.mark.parametrize('scores', [[float('nan')], [float('inf')], [-0.1], [1.1]])
def test_invalid_probabilities_rejected(scores):
    with pytest.raises(ValueError, match='finite probabilities'):
        evaluate_predictions(np.array([0]), np.array(scores))


def test_fractional_labels_rejected():
    with pytest.raises(ValueError, match='Binary outcomes'):
        evaluate_predictions(np.array([0.5]), np.array([0.2]))


def test_threshold_is_fixed_at_point_five():
    metrics = evaluate_predictions(np.array([0, 1]), np.array([0.49, 0.5]))
    assert metrics['confusion_matrix'] == [[1, 0], [0, 1]]


def test_grouped_holdout_reproducible_and_disjoint():
    # Abstract group/label vectors only, not clinical records.
    meta = pd.DataFrame({'subject_id': [str(i // 2) for i in range(40)],
                         'hospital_expire_flag': [0, 1] * 20})
    train, test = grouped_holdout(meta)
    other_train, other_test = grouped_holdout(meta)
    np.testing.assert_array_equal(train, other_train)
    np.testing.assert_array_equal(test, other_test)
    assert not set(meta.iloc[train].subject_id) & set(meta.iloc[test].subject_id)
    assert len(set(train) | set(test)) == len(meta)


def test_five_cv_folds_cover_training_only_without_patient_overlap():
    meta = pd.DataFrame({'subject_id': [str(i // 2) for i in range(40)],
                         'hospital_expire_flag': [0, 1] * 20})
    train, holdout = grouped_holdout(meta)
    folds = grouped_cv(meta, train)
    assert len(folds) == 5
    validation = []
    for fit, valid in folds:
        assert not set(meta.iloc[fit].subject_id) & set(meta.iloc[valid].subject_id)
        assert not set(fit) & set(holdout) and not set(valid) & set(holdout)
        validation.extend(valid)
    assert sorted(validation) == sorted(train)


def test_infeasible_cv_fails_instead_of_reducing_folds(metadata):
    with pytest.raises(ValueError, match='Insufficient'):
        grouped_cv(metadata, np.arange(len(metadata)))


def test_single_class_training_rejected():
    meta = pd.DataFrame({'subject_id': ['a', 'b', 'c'], 'hospital_expire_flag': [0, 0, 0]})
    with pytest.raises(ValueError, match='one outcome class'):
        grouped_holdout(meta)


def test_models_balance_current_partition_only():
    models = make_models(pd.Series([0, 0, 0, 1]), trees=4)
    assert models['logistic_regression'].class_weight == 'balanced'
    assert models['random_forest'].class_weight == 'balanced_subsample'
    assert models['xgboost'].get_params()['scale_pos_weight'] == 3
    assert all(model.get_params()['random_state'] == 42 for model in models.values())
    for model in models.values():
        json.dumps(auditable_parameters(model), allow_nan=False)


def test_partition_encoding_is_training_only_and_unknown_categories_work(fitted):
    assert fitted.builder.train_subject_ids_ == ['A', 'B']
    assert fitted.builder.lab_items_ == ['X']
    encoder = fitted.pipelines['logistic_regression'].named_steps['encoding'].named_transformers_['categorical']
    assert 'Z' not in encoder.categories_[2]
    for name in MODEL_NAMES:
        assert fitted.probabilities[name].shape == (1,)
        assert np.isfinite(fitted.probabilities[name]).all()
    assert not any(name in fitted.encoded_feature_names for name in ['subject_id', 'hadm_id', 'hospital_expire_flag', 'dod', 'discharge_location'])


def test_heldout_measurements_do_not_change_trained_models_or_state(tables, metadata, fitted):
    changed = copy.deepcopy(tables)
    changed['labevents'].loc[changed['labevents'].subject_id.eq('c'), 'valuenum'] = -1e6
    changed['admissions'].loc[3, 'dischtime'] = '2150-01-01'
    other = fit_partition(changed, metadata, np.array([0, 1, 2]), np.array([3]), trees=4)
    assert other.builder.state() == fitted.builder.state()
    original_x = fitted.builder.transform(tables).iloc[:3]
    changed_x = other.builder.transform(changed).iloc[:3]
    assert_frame_equal(original_x, changed_x)
    for name in MODEL_NAMES:
        np.testing.assert_allclose(fitted.pipelines[name].predict_proba(original_x),
                                   other.pipelines[name].predict_proba(changed_x))


def test_patient_overlap_rejected(tables, metadata):
    with pytest.raises(ValueError, match='must not overlap'):
        fit_partition(tables, metadata, np.array([0, 2]), np.array([1, 3]), trees=4)


def test_misaligned_metadata_rejected(tables, metadata):
    with pytest.raises(ValueError, match='not aligned'):
        fit_partition(tables, metadata.iloc[::-1].reset_index(drop=True), np.array([0, 1, 2]), np.array([3]), trees=4)


def test_fitted_bundle_pickle_roundtrip(fitted, tables):
    restored = pickle.loads(pickle.dumps({'state': fitted.builder.state(), 'pipelines': fitted.pipelines}))
    builder = AdmissionFeatureBuilder.from_state(restored['state'])
    features = builder.transform(tables)
    for name in MODEL_NAMES:
        np.testing.assert_allclose(restored['pipelines'][name].predict_proba(features),
                                   fitted.pipelines[name].predict_proba(features))


def test_prediction_ids_align_with_probabilities(metadata, fitted):
    rows = prediction_rows(metadata, np.array([3]), fitted, fold=1)
    assert len(rows) == 3
    assert {row['hadm_id'] for row in rows} == {'C1'}
    assert {row['y_true'] for row in rows} == {1}
    assert {row['fold'] for row in rows} == {1}


def test_cv_averages_and_selection_uses_cv_ap():
    rows = []
    for fold in [1, 2]:
        for index, name in enumerate(MODEL_NAMES):
            row = {'fold': fold, 'model': name, **{metric: 0.5 + index * 0.1 for metric in METRICS}}
            rows.append(row)
    summary = summarize_cv(rows)
    assert summary['logistic_regression']['pr_auc_ap']['mean'] == 0.5
    assert summary['logistic_regression']['pr_auc_ap']['std'] == 0
    assert select_model(summary) == 'xgboost'
    assert list(summary) == MODEL_NAMES


def test_cli_missing_data_has_actionable_failure(tmp_path, capsys):
    output = tmp_path / 'out'
    assert main(['--raw-dir', str(tmp_path / 'raw'), '--output-dir', str(output)]) == 1
    assert 'Missing MIMIC-III' in capsys.readouterr().err
    assert not output.exists()


def test_plots_created_for_known_predictions(tmp_path):
    frame = pd.DataFrame([{'model': name, 'hadm_id': str(i), 'y_true': y,
        'probability': p, 'predicted': int(p >= 0.5)}
        for name in MODEL_NAMES for i, (y, p) in enumerate([(0, 0.1), (1, 0.9)])])
    save_plots(frame, tmp_path)
    for name in ['roc_curves.png', 'precision_recall_curves.png', 'confusion_matrices.png']:
        assert (tmp_path / name).read_bytes().startswith(b'\x89PNG')
