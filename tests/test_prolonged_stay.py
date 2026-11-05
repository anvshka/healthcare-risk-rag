"""Tests for the optional secondary prolonged-stay outcome; fixtures are in-memory only."""
import pandas as pd
import pytest

from test_stage2 import tables
from src.prolonged_stay import EXCLUDED_FEATURE, OUTCOME_NAME, outcome_metadata


def test_outcome_uses_seven_day_threshold_and_keeps_ids(tables):
    metadata = outcome_metadata(tables)
    assert metadata.columns.tolist() == ['subject_id', 'hadm_id', OUTCOME_NAME]
    assert metadata[OUTCOME_NAME].tolist() == [0, 0, 0, 1]
    assert EXCLUDED_FEATURE == 'length_of_stay_days'


def test_outcome_rejects_missing_discharge_time(tables):
    tables['admissions'].loc[0, 'dischtime'] = pd.NA
    with pytest.raises(ValueError, match='valid non-negative'):
        outcome_metadata(tables)
