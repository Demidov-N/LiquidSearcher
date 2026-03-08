import pytest
import pandas as pd

from src.training.validator import CrossRegimeValidator


def test_validator_initialization():
    """Test validator can be initialized."""
    validator = CrossRegimeValidator()
    assert validator is not None
    assert hasattr(validator, "val_folds")


def test_validator_date_ranges():
    """Test validator has correct date ranges for folds."""
    validator = CrossRegimeValidator()

    # Check training range
    assert validator.train_start == "2010-01-01"
    assert validator.train_end == "2018-12-31"

    # Check validation folds
    assert len(validator.val_folds) == 3

    # First fold: COVID
    assert validator.val_folds[0].name == "COVID Crash + Recovery"
    assert validator.val_folds[0].start == "2020-01-01"
    assert validator.val_folds[0].end == "2020-12-31"

    # Check test set
    assert validator.test_start == "2024-02-01"
    assert validator.test_end == "2024-12-31"


def test_validator_no_overlap():
    """Test that splits don't overlap (no data leakage)."""
    validator = CrossRegimeValidator()

    # This should not raise
    assert validator.validate_no_overlap() is True
