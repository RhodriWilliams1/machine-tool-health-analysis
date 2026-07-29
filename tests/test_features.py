"""
test/test_features.py

This code contains tests for the features.py module
"""

### Imports
import pytest
from scipy.stats import kurtosis
import pandas as pd
import numpy as np

from machine_tool_health_analysis.features import (
    get_peak_to_peak,
    get_mean,
    get_kurtosis,
    select_features,
)


# --- Fixtures ---


@pytest.fixture
def sample_data():
    """Provides a synthetic DataFrame matching the expected schema."""
    np.random.seed(42)
    return pd.DataFrame(
        {
            "ptp_x_acc": np.random.randn(100),
            "mean_x_acc": np.random.randn(100),
            "kurtosis_y_acc": np.random.randn(100),
            "unrelated_feature": np.random.randn(100),  # Should be filtered out
            "label": np.random.choice([0, 1], size=100),
        }
    )


# --- Tests for feature generation ---


def test_get_peak_to_peak():
    signal = np.array([1, 5, 2, -1])
    assert get_peak_to_peak(signal) == pytest.approx(6)


def test_get_mean():
    signal = np.array([1, 2, 3, 4])
    assert get_mean(signal) == pytest.approx(2.5)


def test_get_kurtosis():
    signal = np.array([1, 2, 3, 4, 5])
    assert get_kurtosis(signal) == pytest.approx(kurtosis(signal, fisher=True))


# --- Tests for select_features ---


def test_select_features_output_shapes_and_types(sample_data):
    """Verify select_features extracts the exact columns and correct types."""
    X, y = select_features(sample_data)

    expected_cols = ["ptp_x_acc", "mean_x_acc", "kurtosis_y_acc"]

    # Type assertions
    assert isinstance(X, pd.DataFrame)
    assert isinstance(y, pd.Series)

    # Content assertions
    assert list(X.columns) == expected_cols
    assert y.name == "label"
    assert len(X) == len(sample_data)


def test_select_features_missing_column():
    """Verify select_features raises a KeyError if a required feature is missing."""
    incomplete_df = pd.DataFrame(
        {
            "ptp_x_acc": [1.0, 2.0],
            "label": [0, 1],
            # Missing 'mean_x_acc' and 'kurtosis_y_acc'
        }
    )

    with pytest.raises(KeyError):
        select_features(incomplete_df)
