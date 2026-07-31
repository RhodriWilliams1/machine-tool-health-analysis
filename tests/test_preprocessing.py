"""
test/test_preprocessing.py

This code contains tests for the preprocessing.py module
"""

### Imports
import pytest
import numpy as np

from machine_tool_health_analysis.preprocessing import validate_signal, scale_features
import pandas as pd


### Testing validation functionality


def test_validate_signal_valid():
    signal = np.array([1, 2, 3])

    # Should not raise an exception
    validate_signal(signal)


def test_validate_wrong_type_signal():
    signal = [1, 2, 3]

    with pytest.raises(TypeError, match="Signal must be a NumPy array."):
        validate_signal(signal)


def test_validate_signal_empty():
    signal = np.array([])

    with pytest.raises(ValueError, match="Signal is empty"):
        validate_signal(signal)


def test_validate_signal_nan():
    signal = np.array([1, np.nan, 3])

    with pytest.raises(ValueError, match="Signal contains NaN"):
        validate_signal(signal)


def test_validate_signal_infinite():
    signal = np.array([1, np.inf, 3])

    with pytest.raises(ValueError, match="Signal contains infinite"):
        validate_signal(signal)


### Testing feature scaling functionality
# Fixture


@pytest.fixture
def sample_train_test_data():
    """Provides isolated training and testing DataFrames with different distributions."""
    np.random.seed(42)

    # Train set: 100 rows, mean ~10
    X_train = pd.DataFrame(
        {
            "feature_a": np.random.normal(loc=10.0, scale=2.0, size=100),
            "feature_b": np.random.normal(loc=50.0, scale=5.0, size=100),
        }
    )

    # Test set: 30 rows, intentionally shifted mean ~20 to detect improper re-fitting
    X_test = pd.DataFrame(
        {
            "feature_a": np.random.normal(loc=20.0, scale=2.0, size=30),
            "feature_b": np.random.normal(loc=70.0, scale=5.0, size=30),
        }
    )

    return X_train, X_test


def test_scale_features(sample_train_test_data):
    X_train, X_test = sample_train_test_data
    X_train_scaled, X_test_scaled, scaler = scale_features(X_train, X_test)

    # 1. Check types and shapes
    assert isinstance(X_train_scaled, np.ndarray)
    assert X_train_scaled.shape == X_train.shape
    assert X_test_scaled.shape == X_test.shape

    # 2. Check math (X_train mean ~0, std ~1)
    np.testing.assert_allclose(X_train_scaled.mean(axis=0), 0, atol=1e-7)
    np.testing.assert_allclose(X_train_scaled.std(axis=0), 1, atol=1e-7)

    # 3. Check for data leakage (X_test transformed using train scaler)
    expected_test = (X_test.values - scaler.mean_) / scaler.scale_
    np.testing.assert_allclose(X_test_scaled, expected_test)
