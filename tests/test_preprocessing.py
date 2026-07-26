"""
test/test_preprocessing.py

This code contains tests for the preprocessing.py module
"""

### Imports
import pytest
import numpy as np

from machine_tool_health_analysis.preprocessing import validate_signal


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
