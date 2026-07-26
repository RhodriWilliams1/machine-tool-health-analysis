"""
test/test_data_access.py

This code contains tests for the data_access.py module
"""

### Imports
import pytest
import numpy as np

from machine_tool_health_analysis.data_access import get_signal


@pytest.fixture
def sample_data():
    return {
        "Segmented_Linear_Baseline": {
            "SpindleAccX": [np.array([1, 2, 3]), np.array([4, 5, 6])]
        }
    }


def test_get_signal_returns_correct_segment(sample_data):
    signal = get_signal(sample_data, "SpindleAccX", 0)

    assert isinstance(signal, np.ndarray)
    assert signal.shape == (3,)
    np.testing.assert_array_equal(signal, np.array([1, 2, 3]))

    signal2 = get_signal(sample_data, "SpindleAccX", 1)

    assert isinstance(signal2, np.ndarray)
    assert signal2.shape == (3,)
    np.testing.assert_array_equal(signal2, np.array([4, 5, 6]))


def test_get_signal_empty_dataset():
    with pytest.raises(ValueError, match="Dataset is empty"):
        get_signal({}, "SpindleAccX", 0)


def test_get_signal_invalid_signal_name(sample_data):
    with pytest.raises(KeyError):
        get_signal(sample_data, "NotARealSignal", 0)


def test_get_signal_segment_type(sample_data):
    with pytest.raises(TypeError):
        get_signal(sample_data, "SpindleAccX", "zero")


def test_get_signal_invalid_segment(sample_data):
    with pytest.raises(IndexError):
        get_signal(sample_data, "SpindleAccX", 999)
