"""
test/test_features.py

This code contains tests for the features.py module
"""

### Imports
import pytest
from scipy.stats import kurtosis
import numpy as np

from machine_tool_health_analysis.features import (
    get_peak_to_peak,
    get_mean,
    get_kurtosis,
)


def test_get_peak_to_peak():
    signal = np.array([1, 5, 2, -1])
    assert get_peak_to_peak(signal) == pytest.approx(6)


def test_get_mean():
    signal = np.array([1, 2, 3, 4])
    assert get_mean(signal) == pytest.approx(2.5)


def test_get_kurtosis():
    signal = np.array([1, 2, 3, 4, 5])
    assert get_kurtosis(signal) == pytest.approx(kurtosis(signal, fisher=True))
