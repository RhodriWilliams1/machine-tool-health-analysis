"""
test/test_feature_pipeline.py

This code contains tests for the feature_pipeline.py module
"""

### Imports
import pytest
import numpy as np
import pandas as pd

from machine_tool_health_analysis.feature_pipeline import create_feature_dataset


@pytest.fixture
def sample_data():
    return {
        "Segmented_Linear_Baseline": {
            "RunNo": [np.array([1.0]), np.array([2.0])],
            "SpindleAccX": [np.array([1, 2, 3]), np.array([4, 5, 6])],
            "SpindleAccY": [np.array([2, 4, 6]), np.array([1, 3, 5])],
        }
    }


def test_create_feature_dataset_returns_dataframe(sample_data):
    df = create_feature_dataset(sample_data, "Baseline")

    assert isinstance(df, pd.DataFrame)


def test_create_feature_dataset_columns(sample_data):
    df = create_feature_dataset(sample_data, "Baseline")

    assert list(df.columns) == [
        "run",
        "ptp_x_acc",
        "mean_x_acc",
        "kurtosis_y_acc",
        "label",
    ]


def test_create_feature_dataset_number_of_rows(sample_data):
    df = create_feature_dataset(sample_data, "Baseline")

    assert len(df) == 2


def test_create_feature_dataset_label(sample_data):
    df = create_feature_dataset(sample_data, "Baseline")

    assert all(df["label"] == "Baseline")
