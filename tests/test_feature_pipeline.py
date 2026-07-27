"""
test/test_feature_pipeline.py

This code contains tests for the feature_pipeline.py module
"""

### Imports
import pytest
import numpy as np
import pandas as pd

from machine_tool_health_analysis.feature_pipeline import (
    create_feature_dataset,
    combine_feature_datasets,
    build_training_dataset,
)


### Testing the generation of a feature dataset for a single experiment


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


### Testing for combining datasets


def test_combine_feature_datasets():
    df1 = pd.DataFrame(
        {
            "run": [1, 2],
            "ptp_x_acc": [10, 20],
            "label": ["baseline", "baseline"],
        }
    )

    df2 = pd.DataFrame(
        {
            "run": [1, 2],
            "ptp_x_acc": [30, 40],
            "label": ["heavy", "heavy"],
        }
    )

    result = combine_feature_datasets([df1, df2])

    assert len(result) == 4
    assert list(result["label"]) == [
        "baseline",
        "baseline",
        "heavy",
        "heavy",
    ]


def test_combine_feature_datasets_resets_index():
    df1 = pd.DataFrame({"value": [1, 2]}, index=[5, 6])
    df2 = pd.DataFrame({"value": [3, 4]}, index=[10, 11])

    result = combine_feature_datasets([df1, df2])

    assert list(result.index) == [0, 1, 2, 3]


### Testing the build traning dataset functionality


def test_build_training_dataset(monkeypatch):
    def mock_create_feature_dataset(data, label):
        return pd.DataFrame(
            {
                "run": [1],
                "label": [label],
            }
        )

    monkeypatch.setattr(
        "machine_tool_health_analysis.feature_pipeline.create_feature_dataset",
        mock_create_feature_dataset,
    )

    experiments = [
        ({"dummy": "data"}, "baseline"),
        ({"dummy": "data"}, "heavy"),
    ]

    result = build_training_dataset(experiments)

    assert len(result) == 2
    assert list(result["label"]) == [
        "baseline",
        "heavy",
    ]


def test_combine_feature_datasets_empty():
    with pytest.raises(ValueError):
        combine_feature_datasets([])
