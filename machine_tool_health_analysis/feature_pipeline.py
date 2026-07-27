"""
feature_pipeline.py
"""

### Import
import pandas as pd

### Local Imports
from machine_tool_health_analysis.data_access import get_signal
from machine_tool_health_analysis.preprocessing import validate_signal
from machine_tool_health_analysis.features import (
    get_peak_to_peak,
    get_mean,
    get_kurtosis,
)


def create_feature_dataset(data: dict, label: str) -> pd.DataFrame:
    """
    Create the feature dataset required by the classifier

    Parameters
    ----------
    data : dict
        Dictionary returned by ``DataLoader.load_file()`` containing the
        loaded machine tool dataset.

    Returns
    -------
    pandas.DataFrame
        A dataframe containing the extracted features for each run of a
        simulated test, together with the corresponding class label.

    Raises
    ------

    """

    # Access dataset top level name
    dataset = list(data.values())[0]
    # Create list to build the data frame
    rows = []

    for segment_id in range(len(dataset["RunNo"])):
        # Access X-axis spindle acceleration signal
        # and validate signal is is the right format for feature extraction
        x_acc = get_signal(
            data=data,
            signal_name="SpindleAccX",
            segment_id=segment_id,
        )
        validate_signal(x_acc)

        # Access Y-axis spindle acceleration signal
        # and validate signal is is the right format for feature extraction
        y_acc = get_signal(
            data=data,
            signal_name="SpindleAccY",
            segment_id=segment_id,
        )
        validate_signal(y_acc)

        # Collect measures
        rows.append(
            {
                "run": dataset["RunNo"][segment_id],
                "ptp_x_acc": get_peak_to_peak(x_acc),
                "mean_x_acc": get_mean(x_acc),
                "kurtosis_y_acc": get_kurtosis(y_acc),
                "label": label,
            }
        )

    return pd.DataFrame(rows)


def combine_feature_datasets(datasets: list[pd.DataFrame]) -> pd.DataFrame:
    """
    Combine a list of dataframes into a single dataset

    Parameters
    ----------
    datasets : list[pd.DataFrame]
        A list of DataFrames each containing the extracted features for each
        run of an simulated test, together with the corresponding class label.

    Returns
    -------
    pandas.DataFrame
        A dataframe containing the extracted features for each run of all
        simulated tests, together with the corresponding class label.

    Raises
    ------
    ValueError
        If the dataset is empty.
    """
    if not datasets:
        raise ValueError("No datasets provided.")

    return pd.concat(datasets, ignore_index=True)


def build_training_dataset(experiments: list[tuple[dict, str]]) -> pd.DataFrame:
    """
    Build the complete classifier dataset from multiple experiments.

    Parameters
    ----------
    experiments : list[tuple[dict, str]]
        A list where each element contains a tuple with the data stored in a
        dict and the corresponding label for the experiment stored as a string

    Returns
    -------
    pandas.DataFrame
        A dataframe containing the extracted features for each run of all
        simulated tests, together with the corresponding class label.

    Raises
    ------

    """

    feature_tables = []

    for data, label in experiments:
        feature_tables.append(create_feature_dataset(data, label))

    return combine_feature_datasets(feature_tables)
