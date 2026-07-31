"""
features.py
"""

### Imports
import numpy as np
from scipy.stats import kurtosis


def get_peak_to_peak(signal: np.ndarray) -> float:
    """
    Calculate the peak to peak value for a signal

    Parameters
    ----------
    signal : numpy.ndarray
        The validated one-dimensional signal

    Returns
    -------
    float
        The difference between the maximum and minimum values of the signal.
    """
    return np.ptp(signal)


def get_kurtosis(signal: np.ndarray) -> float:
    """
    Calculate the kurtosis value for a signal

    Parameters
    ----------
    signal : numpy.ndarray
        The validated one-dimensional signal

    Returns
    -------
    float
        The kurtosis value of the signal.
    """
    return kurtosis(signal, fisher=True)


def get_mean(signal: np.ndarray) -> float:
    """
    Calculate the mean value for a signal

    Parameters
    ----------
    signal : numpy.ndarray
        The validated one-dimensional signal

    Returns
    -------
    float
        The mean value of the signal.
    """
    return np.mean(signal)


def select_features(df):
    """
    Select the feature matrix and target labels for model development.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing all features and labels.

    Returns
    -------
    X : pandas.DataFrame
        Feature matrix.
    y : pandas.Series
        Target labels.
    """
    X = df[["ptp_x_acc", "mean_x_acc", "kurtosis_y_acc"]]
    y = df["label"]
    return X, y
