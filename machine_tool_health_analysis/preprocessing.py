"""
preprocessing.py
"""

# Imports
import numpy as np
from sklearn.preprocessing import StandardScaler


def validate_signal(signal: np.ndarray):
    """
    Check whether a signal is suitable for feature extraction.

    Parameters
    ----------
    numpy.ndarray
        The signal for the specified segment as a one-dimensional
        NumPy array to be validated.

    Returns
    -------


    Raises
    ------
    TypeError
        If the signal is not a NumPy array
    ValueError
        If the signal is empty.
        If the signal contains NaN
        If the signal contains infinite values
    """

    if not isinstance(signal, np.ndarray):
        raise TypeError("Signal must be a NumPy array.")

    if signal.size == 0:
        raise ValueError("Signal is empty.")

    if np.isnan(signal).any():
        raise ValueError("Signal contains NaN values.")

    if np.isinf(signal).any():
        raise ValueError("Signal contains infinite values.")


def scale_features(X_train, X_test):
    """
    Standardise predictor variables using z-score normalisation.

    The scaler is fitted using the training data only and is then used to
    transform both the training and testing datasets.

    Parameters
    ----------
    X_train : pandas.DataFrame
        Training predictor variables.
    X_test : pandas.DataFrame
        Testing predictor variables.

    Returns
    -------
    X_train_scaled : numpy.ndarray
        Standardised training predictor variables.
    X_test_scaled : numpy.ndarray
        Standardised testing predictor variables.
    scaler : sklearn.preprocessing.StandardScaler
        Fitted scaler used to transform the datasets.

    """
    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, scaler
