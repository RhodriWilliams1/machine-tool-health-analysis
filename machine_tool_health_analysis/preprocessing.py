"""
preprocessing.py
"""

# Imports
import numpy as np


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
