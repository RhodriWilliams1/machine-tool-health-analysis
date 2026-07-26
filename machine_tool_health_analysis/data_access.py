"""
data_access.py
"""


def get_signal(data: dict, signal_name: str, segment_id: int):
    """
    Retrieve a single signal for a specific machining segment.

    Parameters
    ----------
    data : dict
        Dictionary returned by ``DataLoader.load_file()`` containing the
        loaded machine tool dataset.
    signal_name : str
        Name of the signal to retrieve (e.g. ``"SpindleAccX"``,
        ``"PlateHFAccZ"``, or ``"Power"``).
    segment_id : int
        Zero-based index of the machining segment to retrieve.

    Returns
    -------
    numpy.ndarray
        The requested signal for the specified segment as a one-dimensional
        NumPy array.

    Raises
    ------
    ValueError
        If the dataset is empty.
    KeyError
        If the requested signal does not exist.
    IndexError
        If the segment ID is outside the available range.
    TypeError
        If segment_id is not an integer.

    """

    # Check dataset exists
    if not data:
        raise ValueError("Dataset is empty.")

    # Access dataset top level name
    dataset = list(data.values())[0]

    # Check signal exists
    if signal_name not in dataset:
        available_signals = list(dataset.keys())
        raise KeyError(
            f"Signal '{signal_name}' not found. "
            f"Available signals: {available_signals}"
        )

    # Check segment ID type
    if not isinstance(segment_id, int):
        raise TypeError(
            f"segment_id must be an integer, got {type(segment_id).__name__}"
        )

    # Check segment exists
    if segment_id < 0 or segment_id >= len(dataset[signal_name]):
        raise IndexError(
            f"Segment ID {segment_id} out of range. "
            f"Available segments: 0-{len(dataset[signal_name])-1}"
        )

    return dataset[signal_name][segment_id]
