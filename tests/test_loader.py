"""
test/test_loader.py

This code contains for the loader.py module
"""

import numpy as np
import pytest
from scipy.io import savemat

from machine_tool_health_analysis.loader import DataLoader


@pytest.fixture
def sample_mat_files(tmp_path):
    """Create multiple temporary MATLAB files for testing."""
    files = [
        ("Segmented_Linear_Baseline.mat", {"SpindleAccX": np.array([1, 2, 3])}),
        ("Segmented_Linear_Heavy.mat", {"Power": np.array([10, 20, 30])}),
    ]

    for filename, data in files:
        savemat(tmp_path / filename, data)

    return tmp_path


def test_load_all_with_path_object(sample_mat_files):
    """Tests loading data when data_path is a Path object."""
    loader = DataLoader(sample_mat_files)
    dataset = loader.load_all()

    assert isinstance(dataset, dict)
    assert len(dataset) == 2
    assert "Segmented_Linear_Baseline" in dataset
    assert "Segmented_Linear_Heavy" in dataset

    # Verify actual array content (flattening because scipy transforms 1D -> 2D)
    loaded_acc = dataset["Segmented_Linear_Baseline"]["SpindleAccX"].flatten()
    np.testing.assert_array_equal(loaded_acc, np.array([1, 2, 3]))


def test_load_all_with_string_path(sample_mat_files):
    """Tests loading data when data_path is passed as a string."""
    loader = DataLoader(str(sample_mat_files))
    dataset = loader.load_all()

    assert isinstance(dataset, dict)
    assert len(dataset) == 2


def test_load_all_nonexistent_directory(tmp_path):
    """Tests that a FileNotFoundError is raised if the path does not exist."""
    fake_path = tmp_path / "non_existent_folder"
    loader = DataLoader(fake_path)

    with pytest.raises(FileNotFoundError):
        loader.load_all()


def test_load_all_empty_directory(tmp_path):
    """Tests loading from an empty folder returns an empty dict."""
    loader = DataLoader(tmp_path)
    dataset = loader.load_all()

    assert dataset == {}
