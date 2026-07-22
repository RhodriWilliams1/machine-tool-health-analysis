"""
loader.py

This codes the data loading functionality
"""

### Imports
from pathlib import Path
from scipy.io import loadmat
from typing import Union


class DataLoader:
    """
    Loads and stores the raw signal data from a folder of .mat files.
    """

    def __init__(self, data_path: Union[str, Path]):
        """
        Constructor

        Parameters
        ----------
        data_path : str
            Path to the folder containing the data

        Returns
        -------
        None
        """

        self.data_path = Path(data_path)

    def load_all(self):
        """
        Finds all the .mat files and loads each one, returning a dictionary of
        the complete data source

        Parameters
        ----------
        None

        Returns
        -------
        dataset : dict
            Dictionary containing data from all .mat files keyed by file stem.

        Raises
        ------
        FileNotFoundError
            If data_path does not exist.
        NotADirectoryError
            If data_path is not a directory.
        """

        # Raise errors for incomplete file structures
        if not self.data_path.exists():
            raise FileNotFoundError(f"Path does not exist: {self.data_path}")
        if not self.data_path.is_dir():
            raise NotADirectoryError(f"Path is not a directory: {self.data_path}")

        dataset = {}

        for file in self.data_path.glob("*.mat"):
            dataset[file.stem] = loadmat(file)

        return dataset
