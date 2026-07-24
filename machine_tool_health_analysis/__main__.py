"""
This is the main file of the program
"""

### Local imports
from . import cli
from .loader import DataLoader


def main():
    """
    Run the application

    Returns
    -------
    None.
    """
    ### Call the command line interface
    options = cli.command_line()
    # Tell the user the application has started
    print("Machine tool health analysis...")

    ### Load the data
    loader = DataLoader(options.input_dir)
    dataset = loader.load_all()
    # Check it has worked
    print(f"Loaded {len(dataset)} files.")


if __name__ == "__main__":
    main()
