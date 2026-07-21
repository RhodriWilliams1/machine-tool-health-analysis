"""
This codes the command line interface
"""

### Standard library
import argparse


def command_line():
    """
    This function codes the command line interface of the program.

    Parameters
    ----------
    None

    Returns
    -------
    args : NameSpace
                Argument analysis

    """
    # Create the parser object
    parser = argparse.ArgumentParser(
        description="A tool to analyse machine tool health using sensor data."
    )

    # Add arguments to the parser
    parser.add_argument(
        "--input_file", required=True, help="Input data file provided to the model"
    )
    parser.add_argument(
        "--output_dir", required=True, help="Directory for saving results"
    )

    args = parser.parse_args()
    return args
