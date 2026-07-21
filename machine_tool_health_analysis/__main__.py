"""
This is the main file of the program
"""

### Local imports
from . import cli

if __name__ == "__main__":
    # Call the command line interface
    options = cli.command_line()

    print("Machine tool health analysis...")
    print(f"Input file: {options.input_file}")
    print(f"Output directory: {options.output_dir}")

    print(options)
