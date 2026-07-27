"""
This is the main file of the program
"""

### Imports
from pathlib import Path

### Local imports
from . import cli
from .loader import DataLoader
from .feature_pipeline import build_training_dataset


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
    datasets = loader.load_all()
    # Check it has worked
    print(f"Loaded {len(datasets)} files.")

    LABELS = {
        "Segmented_Linear_Baseline": "baseline",
        "Segmented_Linear_Heavy": "heavy",
        "Segmented_Linear_Override": "override",
    }

    experiments = [(datasets[name], LABELS[name]) for name in LABELS]

    df = build_training_dataset(experiments)

    output_file = Path(options.output_dir) / "feature_dataset.csv"
    df.to_csv(output_file, index=False)

    print(f"Saved feature dataset to {output_file}")


if __name__ == "__main__":
    main()
