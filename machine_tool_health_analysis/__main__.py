"""
This is the main file of the program
"""

### Imports
from pathlib import Path

### Local imports
from . import cli
from .loader import DataLoader
from .features import select_features
from .feature_pipeline import build_training_dataset, split_dataset
from .preprocessing import scale_features


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

    X, y = select_features(df)
    X_train, X_test, y_train, y_test = split_dataset(X, y)
    X_train_scaled, X_test_scaled, scaler = scale_features(X_train, X_test)

    print("Features selected, split for train and testing, and predictors normalised")


if __name__ == "__main__":
    main()
