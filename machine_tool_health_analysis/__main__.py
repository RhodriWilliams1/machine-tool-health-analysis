"""
This is the main file of the program
"""

### Imports
from pathlib import Path
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns


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

    # Convert feature dataset into predictors for the classification
    X, y = select_features(df)
    X_train, X_test, y_train, y_test = split_dataset(X, y)
    X_train_scaled, X_test_scaled, scaler = scale_features(X_train, X_test)

    print("Features selected, split for train and testing, and predictors normalised")

    # Training the KNN model
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X_train_scaled, y_train)

    # Making predictions
    y_pred = knn.predict(X_test_scaled)

    # Evaluate Model Performance
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy:.2f}")
    print(classification_report(y_test, y_pred))

    # Generate confusion matrix
    cm = confusion_matrix(y_test, y_pred)

    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        xticklabels=knn.classes_,
        yticklabels=knn.classes_,
        cbar=False,
    )

    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    plt.title("Confusion Matrix - KNN Classifier")

    plt.tight_layout()
    plt.savefig(Path(options.output_dir) / "confusion_matrix.png", dpi=300)
    plt.close()

    print("Saved confusion matrix.")


if __name__ == "__main__":
    main()
