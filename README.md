# Sensor-Based Machine Tool Health Assessment
License: MIT

## Overview

This project develops a Python-based data analysis workflow for assessing machine tool and manufacturing process health using sensor signal data.

The software is designed to process manufacturing sensor data, perform statistical analysis, and identify patterns associated with process conditions and potential defects. The project forms part of the Data Science and Research Software Project module, focusing on applying data science and software engineering techniques within a manufacturing context.

The workflow is developed using an open dataset containing sensor signals for machine tool and process health assessment.

## Features

- Load MATLAB (.mat) sensor datasets
- Preprocess manufacturing sensor signals
- Perform statistical analysis
- Generate summary outputs
- Support reproducible research workflows

## Project Status

This project is currently under active development.


## Purpose

The aim of this software is to provide a reproducible workflow for analysing sensor signals collected from manufacturing processes. The software supports:

- loading and preprocessing sensor datasets
- handling data quality issues
- performing statistical analysis
- identifying patterns associated with machine/process health
- generating outputs to support manufacturing decision-making


## Target Audience

This software is intended for:

- researchers working with manufacturing sensor data
- engineers interested in machine health monitoring
- data scientists applying statistical and machine learning methods to manufacturing problems
- students learning reproducible research software practices


## Installation

### Prerequisites
- Python 3.11
- Conda
- Git

Clone the repository:

```bash
git clone https://github.com/<username>/machine-tool-health-analysis.git
cd machine-tool-health-analysis
```

Create the Conda environment using the provided environment file:
```bash
conda env create -f environment.yml
```
Activate the environment:
```bash
conda activate machine-tool-health-analysis
```
Install the package in editable mode:
```bash
pip install -e .
```
The project is now installed and can be run using the command-line interface.


## Data

The dataset is not included in this repository due to file size and data management considerations.

This project uses the following open dataset:

**Sensor signals for machine tool and process health assessment**
Authors: Javier Alejandro Dominguez Caballero, James Moore, Jon Stammers
Dataset DOI: https://doi.org/10.15131/shef.data.24125715

The dataset contains sensor signals collected for machine tool and machining process health assessment, including simulated fault conditions such as tool imbalance, tool wear, misalignment, and surface cracks.

Download the dataset.
Extract the files into

```bash
data/raw/
```

The repository should then look like

```bash
data/
    raw/
        Segmented_Machining_Baseline.mat
        Segmented_Machining_ToolWear.mat
        ...
```

## Usage

The software can be run from the command line

Examples:
$ python -m machine_tool_health_analysis --input_dir data/raw --output_dir data/processed

| Argument       | Description                     |
| -------------- | ------------------------------- |
| `--input_dir` | Path to the input `.mat` files   |
| `--output_dir` | Directory for generated outputs |

The software generates:
- processed datasets
- summary statistics
- plots
- extracted features
- machine learning outputs

## Repository Structure
```bash
machine-tool-health-analysis/
│
├── .github/
│   └── workflows/
│       └── tests.yaml              # GitHub Actions CI workflow
│
├── data/
│   ├── raw/                        # Original sensor datasets (not tracked on GitHub)
│   ├── processed/                  # Generated feature datasets and outputs
│   └── README.md                   # Data description and download instructions
│
├── machine_tool_health_analysis/
│   ├── __init__.py
│   ├── __main__.py                 # Package entry point
│   ├── cli.py                      # Command-line interface
│   ├── data_access.py              # Data input/output handling
│   ├── loader.py                   # Loading MATLAB sensor files
│   ├── preprocessing.py            # Data cleaning and preparation
│   ├── features.py                 # Feature extraction methods
│   └── feature_pipeline.py         # End-to-end feature generation pipeline
│
├── scripts/
│   └── inspect_data.py             # Utility script for dataset inspection
│
├── tests/
│   ├── test_data_access.py
│   ├── test_features.py
│   ├── test_feature_pipeline.py
│   ├── test_loader.py
│   └── test_preprocessing.py
│
├── .gitignore
├── .pre-commit-config.yaml
├── environment.yml                 # Conda environment definition
├── LICENSE
├── pyproject.toml                  # Package configuration and dependencies
└── README.md                       # Project documentation
```


## Version control

Development is managed using Git and GitHub. Feature branches are used for new functionality before merging into the main branch.

Automated unit tests are implemented using pytest.

Run all tests with

```bash
pytest
```

## License
This project is released under the MIT License.

See the LICENSE file for further details.


## Authors and Maintainers

Author:
- Rhodri Williams

Maintainer:
- Rhodri Williams

Contact:
- rwilliams15@amrc.co.uk


## References

1. Dominguez Caballero, J. A., Moore, J., & Stammers, J. (2023).
   *Sensor signals for machine tool and process health assessment.*
   The University of Sheffield Dataset.
   https://doi.org/10.15131/shef.data.24125715.v1
