# Sensor-Based Machine Tool Health Assessment

## Overview

This project develops a Python-based data analysis workflow for assessing machine tool and manufacturing process health using sensor signal data.

The software is designed to process manufacturing sensor data, perform statistical analysis, and identify patterns associated with process conditions and potential defects. The project forms part of the Data Science and Research Software Project module, focusing on applying data science and software engineering techniques within a manufacturing context.

The workflow is developed using an open dataset containing sensor signals for machine tool and process health assessment.



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

- Conda

Create the project environment using the provided environment file:

```bash
conda env create -f environment.yml
```

Activate the environment:

```bash
conda activate machine-health
```


## Data

The dataset is not included in this repository due to file size and data managment considerations.

This project uses the following open dataset:

**Sensor signals for machine tool and process health assessment**
Authors: Javier Alejandro Dominguez Caballero, James Moore, Jon Stammers
Dataset DOI: https://doi.org/10.15131/shef.data.24125715

The dataset contains sensor signals collected for machine tool and machining process health assessment, including simulated fault conditions such as tool imbalance, tool wear, misalignment, and surface cracks.


## Usage

The software can be run from the command line

Examples:
- Example commands will be added as the workflow develops.


## Repository Structure
- The repository strucutre will be added as the workflow develops.


## Version control

Testing will be performed using automated tests to ensure that key software components produce reliable results.

Testing methods and results will be documented as the project develops.


## License
This project is released under the MIT License.

See the LICENSE file for further details.


## Authors and Maintainers

- Author:
  - Rhodri Williams
- Maintainer:
  - Rhodri Williams
- Contact
  - rwilliams15@amrc.co.uk


## References

1. Dominguez Caballero, J. A., Moore, J., & Stammers, J. (2023).
   *Sensor signals for machine tool and process health assessment.*
   The University of Sheffield Dataset.
   https://doi.org/10.15131/shef.data.24125715.v1
