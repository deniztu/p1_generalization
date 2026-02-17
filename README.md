# RNN Generalization Project (P1)

This repository contains code for the following paper:

[Exploration-exploitation mechanisms in recurrent neural networks and human learners in restless bandit problems](https://doi.org/10.1101/2023.04.27.538570)

- **R** is used in ``scripts`` (.R files), ``cognitive models`` (.stan files) and to save/load data (.RData files)
- **Python** is used in all .py files (main.py, classes) to train/test RNNs and in ``notebooks`` (**Jupyter Notebooks**) for analysis and plotting

## TensorFlow Versions

This repository supports both TensorFlow 1 (original) and TensorFlow 2:

- **TensorFlow 1.x** (Original): Use `main.py` and original neural network classes
- **TensorFlow 2.x** (New): Use `main_tf2.py` and TF2 neural network classes

**→ For new users, we recommend using TensorFlow 2.** See [TF2_QUICKSTART.md](TF2_QUICKSTART.md) for a quick start guide.

## Install Requirements

### Python (TensorFlow 2 - Recommended)

Install [Anaconda](https://www.anaconda.com/download), then open a terminal and navigate to this repository.

Create the TensorFlow 2 environment:

```bash
conda env create -f environment_tf2.yml
conda activate RNNExplore_TF2
```

You now have a virtual environment called ``RNNExplore_TF2`` with Python 3.9, TensorFlow 2.16.2, and all necessary dependencies.

**Quick Start:**
```bash
# Verify installation
python diagnose_tf2.py

# Run training
python main_tf2.py
```

See [TF2_QUICKSTART.md](TF2_QUICKSTART.md) for detailed instructions.

### Python (TensorFlow 1 - Legacy)

For the original TensorFlow 1 version: 
### Python (TensorFlow 1 - Legacy)

For the original TensorFlow 1 version:

```bash
conda env create -f environment.yml
conda activate RNNExplore
```

You should now have a virtual environment called ``RNNExplore`` with all the necessary dependencies for TensorFlow 1. 

### R

Install [R](https://www.r-project.org/) (This work is based on version 4.1.1) and [R Studio](https://posit.co/download/rstudio-desktop/) (This work is based on version 2021.09.1)

Then double click on ``p1_generalization.Rproj``, this will open R Studio, then in the console type the following to install all R dependancies: 

```r
renv::restore()
```
Alternatively, you can load the necessary packages from an .RData file and install them manually. 

```R
load("r_packages.RData")
install.packages(r_pckgs)
```


## Directory Structure

```
p1_generalization
│   README.md
│   main.py
│   helpers.py
|   .Rprofile
|   .gitignore
|   environment.yaml
|   helpers.py
|   p1_generalization.RData
|   r_packages.RData
|   renv.lock  
│
└───classes
│   │
│   └───bandits
│   │   
│   └───neural_networks
|       |
|       └───rnns
│   
└───cognitive_models
│
└───data
│   │   
│   └───intermediate_data
│   │   │   
│   │   └───fixed_bandits
│   │   │ 
│   │   └───modeling
│   │   │   │ 
│   │   │   └───modeling_fits
│   │   │   │    
│   │   │   └───preprocessed_data_for_modeling
│   │   │
│   │   └───jasp_analysis
│   │      
│   └───rnn_raw_data
│   │      
│   └───human_raw_data
│
└───doc
│
└───notebooks
│
└───plots
│
└───renv
│
└───saved_models
│
└───scripts
│
└───tensorboard
```

## Content
The repository contains:

* ``main.py``: Main script for TensorFlow 1 (legacy) - training, testing and simulating RNNs
* ``main_tf2.py``: Main script for TensorFlow 2 (recommended) - same functionality with modern TF2 APIs
* ``helpers.py``: contains classes/functions to handle .zip, .feather files and other helper functions
* ``classes``: contains classes for creating bandit tasks and RNNs (both TF1 and TF2 versions)
* ``cognitive_models``: contains bayesian cognitive models written in STAN
* ``data``: contains RNN and human behavioural data, pregenerated bandits to test the RNNs, data to conduct bayesian analysis with JASP, preprocessed data for cognitive modeling and posterior model fits.
* ``saved_models``: contains trained RNN weights and biases
* ``scripts``: contains scripts used in this project (see [scripts_explanations.md](../main/doc/scripts_explanations.md))
* ``notebooks``: contains jupyter notebooks used in this project (see [notebooks_explanations.md](../main/doc/notebooks_explanations.md))
* ``tensorboard``: contains saved files to plot RNN training logs in tensorboard
* ``doc``: contains documentation and how-to guides for this project

## Documentation

- **[TF2_QUICKSTART.md](TF2_QUICKSTART.md)** - Quick start guide for TensorFlow 2 version
- **[INSTALL_TF2.md](INSTALL_TF2.md)** - Detailed installation instructions for TF2
- **[TF2_MIGRATION_README.md](TF2_MIGRATION_README.md)** - Technical details of TF1→TF2 migration
- **[TF2_CONVERSION_SUMMARY.md](TF2_CONVERSION_SUMMARY.md)** - Summary of all TF2 changes
- **[TROUBLESHOOTING_TF2.md](TROUBLESHOOTING_TF2.md)** - Common issues and solutions


