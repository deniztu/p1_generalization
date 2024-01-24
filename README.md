# RNN Generalization Project (P1)

This repository contains code for the following paper:

PAPER CITATION

## Install Requirements

### Python

Install [Anaconda](https://www.anaconda.com/download), then open Anaconda Prompt and navigate to this repository. 

Then recreate the environment of this project by typing the following in Anaconda Prompt:

```bash
conda env create -f environment.yml
```

You should now have a virtual environment called ``RNNExplore`` with all the necessary dependancies regarding Python. 

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
│   requirements.txt 
│
└───classes
│   │
│   └───bandits
│   │
│   └───neural_networks
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
│   │   └───pca
│   │      
│   └───rnn_raw_data
│
└───saved_models
│
└───scripts
│
└───tensorboard
```

## Content
The repository contains:

* ``main.py``: This is the main script, which can be used for training, testing and simulating the RNNs.
* ``helpers.py``: contains classes/functions to handle .zip, .feather files and other helper functions
* ``classes``: contains classes for simulating bandits and RNNs
* ``cognitive_models``: contains bayesian cognitive models written in STAN
* ``data``: contrains RNN behavioural data, principle components of the RNNs and preprocessed data for cognitive modeling and posterior model fits.
* ``saved_models``: contains trained RNNs
* ``scripts``: contains functions used to apply cognitive models to RNNs
* ``tensorboard``: contains saved files to plot RNN training logs in tensorboard

