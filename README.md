# Genetic Algorithm optimisation of Auto-Encoders

This project provides a comprehensive pipeline for the analysis of genomic and transcriptomic data through custom PyTorch based tools. These tools are packaged as libraries for data processing, benchmarking, visualization, as well as Python notebooks for in-depth analyses.



## Intro 

This project enables the analysis of genetic/transcriptomic dataset. 
 
## Data Generation

The `utils` folder contains the `data_handler.py` script, which serves as the primary interface with our dataset. Additionally, you'll find features in the `feature_selection.py` file. These tools provide the essential machinery to construct a PyTorch dataset and allow users to fine-tune feature selection and data engineering.

On the numerical side of thing, the following filtering techniques are implemented:

- Mean absolute deviation threshold/ceiling
- LASSO selection
- log1p transform
- Normalisation
- Min Max Scaling to [0,1]
- Subsampling (for efficient testing on smaller dataset)


Some features are enabled using logics based on the names of the featuer or their position on the genome:

- Study Phase selection
- Time point selection
- Cohort selection
- Aberration removal (handling NA's and incorrect gene counts)
- Genes symbols and possition can be retrieves an filtered on wether or not they are present in a given database
- Symbols can be sorted by their genomic positions.
- Certain gene families, such as mitochondrial genes, can be excluded




The dataset can then be output with the following architecture:

- Cell wise analysis, where each reading is a single sample at a single timepoint, without taking for account the fact that a given reading is a single observation from different time points linked to a patient.


## Auto Encoder

This project is built around a modular Auto-Encoder architechture that can be tune around many parameters. Some structural changes such as the general architechture (Multi layer perceptron, Transformer, Convolutional Neural Net.) or some more straightforward variations (number of layers, dropout rate...) can be done through the constructor method of the `Autoencoder` Class in the `src/model.py` module. Other structural changes are also handled, such as Variational Layer, as well as Vector Quantized Variational Auto Encoders.


## Experiment

The `experiment.py` module is a wrapper for dataset and model to be handled in a compact manner.

## Genetic Algorithm

The genetic algorithm is handled in the `search.py` file. It relies on an array of parametter as well as a list of configurations. Some logic has been implemented in order to avoid non-sensical configuration through generation steps. It

## Monitoring

The `monotoring.py` file is our approach to callbacks. It computes metrics, provide with visualisation call and monitoring throughout the training. It is also a key element for the search algorithm because the fitness function relies on the monitored metrics.

## The Notebooks - Purpose of the work

This work was conducted around a scientific question regarding the discovery of cluster in PD data. While the data is not widely available, a validation pipeline is designed around the BRCA-TCGA dataset which is open access.

The folder `thesis_related` encapsulate all the work done around this scientific question.

You can Find example for training in the `model_training` folder, where both Genetic algorithm and standalone approach are used.



