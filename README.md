# Genetic Algorithm optimisation of Auto-Encoders

This project provides a comprehensive pipeline for the analysis of genomic and transcriptomic data through custom PyTorch based tools. These tools are packaged as libraries for data processing, monitoring, visualization.

## Intro 

This project enables the analysis of genetic/transcriptomic dataset. 
You can find the documentation [here](https://aygalic.github.io/biosequence_encoding/).
 
## Data Generation

The scripts `app/generate_data_brca.py` and `app/generate_data_cptac_3.py` enable dataset generation from the previously downloaded data (NIH website). 

The `rna_code` module contains the `data` submodule that take care of every aspect of the data pipeline.
- The `interface` submodule take care of interfacing the app and `pandas` data format with the file system.
- The `feature_selection` submodule take care of the data processing.
- The `data_module` submodule take care of interfacing the final dataset with `pytorch_lightning`.

Dataset can be matched for transfer learning using `app/merge_datasets.py`

## Auto Encoder

This project is built around a modular Auto-Encoder architechture that can be tune around many parameters. Some structural changes such as the general architechture (Multi layer perceptron, Transformer, Convolutional Neural Net.) or some more straightforward variations (number of layers, dropout rate...) can be done through the constructor method of the `Autoencoder` Class in the `rna_code.models` module. Other structural changes are also handled, such as Variational Layer, as well as Vector Quantized Variational Auto Encoders.


## Experiment

The `experiment.py` module is a wrapper for dataset and model to be handled in a compact manner.

## Genetic Algorithm (deprecated)

The genetic algorithm is handled in the `search.py` file. It relies on an array of parametter as well as a list of configurations. Some logic has been implemented in order to avoid non-sensical configuration through generation steps. It

## Monitoring

The `monotoring.py` file is our approach to callbacks. It computes metrics, provide with visualisation call and monitoring throughout the training. It is also a key element for the search algorithm because the fitness function relies on the monitored metrics.

##  Purpose of the work

This work was conducted around a scientific question regarding the discovery of cluster in PD data. While the data is not widely available, a validation pipeline is designed around the BRCA-TCGA dataset which is open access.

The legacy code taking care of handling experiments in that context has been removed and replaces by some use-cases.

Use cases include : generating data for supported datasets and  training strategies.

You can check out individual use-cases in the `app` folder for more info.

[View PCA Animation](https://aygalic.github.io/biosequence_encoding/pca_animation.html)



