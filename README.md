# Deep learning auto-encoding & analysis of genomic/transcriptomic data 

This project provides a comprehensive pipeline for the analysis of genomic and transcriptomic data through custom Python and R tools. These tools are packaged as libraries for data processing, benchmarking, visualization, as well as Python notebooks coupled with R scripts for in-depth analyses.



## Intro 

This project enables the analysis of genetic/transcriptomic dataset. Some preliminary steps are performed in R. The `LASSO.R` and `dataset_generation.R` files correspond to such analysis. It had the purpose of determining wether or not LASSO could be a good approach as well as some helpers script about dataset generation in R.

Those suffer from performance issues that have been hard to overcome due to the slow nature of R. The file `dataset_generation.R` tries to aliviate those limitations by providing lighter files to perform the analysis on.



This project facilitates the analysis of genetic and transcriptomic datasets. Some initial steps are performed in R. The `LASSO.R` and `dataset_generation.R` files correspond to these preliminary analyses, aiming to determine whether LASSO could be a suitable approach and to provide helper scripts for dataset generation in R.

These R scripts have faced performance issues due to the inherent slowness of R. To address these limitations, the `dataset_generation.R` file offers more efficient alternatives for analysis.
 
## Data Helpers

The `utils` folder contains the `data_handler.py` script, which serves as the primary interface with our dataset. Additionally, you'll find features in the feature_selection.py file. These tools not only provide the essential machinery to construct a TensorFlow dataset but also allow users to fine-tune feature selection and data engineering.

The following procedures are supported:

For patient selection:

- Study Phase selection
- Subsampling (for efficient testing on smaller dataset)
- Time point selection
- Aberration removal (handling NA's and incorrect gene counts)


For data-driven feature engineering:

- Mean absolute deviation threshold/ceiling
- LASSO selection
- log1p transform
- Normalisation
- Min Max Scaling to [0,1]

For application-specific feature engineering:

- Genes symbols and possition can be retrieves an filtered on wether or not they are present in a given database
- Symbols can be sorted by their genomic positions.
- Certain gene families, such as mitochondrial genes, can be excluded


The dataset can then be output following 2 different architectures:

- Cell wise analysis, where each reading is considered on its own, without taking for account the fact that a given reading is a single observation from different time points linked to a patient.
- Time series analysis, we build sequences of time points for every patient

A transpose operation is also possible for time series data, allowing the processing of genes as a sequence or as simple features.


## Auto encoding

Throughout this project, we explore multiple architectural approaches to obtain meaningful representations of the dataset. All models are available in the `utils/models` folder, including:


- Simple Fully Connected Autoencoder (`vanilla_autoencoder.py`): Effective for quick encoding of individual observations.
- Fully Connected Variational Autoencoder (`vae.py`): An exploratory VAE with similar results to the simple autoencoder.
- Convolutional Neural Network Autoencoder (`cnn_encoder.py`): Designed to handle datasets with paterns.
- Variational Convolutional Neural Network Autoencoder (`ConvVAE.py`): Investigating potential improvements.
- Long Short-Term Memory Autoencoder (`LSTM.py`): Exploring the ability to handle long-term dependencies.

upcoming : 
- variational LSTM autoencoder

## The Notebooks


Each dataset and network architecture are thoroughly explored in dedicated Python notebooks. The dataset notebooks break down the data analysis pipeline:

- `Data_Analysis_cancer.ipynb`
- `Data_Analysis_genes.ipynb`
- `Data_Analysis_transcripts.ipynb`

The following notebooks focus on different model types:

- `FC_autoencoder.ipynb` most basic model
- `FC_VAE.ipynb`
- `cnn.ipynb` - 
- `ConVAE.ipynb`
- `LSTM.ipynb`
- `LSTM_transpose.ipynb`

Those notebook are designed around the following steps for each model architecture:

- building dataset
- building model
- fitting the model to the dataset
- monitoring the fit
- benchmarking the model against others
- providing some preliminary visualisation of the dataset latent space and reconstruction
- when needed, some troubleshooting steps to enlighten model weaknesses
- generation and saving of the encoded dataset in a usable format

Here, you can see an example of a single-cell encoding visualization, including the original data, latent representation, and reconstitution:

![Single Cell encoding - visualisation of orignial data, latent representation and reconstitution](img/single_cell_encoding.png)

And here is an example of the entire dataset's encoding visualization:

![Whole dataset encoding - visualisation of orignial data, latent representation and reconstitution](img/full_dataset_encoding.png)

## Model Benchmarking

The `benchmark.py` script allows you to benchmark models on the same dataset or different subsets, accommodating varying numbers of features and entry points by averaging the loss function. Models are then ranked and plotted on a complexity/loss plot.

## Post encoding analysis

Post-encoding analysis is performed through three R files located in the `post_training_analysis` folder. 

* `cell_encoding_analysis.R` Dedicated to data encoded from individual cells.
* `time_series_encoding_analysis.R` Focused on data encoded from time series.
* `cancer_encoding_analysis.R` Tailored for data encoded from the cancer dataset.

Each of these approaches requires slightly different analysis techniques. The analysis includes:

### PCA Visualisation

Principal Component Analysis (PCA) is utilized to explore the dataset's structure and identify clusters.

![Alt text](img/PCA_cancer.png)

Various PCA parameters are monitored before plotting the dataset in the projection space.

### t-SNE Visualisation

t-Distributed Stochastic Neighbor Embedding (t-SNE) is used to create multiple plots across different parameter settings.

An animated plot is employed to search for a satisfactory representation.


![t-SNE simple auto encoder on cancer data](img/cancer_tsne.png)
