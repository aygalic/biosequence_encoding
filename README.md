# genome_analysis_parkinson

## Intro 

This project has for goal to perform an analysis over a dataset of parkinson disease transcriptomic dataset.

## Approach

First step : performing some exploratory analysis with R

This is done in the files "dataset_genration.R" and "LASSO.R" but are now a bit outdated compared to what has been implemented in the python library

Second step :

Building an auto encoder pipeline. At first we need a good interface with the dataset which is provided in the /src/utils dir.

We implemented multiple data preprocessing and feature selection techniques.

Then we implement two main classes of models : the one based on simple patient representation, without taking into account the temporality of the data, and a second one, dealing with the data in form of time series.

Third step : Post training and encoding analysis of the latent representation of each patient. This is done in R in two separate files for the time series and non time series data.








