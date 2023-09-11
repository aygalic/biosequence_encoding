import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

import plotly.express as px
import plotly.subplots as sp
import plotly.graph_objs as go


# this function is used in the dataset analysis. it plots the whole dataset as a heatmap, 
# as well as the density of total expression of genes
def dataset_plot(data):

    # Create a single figure with two subplots
    plt.figure(figsize=(12, 6))

    #plt.subplots(1, 2, figsize=(12, 6))
    plt.subplot(1, 2, 1)
    sns.heatmap(data, yticklabels=False, xticklabels=False, cbar=True)
    plt.title('Gene expression plot')
    plt.xlabel('Genes')
    plt.ylabel('Cells')


    # Create the KDE plot in the second subplot
    plt.subplot(1, 2, 2)  # Create a new subplot for the KDE plot
    sns.kdeplot(data.sum(axis=0))
    plt.title('Density of total expression throughout the dataset for each Gene')
    plt.xlabel('Sum of Expression')
    plt.ylabel('Density')



    plt.tight_layout()  # Ensure plots don't overlap
    plt.show()

# plot a single observation, its latent space as well as its reconstruction
def plot_single_obs_processing(x_train, autoencoder):
    
    e = iter(x_train).next()
    
    if(autoencoder._is_variational == True):
        _,__, z = autoencoder.encoder(e)
    else :
        z = autoencoder.encoder(e)

    decoded = autoencoder.decoder(z)


    if(x_train._is_time_series):
        e_ = e[0]  
        z_ = z[0].reshape(1, -1) 
        decoded_ = decoded[0]  
    else:
        e_ = e[0].reshape(1, -1) 
        z_ = z[0].reshape(1, -1) 
        decoded_ = decoded[0].reshape(1, -1) 



    # Create subplot grid with vertical stacking
    fig = sp.make_subplots(rows=3, cols=1, shared_xaxes=False, vertical_spacing=0.1)

    # Add the original image as a heatmap-like plot
    heatmap_trace1 = go.Heatmap(z=e_, colorscale='viridis')
    fig.add_trace(heatmap_trace1, row=1, col=1)

    # Add the latent representation as a heatmap-like plot
    heatmap_trace2 = go.Heatmap(z=z_, colorscale='viridis')
    fig.add_trace(heatmap_trace2, row=2, col=1)

    # Add the decoded image as a heatmap-like plot
    heatmap_trace3 = go.Heatmap(z=decoded_, colorscale='viridis')
    fig.add_trace(heatmap_trace3, row=3, col=1)

    # Update layout
    fig.update_layout(title='Stacked Graph of Image and Latent Space', showlegend=False)

    fig.show()



# plot the whole , its latent representation as well as its reconstruction
def plot_dataset_processing(x_train, autoencoder):

    # get everything out of TensorFlow back to numpy/pandas
    data = np.concatenate(list(x_train.as_numpy_iterator()), axis=0)

    if(autoencoder._is_variational == True):
        _,__, z = autoencoder.encoder(data)
    else :
        z = autoencoder.encoder(data)



    reconstruction = autoencoder.decoder.predict(z)


    if(x_train._is_time_series):
        data = data.reshape(data.shape[0], data.shape[2]*data.shape[1])
        reconstruction = reconstruction.reshape(reconstruction.shape[0], reconstruction.shape[2]*reconstruction.shape[1])
 

    # Create a single figure with two subplots
    plt.figure(figsize=(18, 6))

    plt.subplot(1, 3, 1)
    sns.heatmap(data, yticklabels=False, xticklabels=False, cbar=True)
    # sns.clustermap(data,yticklabels=False,xticklabels=False) # IF I WANT CLUSTERS

    plt.title('Gene expression plot')
    plt.xlabel('Genes')
    plt.ylabel('Cells')

    plt.subplot(1, 3, 2)
    sns.heatmap(z, yticklabels=False, xticklabels=False, cbar=True)
    plt.title('Latent representaiton plot')
    plt.xlabel('Latent variables')
    plt.ylabel('Cells')

    plt.subplot(1, 3, 3)
    sns.heatmap(reconstruction, yticklabels=False, xticklabels=False, cbar=True)
    plt.title('Reconstruction - Gene expression plot')
    plt.xlabel('Genes')
    plt.ylabel('Cells')


    plt.tight_layout()  # Ensure plots don't overlap
    plt.show()