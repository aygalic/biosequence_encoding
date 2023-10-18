from .helpers import encode_recon_dataset

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd 

import plotly.express as px
import plotly.subplots as sp
import plotly.graph_objs as go

import sys
sys.path.append('../src')


# Silhouette score of original groups
from sklearn.metrics import silhouette_score
from utils.benchmark import dunn_index
from utils.benchmark import davies_bouldin


# from vq-vae
from sklearn.metrics import classification_report, silhouette_score, silhouette_samples
from sklearn.decomposition import PCA
from sklearn import metrics
from sklearn.manifold import TSNE
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix


# new plots for the pytorch refacto
def callback_viz(pca_result, encoded_set, stack, loss_hist, labels):

    # prepping a 1x4 plot to monitor the model through training
    fig, axs = plt.subplots(1, 4, figsize=(12, 3))


    # Plot the line plot in the second subplot
    axs[0].plot(loss_hist, label='Training Loss')
    axs[0].set_title('Training Loss Plot')
    #axs[0].set_xticks([])

    sns.heatmap(stack, ax=axs[1], cbar=False)
    axs[1].set_title('Stacked heatmap of two samples')
    axs[1].set_xticks([])
    axs[1].set_yticks([])


    sns.heatmap(encoded_set, ax = axs[2], cbar=False)
    axs[2].set_title('Heatmap of hole quantized dataset')
    axs[2].set_xticks([])
    axs[2].set_yticks([])


    sns.scatterplot(x = pca_result[:, 0], y = pca_result[:, 1], hue=labels, ax=axs[3])
    axs[3].set_title('PCA')
    #axs[3].set_xticks([])
    axs[3].set_yticks([])

    plt.subplots_adjust(wspace=0)  
    plt.tight_layout()
    plt.show()


def post_training_viz(data, dataloader, model, DEVICE, loss_hist, labels):

    encode_out, reconstruction_out = encode_recon_dataset(dataloader, model, DEVICE)

    # PCA of the latent space
    pca = PCA(n_components=2)
    pca.fit(encode_out)
    pca_result = pca.transform(encode_out)

    x = iter(dataloader).__next__()

    if model.is_variational:
            x_reconstructed, _, _ = model.forward(x.to(DEVICE))

    else:
        x_reconstructed = model.forward(x.to(DEVICE))
    x_reconstructed = x_reconstructed.cpu().detach().numpy()

    # stacking a single observation as well as its reconstruction in order to evaluate the results
    stack = np.vstack([x[0], x_reconstructed[0]])



    # prepping a 1x4 plot to monitor the model through training
    fig, axs = plt.subplots(2, 3, figsize=(12, 6))


    # Plot the line plot in the second subplot
    axs[0,0].plot(loss_hist, label='Training Loss')
    axs[0,0].set_title('Training Loss Plot')


    sns.heatmap(stack, ax=axs[0,1], cbar=False)
    axs[0,1].set_title('Stacked heatmap of two samples')
    axs[0,1].set_xticks([])
    axs[0,1].set_yticks([])

    sns.scatterplot(x=pca_result[:, 0], y=pca_result[:, 1], hue=labels, ax = axs[0,2])
    axs[0,2].set_title('PCA of the latent space')
    axs[0,2].set_xticks([])
    axs[0,2].set_yticks([])


    sns.heatmap(data, ax = axs[1,0], cbar=False)
    axs[1,0].set_title('Heatmap of the hole dataset')
    axs[1,0].set_xticks([])
    axs[1,0].set_yticks([])

    sns.heatmap(encode_out, ax = axs[1,1], cbar=False)
    axs[1,1].set_title('Heatmap of the hole latent space')
    axs[1,1].set_xticks([])
    axs[1,1].set_yticks([])

    sns.heatmap(reconstruction_out, ax = axs[1,2], cbar=False)
    axs[1,2].set_title('Heatmap of the hole recontruction')
    axs[1,2].set_xticks([])
    axs[1,2].set_yticks([])

    plt.subplots_adjust(wspace=0)  
    plt.tight_layout()
    plt.show()














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
def plot_single_obs_processing(dataset, autoencoder, dataset_metadata):
    e = dataset[0]    
    if(autoencoder._is_variational == True):
        _,__, z = autoencoder.encoder(e)
    else :
        z = autoencoder.encoder(e)
    decoded = autoencoder.decoder(z)
    if(dataset_metadata["is_time_series"]):
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
def plot_dataset_processing(data, autoencoder, dataset_metadata):
    if(autoencoder._is_variational == True):
        _,__, z = autoencoder.encoder(data)
    else :
        z = autoencoder.encoder(data)
    reconstruction = autoencoder.decoder.predict(z)
    if(dataset_metadata["is_time_series"]):
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


def plot_clusters(latent_Z, True_labels):
    True_labels = True_labels.tolist()  
    # Map string labels to numeric values
    my_cmap = plt.get_cmap('viridis', len(np.unique(True_labels)))
    subtype_labels = np.unique(True_labels)
    subtype_to_numeric = {subtype: i for i, subtype in enumerate(subtype_labels)}
    colors = [my_cmap(subtype_to_numeric[subtype]) for subtype in True_labels]
    #### PCA of learened feature
    pca = PCA(n_components=2)
    pca.fit(latent_Z)
    pca_result = pca.transform(latent_Z)
    # Plot the second subplot (PCA)
    sns.scatterplot(x = pca_result[:, 0], y = pca_result[:, 1], cmap=my_cmap, c=colors)
    #### Joinplot
    f = sns.jointplot(x=pca_result[:, 0], y=pca_result[:, 1], cmap="Blues", fill=True, kind='kde',height=6,
                 marginal_kws={"alpha":.2},thresh=0.05, alpha=.8)
    #### blobs 
    f = sns.jointplot(x=pca_result[:, 0], y=pca_result[:, 1], fill=True, kind='kde',hue=True_labels,height=6,marginal_kws={"alpha":.2},thresh=0.05, alpha=.9)
    f.ax_joint.legend_._visible=False


def compare_cluser_vis(data, label_1, label_2 ): 
    # create a viz to compare the labels on
    #### PCA of learened feature
    pca = PCA(n_components=2)
    pca.fit(data)
    pca_result = pca.transform(data)
    # Create a single figure with two subplots
    plt.figure(figsize=(12, 12))
    plt.subplot(2, 2, 1)
    sns.scatterplot(data=pca_result[:, 0:1], x='TSNE_Dim1', y='TSNE_Dim2', hue='label_1' )
    plt.title('True Labels')
    # Create the KDE plot in the second subplot
    plt.subplot(2, 2, 2)  # Create a new subplot for the KDE plot
    sns.scatterplot(data=pca_result[:, 0:1], x='TSNE_Dim1', y='TSNE_Dim2', hue='label_2')
    plt.title('Discovered Labels')
    conf_matrix = confusion_matrix(label_1, label_2)
    print(conf_matrix)


