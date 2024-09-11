"""
This module contains a collection of visualization functions designed to aid in the
analysis and interpretation of machine learning model outputs, particularly focusing
on aspects like training process, latent space characteristics, and dataset features.
It includes functionalities for creating various plots and animations to visually
represent model performance metrics, PCA results, dataset characteristics, and
reconstruction quality.

Functions:

- `callback_viz`: Creates a multi-faceted visualization during model training callbacks,
featuring training loss, heatmaps, and PCA scatter plots.

- `post_training_viz`: Generates comprehensive visualizations after model training,
including PCA scatter plots, heatmaps of original data, encoded space, reconstructions,
and training loss.

- `post_training_animation`: Produces an animation to visualize the evolution of PCA
results over the course of training epochs.

- `dataset_plot`: Plots the entire dataset as a heatmap and provides a kernel density
estimation plot for the total gene expression, facilitating an understanding of dataset-
wide gene expression patterns.

These functions are intended to provide intuitive and informative visual cues that help
in assessing model behavior and performance, understanding data distributions, and
identifying key characteristics of the latent space and reconstructed outputs.

Typical use cases include monitoring model training progress, evaluating model
performance, and exploring data characteristics for insights that guide further model
development and refinement.
"""

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from .helpers import encode_recon_dataset

from sklearn.decomposition import PCA

import plotly.graph_objects as go
from plotly.subplots import make_subplots

from .. import STATIC_OUTPUT_PATH

def callback_viz(pca_result, encoded_set, stack, loss_hist, labels):
    """
    Generates a 1x4 subplot visualization during model training callbacks.

    Args:
        pca_result (numpy.ndarray): PCA results for plotting.
        encoded_set (numpy.ndarray): Encoded dataset for heat map visualization.
        stack (numpy.ndarray): Stacked array of an observation and its reconstruction.
        loss_hist (list): History of training loss values.
        labels (list): Labels for data points, used in PCA scatter plot.

    Displays:
        A figure with four subplots: training loss, heatmap of samples, heatmap of the entire encoded dataset, and a PCA scatter plot.
    """
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
    """
    Generates 2x3 visualizations after model training, including PCA, heatmaps, and loss plots.

    Args:
        data (numpy.ndarray): Original dataset.
        dataloader (DataLoader): DataLoader object for the dataset.
        model (torch.nn.Module): Trained model for encoding and reconstruction.
        DEVICE (torch.device): Device on which the model is running.
        loss_hist (list): History of training loss values.
        labels (list): Labels for data points, used in PCA scatter plot.

    Displays:
        A figure with six subplots: two rows with training loss plot, heatmaps of the original dataset, encoded space, reconstruction, and a PCA scatter plot.
    """
    encode_out, reconstruction_out = encode_recon_dataset(dataloader, model, DEVICE)

    # PCA of the latent space
    pca = PCA(n_components=2)
    pca.fit(encode_out)
    pca_result = pca.transform(encode_out)

    x, _ = iter(dataloader).__next__()

    if model.variational == "VAE":
            x_reconstructed, _, _ = model.forward(x.to(DEVICE))
    elif model.variational == "VQ-VAE":
            # for VQ-VAE the latent space is the quantized space, not the encodings.
            vq_loss, x_reconstructed, perplexity, encodings, quantized = model(x.to(DEVICE))
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




def post_training_animation(monitor, metadata):
    """
    Creates a smooth animation showing the evolution of PCA results over training epochs using Plotly.

    Args:
        monitor (Monitor): Monitor object containing PCA results for each epoch.
        metadata (dict): Metadata containing labels for the data points.
    """
    # Create figure
    fig = make_subplots()

    # Create a trace for each subtype
    subtypes = set(metadata["subtypes"])
    traces = {subtype: go.Scatter(
        x=[], y=[],
        mode='markers',
        name=subtype,
        text=[],  # for hover text
        hoverinfo='text'
    ) for subtype in subtypes}

    # Create frames
    frames = []
    for i, pca_result in enumerate(monitor.frames):
        frame_data = []
        for subtype in subtypes:
            mask = [s == subtype for s in metadata["subtypes"]]
            x = pca_result[mask, 1]
            y = pca_result[mask, 2]
            text = [f"Epoch: {i}, Subtype: {subtype}" for _ in x]
            frame_data.append(go.Scatter(x=x, y=y, mode='markers', name=subtype, text=text, hoverinfo='text'))
        frames.append(go.Frame(data=frame_data, name=str(i)))

    # Add traces to figure
    for trace in traces.values():
        fig.add_trace(trace)

    # Update layout
    fig.update_layout(
        updatemenus=[{
            'buttons': [
                {
                    'args': [None, {'frame': {'duration': 500, 'redraw': True}, 'fromcurrent': True}],
                    'label': 'Play',
                    'method': 'animate',
                },
                {
                    'args': [[None], {'frame': {'duration': 0, 'redraw': True}, 'mode': 'immediate', 'transition': {'duration': 0}}],
                    'label': 'Pause',
                    'method': 'animate',
                }
            ],
            'direction': 'left',
            'pad': {'r': 10, 't': 87},
            'showactive': False,
            'type': 'buttons',
            'x': 0.1,
            'xanchor': 'right',
            'y': 0,
            'yanchor': 'top'
        }],
        sliders=[{
            'active': 0,
            'yanchor': 'top',
            'xanchor': 'left',
            'currentvalue': {
                'font': {'size': 20},
                'prefix': 'Epoch:',
                'visible': True,
                'xanchor': 'right'
            },
            'transition': {'duration': 300, 'easing': 'cubic-in-out'},
            'pad': {'b': 10, 't': 50},
            'len': 0.9,
            'x': 0.1,
            'y': 0,
            'steps': [
                {
                    'args': [[f.name], {'frame': {'duration': 300, 'redraw': True}, 'mode': 'immediate', 'transition': {'duration': 300}}],
                    'label': f.name,
                    'method': 'animate'
                } for f in frames
            ]
        }]
    )

    # Update axes
    fig.update_xaxes(title='PCA 1')
    fig.update_yaxes(title='PCA 2')

    # Add frames to figure
    fig.frames = frames

    # Show figure
    fig.show()

    # Save as HTML (can be opened in a web browser)
    savepath = STATIC_OUTPUT_PATH
    savepath.mkdir(parents=True, exist_ok=True)
    fig.write_html(savepath / "pca_animation.html")

def dataset_plot(data):
    """
    Plots the entire dataset as a heatmap and provides a density plot of the total gene expression.

    Args:
        data (numpy.ndarray): Dataset to be visualized.

    Displays:
        A figure with two subplots: a heatmap of gene expression across cells and a KDE plot showing the density of total gene expression.
    """
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
