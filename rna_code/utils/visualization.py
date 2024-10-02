import logging

import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
import seaborn as sns
from plotly.subplots import make_subplots
from sklearn.decomposition import PCA

from .. import STATIC_OUTPUT_PATH
from .helpers import encode_recon_dataset

logging.getLogger('matplotlib').setLevel(logging.WARNING)

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
    pca = PCA(n_components=2)
    pca.fit(encode_out)
    pca_result = pca.transform(encode_out)
    x, _ = iter(dataloader).__next__()
    x_reconstructed = model(x.to(DEVICE))
    x_reconstructed = x_reconstructed.cpu().detach().numpy()
    stack = np.vstack([x[0], x_reconstructed[0]])
    _, axs = plt.subplots(2, 3, figsize=(12, 6))
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

    # Find global min and max for x and y axes
    all_x = []
    all_y = []
    for pca_result in monitor.frames:
        all_x.extend(pca_result[:, 1])
        all_y.extend(pca_result[:, 2])
    #x_min, x_max = min(all_x), max(all_x)
    #y_min, y_max = min(all_y), max(all_y)


    x_min, x_max = -1, 1
    y_min, y_max = -1, 1

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
        frames.append(go.Frame(data=frame_data, name=str(i), layout=go.Layout(
            xaxis=dict(range=[x_min, x_max]),
            yaxis=dict(range=[y_min, y_max])
        )))

    # Add traces to figure
    for trace in traces.values():
        fig.add_trace(trace)

    # Update layout
    fig.update_layout(
        updatemenus=[{
            'buttons': [
                {
                    'args': [None, {'frame': {'duration': 300, 'redraw': True, 'easing': 'linear'}, 'fromcurrent': True}],
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
            'transition': {'duration': 300, 'easing': 'linear'},
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
        }],
        xaxis=dict(range=[x_min, x_max]),
        yaxis=dict(range=[y_min, y_max])
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
