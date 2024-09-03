"""
This module provides functionalities for monitoring and evaluating deep learning models 
during the training process, specifically tailored for clustering and dimensionality reduction tasks.

Key Components:
- Monitor class: A class designed to track and evaluate model performance over training epochs. 
  It includes methods for calculating various clustering metrics, performing PCA, and 
  visualizing both the model's latent space and reconstruction quality.

- DEVICE: A global variable indicating the device (CPU/GPU) to be used for model training and evaluation.

Main Features:
- Tracking model performance metrics such as silhouette score, Hopkins statistic, adjusted Rand index (ARI), 
  and others over epochs.
- Visualizing the model's latent space using PCA and comparing true labels with clustered labels.
- Reconstructing inputs using the model and visualizing the reconstruction quality.
- The ability to handle different types of models including VAE and VQ-VAE.

Usage:
The module is intended to be used in conjunction with model training loops. 
The Monitor class can be instantiated with a model, data loader, and labels, 
and its 'callbacks' method should be called at the end of each training epoch.

Example:
    monitor = Monitor(model, dataloader, labels)
    for epoch in range(num_epochs):
        # ... training logic ...
        monitor.callbacks(epoch)

Dependencies:
- PyTorch for model-related operations.
- Scikit-learn for PCA and clustering metrics.
- Matplotlib and Seaborn for visualization.
- Numpy and Pandas for data handling and manipulation.

Note:
This module is specifically designed for models involved in clustering and dimensionality reduction tasks. 
It assumes the existence of certain methods and attributes in the model being monitored.

"""




from .helpers import encode_recon_dataset
from .visualisation import callback_viz
from . import helpers
from . import benchmark

import math
import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt
import seaborn as sns


from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

from sklearn.metrics import confusion_matrix, adjusted_rand_score, normalized_mutual_info_score
from sklearn.metrics import fowlkes_mallows_score, homogeneity_completeness_v_measure, silhouette_score


from .. import DEVICE


class Monitor():
    def __init__(self, model, dataloader, label, verbose = 1):
        """
        Initialize the Monitor object.

        Args:
            model: The model to be monitored.
            dataloader: DataLoader object providing the data.
            label: Ground truth labels for the data.
            verbose (int): Level of verbosity for output messages.
        """
        #self.checkpoints = [math.floor(x) for x in np.logspace(1,4)] # for 10K epoch
        self.checkpoints = [math.floor(x) for x in np.logspace(1,3)] # for 1K epoch
        self.feature_num = None
        self.DEVICE = DEVICE
        self.train_res_recon_error = []
        self.perplexities = []
        self.frames = []
        self.model = model
        self.dataloader = dataloader
        self.label = label
        self.verbose = verbose
        self.metrics = []
        self.n_clusters = pd.Series(label).nunique()
        
    def set_device(self, device):
        self.DEVICE = device
    def append_loss(self, value):
        self.train_res_recon_error.append(value)

    def callbacks(self, epoch):
        """
        Perform various callback actions at specific epochs during model training.

        This method is designed to be called at the end of each training epoch. 
        It checks if the current epoch matches any pre-defined checkpoints. 
        If so, it performs several actions: computes metrics, conducts PCA on the 
        encoded output, visualizes PCA results, reconstructs a sample input, and 
        visualizes the reconstruction. It also prints the Adjusted Rand Index (ARI) 
        metric if the verbosity level is high enough.

        Args:
            epoch (int): The current epoch number in the training process.

        Note:
            This method modifies the internal state by updating metrics, adding PCA 
            results to frames, and potentially updating visualization plots. It is 
            expected to be called within a training loop.
        """
        if (epoch + 1) in self.checkpoints:
            if(self.feature_num is None):
                self.feature_num = self.model.input_shape

            self.model.eval()

            encode_out, _ = encode_recon_dataset(self.dataloader, self.model, self.DEVICE)

            # first, we compute all the metrics and add them to the list
            self.compute_metrics()
            
            # PCA of the latent space
            pca = PCA(n_components=2)
            pca.fit(encode_out)
            pca_result = pca.transform(encode_out)

            index_column = np.full((pca_result.shape[0], 1), len(self.frames), dtype=int)

            pca_result_with_index = np.hstack((index_column, pca_result))

            self.frames.append(pca_result_with_index)


            x = iter(self.dataloader).__next__()

            if self.model.variational == "VAE":
                    x_reconstructed, _, _ = self.model.forward(x.to(self.DEVICE))
            elif self.model.variational == "VQ-VAE":
                # for VQ-VAE the latent space is the quantized space, not the encodings.
                vq_loss, x_reconstructed, perplexity, encodings, quantized = self.model(x.to(self.DEVICE))
            else:
                x_reconstructed = self.model.forward(x.to(self.DEVICE))

            x_reconstructed = x_reconstructed.cpu().detach().numpy()

            # stacking a single observation as well as its reconstruction in order to evaluate the results
            stack = np.vstack([x[0], x_reconstructed[0]])
            
            if(self.verbose >=1):
                callback_viz(pca_result, encode_out, stack, self.train_res_recon_error, self.label)

                # I am also interested in the ARI:
                print("ARI", self.metrics[-1]["ari"])

            self.model.train()

    def compute_metrics(self):
        """
        Compute and store various clustering and performance metrics.

        Returns:
            list: A list containing dictionaries of computed metrics.
        """
        encode_out, _ = helpers.encode_recon_dataset(self.dataloader, self.model, DEVICE)

        # PCA of the latent space
        pca = PCA(n_components=2)
        pca.fit(encode_out)
        pca_result = pca.transform(encode_out)

        n_clusters = self.n_clusters

        # Initialize the KMeans model
        kmeans = KMeans(n_clusters=n_clusters)

        # Fit the model to data
        kmeans.fit(encode_out)

        # Get the cluster assignments
        labels = kmeans.labels_

        if self.verbose >= 1:
            # plot of True vs Discovered labels
            fig, axs = plt.subplots(1, 2, figsize=(12, 6))
            sns.scatterplot(x=pca_result[:, 0], y=pca_result[:, 1], hue= self.label, ax=axs[1]).set(title='True Labels')
            sns.scatterplot(x=pca_result[:, 0], y=pca_result[:, 1], hue=[str(l) for l in labels], ax=axs[0]).set(title='Found Labels')
            plt.show()

        # These are your cluster labels and true labels.
        y_pred = labels  
        y_true = self.label 


        # Filter out None values
        y_pred = [label for label, test in zip(y_pred, y_true) if test is not None]
        y_true = [label for label in y_true if label is not None]

        # Because your labels might not be integers, or might not start from zero, or not consecutive, we create mappings to ensure the confusion matrix is created correctly.
        true_labels = np.unique(y_true)
        pred_labels = np.unique(y_pred)

        # Create a mapping for true labels
        true_label_mapping = {label: idx for idx, label in enumerate(true_labels)}
        y_true_mapped = np.array([true_label_mapping[label] for label in y_true])

        # Create a mapping for predicted labels
        pred_label_mapping = {label: idx for idx, label in enumerate(pred_labels)}
        y_pred_mapped = np.array([pred_label_mapping[label] for label in y_pred])

        # Calculate confusion matrix
        conf_matrix = confusion_matrix(y_true_mapped, y_pred_mapped)
        
        # Calculate metrics
        silhouette_avg = silhouette_score(encode_out, labels)
        hopkins = benchmark.hopkins(encode_out)
        ari_score = adjusted_rand_score(y_true, y_pred)
        nmi_score = normalized_mutual_info_score(y_true, y_pred, average_method='arithmetic')
        fm_score = fowlkes_mallows_score(y_true, y_pred)  # or use y_true_filtered, y_pred_filtered
        homogeneity, completeness, v_measure = homogeneity_completeness_v_measure(y_true, y_pred)  # or use y_true_filtered, y_pred_filtered

        # we want to score all computed metrics for later analysis.
        self.metrics.append( {
            "hopkins"       : hopkins,
            "ari"           : ari_score,
            "nmi"           : nmi_score,
            "silhouette"    : silhouette_avg,
            "fm"            : fm_score,
            "homogeneity"   : homogeneity,
            "completeness"  : completeness,
            "v_measure"     : v_measure})

        if(self.verbose >= 1):
            # Plot the confusion matrix
            plt.figure(figsize=(10, 8))
            sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=pred_labels, yticklabels=true_labels)
            plt.ylabel('True Label')
            plt.xlabel('Clustered Label')
            plt.title('Confusion Matrix')
            plt.show()


        return self.metrics