from .. import config
from .helpers import Mydatasets, encode_recon_dataset
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

# so many metrics
from sklearn.metrics import confusion_matrix, adjusted_rand_score, normalized_mutual_info_score
from sklearn.metrics import fowlkes_mallows_score, homogeneity_completeness_v_measure, silhouette_score



DEVICE = torch.device(config["DEVICE"])


class Monitor():
    def __init__(self, model, dataloader, label, verbose = 1):
        self.checkpoints = [math.floor(x) for x in np.logspace(1,4)]
        self.feature_num = None
        self.DEVICE = torch.device(config["DEVICE"])
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

                # I am also interested in the Hopkins statistics:
                print("HOPKINS STATISTIC", self.metrics[-1]["hopkins"])



            self.model.train()

    def compute_metrics(self):

        encode_out, _ = helpers.encode_recon_dataset(self.dataloader, self.model, DEVICE)

        # PCA of the latent space
        pca = PCA(n_components=2)
        pca.fit(encode_out)
        pca_result = pca.transform(encode_out)


        # Assuming you've determined that you want 5 clusters
        n_clusters = self.n_clusters




        # Initialize the KMeans model
        kmeans = KMeans(n_clusters=n_clusters)

        # Fit the model on your data
        kmeans.fit(encode_out)

        # Get the cluster assignments
        labels = kmeans.labels_

        # Calculate silhouette score
        silhouette_avg = silhouette_score(encode_out, labels)

        # hopkins statistics
        hopkins = benchmark.hopkins(encode_out)


        if self.verbose >= 1:
            print(f"Silhouette score for {n_clusters} clusters: {silhouette_avg}")

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

        # just in case we wanted to compute extra silouhette
        #X_filtered = [x for x, test in zip(encode_out, y_true) if test is not None]




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