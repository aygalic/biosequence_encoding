import torch
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import (
    adjusted_rand_score, normalized_mutual_info_score,
    fowlkes_mallows_score, homogeneity_completeness_v_measure, silhouette_score
)
import matplotlib.pyplot as plt
import seaborn as sns
from pytorch_lightning.callbacks import Callback

class MonitorCallback(Callback):
    def __init__(self, dataloader, labels, n_clusters, checkpoints=None, verbose=1):
        super().__init__()
        self.dataloader = dataloader
        self.labels = labels
        self.n_clusters = n_clusters
        self.checkpoints = checkpoints or [int(x) for x in np.logspace(1, 3, num=10, dtype=int)]
        self.verbose = verbose
        self.metrics = []
        self.frames = []
        self.loss_values = []


    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        # Capture the loss at the end of each training batch
        if "loss" in outputs:
            self.loss_values.append(outputs["loss"].item())

    def on_train_epoch_end(self, trainer, pl_module):
        epoch = trainer.current_epoch
        if (epoch + 1) in self.checkpoints:
            self.compute_metrics(trainer, pl_module)
            
            if self.verbose >= 1:
                self.visualize_results(trainer, pl_module)

    def compute_metrics(self, trainer, pl_module):
        pl_module.eval()
        encode_out = self.get_encoded_data(trainer, pl_module)

        # Perform K-means clustering
        kmeans = KMeans(n_clusters=self.n_clusters)
        labels = kmeans.fit_predict(encode_out)

        # Compute metrics
        metrics = {
            "hopkins": self.compute_hopkins(encode_out),
            "ari": adjusted_rand_score(self.labels, labels),
            "nmi": normalized_mutual_info_score(self.labels, labels, average_method='arithmetic'),
            "silhouette": silhouette_score(encode_out, labels),
            "fm": fowlkes_mallows_score(self.labels, labels),
        }
        homogeneity, completeness, v_measure = homogeneity_completeness_v_measure(self.labels, labels)
        metrics.update({
            "homogeneity": homogeneity,
            "completeness": completeness,
            "v_measure": v_measure
        })

        self.metrics.append(metrics)
        
        # Compute PCA for visualization
        pca = PCA(n_components=2)
        pca_result = pca.fit_transform(encode_out)
        self.frames.append(np.column_stack((np.full(pca_result.shape[0], len(self.frames)), pca_result)))

        pl_module.train()

    def get_encoded_data(self, trainer, pl_module):
        encoded_data = []
        for batch in self.dataloader:
            x = batch[0].to(pl_module.device)
            with torch.no_grad():
                encoded = pl_module.encode(x)
            encoded_data.append(encoded.cpu().numpy())
        return np.vstack(encoded_data)

    def compute_hopkins(self, X):
        # Implement Hopkins statistic calculation here
        # For brevity, I'm leaving this as a placeholder
        return 0.5  # Placeholder value

    def visualize_results(self, trainer, pl_module):
        epoch = trainer.current_epoch
        pca_result = self.frames[-1][:, 1:]

        # Visualize clustering results
        fig, axs = plt.subplots(1, 2, figsize=(12, 6))
        sns.scatterplot(x=pca_result[:, 0], y=pca_result[:, 1], hue=self.labels, ax=axs[0]).set(title='True Labels')
        sns.scatterplot(x=pca_result[:, 0], y=pca_result[:, 1], hue=KMeans(n_clusters=self.n_clusters).fit_predict(pca_result), ax=axs[1]).set(title='Found Labels')
        plt.savefig(f'clustering_visualization_epoch_{epoch}.png')
        plt.close()

        # Plot metrics
        metrics_df = pd.DataFrame(self.metrics)
        plt.figure(figsize=(12, 6))
        for column in metrics_df.columns:
            plt.plot(metrics_df.index, metrics_df[column], label=column)
        plt.legend()
        plt.title('Clustering Metrics Over Epochs')
        plt.xlabel('Epoch')
        plt.ylabel('Metric Value')
        plt.savefig(f'metrics_plot_epoch_{epoch}.png')
        plt.close()

        print(f"Epoch {epoch}: ARI = {self.metrics[-1]['ari']:.4f}")

# Usage example:
# monitor_callback = MonitorCallback(dataloader, labels, n_clusters=10)
# trainer = pl.Trainer(callbacks=[monitor_callback])
# trainer.fit(model)