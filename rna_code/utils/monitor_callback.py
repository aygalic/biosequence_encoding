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
from typing import  List, Dict, Union, Optional

class MetricsComputer:
    @staticmethod
    def compute_metrics(
        encoded_data: np.ndarray,
        true_labels: List[int],
        n_clusters: int
        ) -> Dict[str, float]:
        kmeans = KMeans(n_clusters=n_clusters)
        pred_labels = kmeans.fit_predict(encoded_data)
        
        metrics = {
            "hopkins": MetricsComputer.compute_hopkins(encoded_data),
            "ari": adjusted_rand_score(true_labels, pred_labels),
            "nmi": normalized_mutual_info_score(true_labels, pred_labels, average_method='arithmetic'),
            "silhouette": silhouette_score(encoded_data, pred_labels),
            "fm": fowlkes_mallows_score(true_labels, pred_labels),
        }
        homogeneity, completeness, v_measure = homogeneity_completeness_v_measure(true_labels, pred_labels)
        metrics.update({
            "homogeneity": homogeneity,
            "completeness": completeness,
            "v_measure": v_measure
        })
        return metrics

    @staticmethod
    def compute_hopkins(X: np.ndarray) -> float:
        # Implement Hopkins statistic calculation here
        return 0.5  # Placeholder value

class Visualizer:
    @staticmethod
    def visualize_clustering(pca_result: np.ndarray, true_labels: List[int], pred_labels: List[int], epoch: int):
        fig, axs = plt.subplots(1, 2, figsize=(12, 6))
        sns.scatterplot(x=pca_result[:, 0], y=pca_result[:, 1], hue=true_labels, ax=axs[0]).set(title='True Labels')
        sns.scatterplot(x=pca_result[:, 0], y=pca_result[:, 1], hue=pred_labels, ax=axs[1]).set(title='Found Labels')
        plt.savefig(f'clustering_visualization_epoch_{epoch}.png')
        plt.close()

    @staticmethod
    def plot_metrics(metrics: List[Dict[str, float]], epoch: int):
        metrics_df = pd.DataFrame(metrics)
        plt.figure(figsize=(12, 6))
        for column in metrics_df.columns:
            plt.plot(metrics_df.index, metrics_df[column], label=column)
        plt.legend()
        plt.title('Clustering Metrics Over Epochs')
        plt.xlabel('Epoch')
        plt.ylabel('Metric Value')
        plt.savefig(f'metrics_plot_epoch_{epoch}.png')
        plt.close()

class MonitorCallback(Callback):
    def __init__(self,
                 dataloader: torch.utils.data.DataLoader,
                 labels: List[int],
                 n_clusters: int,
                 evaluation_intervals: Optional[List[int]] = None,
                 compute_on: str = 'epoch',
                 verbose: int = 0):
        super().__init__()
        self.dataloader = dataloader
        self.labels = self._labels_to_int(labels)
        self.n_clusters = n_clusters
        self.evaluation_intervals = evaluation_intervals or np.unique([int(x) for x in np.logspace(1, 3, num=50)])
        self.compute_on = compute_on
        self.verbose = verbose
        self.metrics = []
        self.frames = []
        self.loss_values = []
        self.global_batch_count = 0


    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        self.global_batch_count += 1

        if "loss" in outputs:
            self.loss_values.append(outputs["loss"].item())
        
        if self.compute_on == 'batch' and (self.global_batch_count) in self.evaluation_intervals:
            self.compute_and_visualize(trainer, pl_module)

    def on_train_epoch_end(self, trainer, pl_module):
        if self.compute_on == 'epoch' and (trainer.current_epoch + 1) in self.evaluation_intervals:
            self.compute_and_visualize(trainer, pl_module)

    def compute_and_visualize(self, trainer, pl_module):
        pl_module.eval()
        encode_out = self.get_encoded_data(trainer, pl_module)

        metrics = MetricsComputer.compute_metrics(encode_out, self.labels, self.n_clusters)
        self.metrics.append(metrics)
        
        pca = PCA(n_components=2)
        pca_result = pca.fit_transform(encode_out)
        self.frames.append(np.column_stack((np.full(pca_result.shape[0], len(self.frames)), pca_result)))

        if self.verbose >= 1:
            Visualizer.visualize_clustering(pca_result, self.labels, 
                                            KMeans(n_clusters=self.n_clusters).fit_predict(encode_out), 
                                            trainer.current_epoch)
            Visualizer.plot_metrics(self.metrics, trainer.current_epoch)
            print(f"Epoch {trainer.current_epoch}: ARI = {metrics['ari']:.4f}")

        pl_module.train()

    def get_encoded_data(self, trainer, pl_module):
        encoded_data = []
        for batch in self.dataloader:
            x = batch[0].to(pl_module.device)
            with torch.no_grad():
                encoded = pl_module.encode(x)
            encoded_data.append(encoded.cpu().numpy())
        return np.vstack(encoded_data)

    @staticmethod
    def _labels_to_int(labels: List[Union[int, str]]) -> List[int]:
        unique_labels = {label: i for (i, label) in enumerate(np.unique(labels))}
        return [unique_labels[label] for label in labels]
