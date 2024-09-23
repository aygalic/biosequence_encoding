"""Module for Laplacian Score based feature selection"""

import logging

import numpy as np
from scipy.spatial.distance import pdist, squareform

from .base_feature_selector import BaseFeatureSelector


class LaplacianSelector(BaseFeatureSelector):
    """Feature selection based on Laplacian score.

    Parameters
    ----------
    threshold : float | None, optional
        Laplacian score threshold, by default None
    k : int, optional
        Number of neighbors for the kNN algorithm, by default 5
    """

    def __init__(self, threshold: float | None = None, k: int = 5):
        super().__init__(threshold)
        self.k = k
        self._plot_title = "Distribution of Laplacian Score (LS)"

    def laplacian_score(self, X):
        """
        Computes the Laplacian Score for each feature of the dataset.

        Parameters:
            X (numpy.ndarray): The dataset (samples x features).
            k (int): Number of neighbors for the KNN graph.

        Returns:
            numpy.ndarray: Array of Laplacian scores for each feature.
        """
        dists = squareform(pdist(X, metric="euclidean"))
        dists_knn = np.sort(dists)[:, 1 : self.k + 1]
        sigma = np.mean(dists_knn)
        W = np.exp(-(dists**2) / (2 * sigma**2))
        D = np.diag(np.sum(W, axis=1))
        L = D - W
        D_inverse_sqrt = np.diag(1 / np.sqrt(np.diag(D)))
        S = D_inverse_sqrt @ L @ D_inverse_sqrt

        fraternities = np.zeros(X.shape[1])
        for i in range(X.shape[1]):
            f = X[:, i] - np.mean(X[:, i])
            fraternities[i] = f.T @ S @ f / (f.T @ D @ f)

        return fraternities

    def select_features(self, data_array):
        """
        Selects features based on Laplacian Score.

        Parameters:
            data_array (numpy.ndarray): The dataset to process.
            threshold (float): The threshold for feature selection.
            k (int): Number of neighbors for the KNN graph.
            verbose (int): Controls the verbosity of the function.

        Returns:
            list: A list of boolean values indicating selected features.
        """
        self.scores = self.laplacian_score(data_array)
        selection = [self.threshold < val < 100 for val in self.scores]
        logging.info(
            "removing %i genes outside the Laplacian score window from the dataset",
            len(selection) - sum(selection),
        )
        return selection
