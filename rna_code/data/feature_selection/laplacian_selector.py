from scipy.spatial.distance import pdist, squareform
from .base_feature_selector import BaseFeatureSelector
import numpy as np


class LaplacianSelector(BaseFeatureSelector):
    def laplacian_score(self, X, k=5):
        """
        Computes the Laplacian Score for each feature of the dataset.

        Parameters:
            X (numpy.ndarray): The dataset (samples x features).
            k (int): Number of neighbors for the KNN graph.

        Returns:
            numpy.ndarray: Array of Laplacian scores for each feature.
        """
        dists = squareform(pdist(X, metric='euclidean'))
        dists_knn = np.sort(dists)[:, 1:k+1]
        sigma = np.mean(dists_knn)
        W = np.exp(-dists ** 2 / (2 * sigma ** 2))
        
        D = np.diag(np.sum(W, axis=1))
        L = D - W
        
        D_inverse_sqrt = np.diag(1 / np.sqrt(np.diag(D)))
        S = D_inverse_sqrt @ L @ D_inverse_sqrt
        
        fraternities = np.zeros(X.shape[1])
        for i in range(X.shape[1]):
            f = X[:, i] - np.mean(X[:, i])
            fraternities[i] = f.T @ S @ f / (f.T @ D @ f)
        
        return fraternities

    def ls_selection(self, data_array, threshold, k=5, verbose=0):
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
        scores = self.laplacian_score(data_array, k)
        
        if verbose:
            self._plot_distribution(scores, threshold, 'Distribution of Laplacian Score (LS)', 
                                    'Scores Value', range_values=[0, 0.01])
        
        return [threshold < val < 100 for val in scores]

