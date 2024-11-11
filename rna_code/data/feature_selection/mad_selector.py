"""Module for Mean absolute deviation based feature selection"""

import logging

import numpy as np
import scipy

from .base_feature_selector import BaseFeatureSelector


class MADSelector(BaseFeatureSelector):
    """Feature selection based on Mean Absolute Deviation threshold

    Parameters
    ----------
    threshold : float
        Minimum threshold for variables, by default None
    n_features : int | None, optional
        Number of features to select for given task, by default None
    ceiling : int, optional
        Maximum value to prevent outliers, by default 150
    """

    def __init__(
        self, threshold: float, ceiling: int = 150, n_features: int | None = None
    ):
        super().__init__(threshold, n_features)
        self.ceiling = ceiling
        self._plot_title = "Distribution of Median Absolute Deviation (MAD)"
        self._plot_range_values: list[float] = [0, ceiling + 20]

    def select_features(self, data_array: np.ndarray) -> list:
        """
        Selects features based on Median Absolute Deviation (MAD).

        Parameters:
            data_array (numpy.ndarray): The dataset to process.

        Returns:
            list: A list of boolean values indicating selected features.
        """
        self.scores = scipy.stats.median_abs_deviation(data_array)
        if self.threshold:
            selection = [self.threshold < val < self.ceiling for val in self.scores]
        elif self.n_features:
            idx = np.argsort(self.scores)
            selection = np.zeros(len(self.scores), dtype=bool)
            valid_features = idx[self.scores[idx] < self.ceiling]
            selection[valid_features[:min(self.n_features, len(valid_features))]] = True        
        logging.info(
            "removing %i genes outside the MAD window from the dataset",
            len(selection) - sum(selection),
        )
        return selection
