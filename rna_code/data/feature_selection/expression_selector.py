"""Expression level based selection module"""

import logging

import numpy as np

from .base_feature_selector import BaseFeatureSelector


class ExpressionSelector(BaseFeatureSelector):
    """Feature selection based on expression threshold

    Parameters
    ----------
    threshold : float | None, optional
        Expression threshold, by default None
    """
    def __init__(self, threshold: float | None = None):
        super().__init__(threshold)
        self._plot_title = "Distribution of Non Zero values per genes"
        self._plot_range_values: list[float] = [0, 1]

    def select_features(self, data_array)  -> np.ndarray:
        """
        Selects features based on gene expression levels.

        Parameters:
            data_array (numpy.ndarray): The dataset to process.
            threshold (float): The threshold for feature selection.

        Returns:
            list: A list of boolean values indicating selected features.
        """
        self.scores = np.count_nonzero(data_array, axis=0) / data_array.shape[0]
        selection = [val > (1 - self.threshold) for val in self.scores]
        logging.info(
            "removing %i genes under the expression threshold from the dataset",
            len(selection) - sum(selection),
        )
        return selection
