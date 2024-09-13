import logging
import scipy

from .base_feature_selector import BaseFeatureSelector


class MADSelector(BaseFeatureSelector):
    def __init__(self, threshold : float | None = None, ceiling : int = 150):
        super().__init__(threshold)
        self.ceiling = ceiling
        self._plot_title = "Distribution of Median Absolute Deviation (MAD)"
        self._plot_range_values : list[float] = [0, ceiling + 20]



    def select_features(self, data_array ):
        """
        Selects features based on Median Absolute Deviation (MAD).

        Parameters:
            data_array (numpy.ndarray): The dataset to process.
            threshold (float): The lower threshold for feature selection.

        Returns:
            list: A list of boolean values indicating selected features.
        """
        self.scores = scipy.stats.median_abs_deviation(data_array)
        selection = [self.threshold < val < self.ceiling for val in self.scores]
        logging.info(
            "removing %i genes outside the MAD window from the dataset",
            len(selection)-sum(selection))
        return selection
