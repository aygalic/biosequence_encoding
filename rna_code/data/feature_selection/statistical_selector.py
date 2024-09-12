from .base_feature_selector import BaseFeatureSelector
import numpy as np
import scipy

class StatisticalSelector(BaseFeatureSelector):
    def mad_selection(self, data_array, threshold, ceiling=100, verbose=0):
        """
        Selects features based on Median Absolute Deviation (MAD).

        Parameters:
            data_array (numpy.ndarray): The dataset to process.
            threshold (float): The lower threshold for feature selection.
            ceiling (float): The upper limit for feature selection.
            verbose (int): Controls the verbosity of the function.

        Returns:
            list: A list of boolean values indicating selected features.
        """
        MAD = scipy.stats.median_abs_deviation(data_array)
        
        if verbose:
            self._plot_distribution(MAD, threshold, 'Distribution of Median Absolute Deviation (MAD)', 
                                    'MAD Value', range_values=[0, ceiling + 20])
        
        return [threshold < val < ceiling for val in MAD]

    def expression_selection(self, data_array, threshold, verbose=0):
        """
        Selects features based on gene expression levels.

        Parameters:
            data_array (numpy.ndarray): The dataset to process.
            threshold (float): The threshold for feature selection.
            verbose (int): Controls the verbosity of the function.

        Returns:
            list: A list of boolean values indicating selected features.
        """
        expr = np.count_nonzero(data_array, axis=0) / data_array.shape[0]
        
        if verbose:
            self._plot_distribution(expr, threshold, 'Distribution of Non Zero values per genes', 
                                    'Counts of Non Zero Values', range_values=[0, 1])
        
        return [val > (1 - threshold) for val in expr]

