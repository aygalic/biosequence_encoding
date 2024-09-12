"""
Feature Selection Module

This module provides a set of classes for feature selection in datasets, particularly focusing on genomic data.
It includes various statistical and machine learning methods to identify the most significant features for modeling.

Classes:
- BaseFeatureSelector: Abstract base class for all feature selectors.
- StatisticalSelector: Implements statistical methods like MAD and expression-based selection.
- LaplacianSelector: Implements Laplacian Score-based selection.
- LassoSelector: Implements LASSO regression-based selection.

Usage:
    from feature_selection import StatisticalSelector, LaplacianSelector, LassoSelector
    
    stat_selector = StatisticalSelector()
    selected_features_mad = stat_selector.mad_selection(data_array, threshold=0.5)
    
    lap_selector = LaplacianSelector()
    selected_features_ls = lap_selector.ls_selection(data_array, threshold=0.1)
    
    lasso_selector = LassoSelector()
    selected_features_lasso = lasso_selector.lasso_selection(data_array, labels)

Note:
The module is designed with genomic datasets in mind, but may be applicable to other types of numerical datasets.
"""

from abc import ABC, abstractmethod
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler

class BaseFeatureSelector(ABC):
    def __init__(self):
        self.scaler = StandardScaler()

    @abstractmethod
    def select_features(self, data_array, **kwargs):
        pass

    def _plot_distribution(self, data, threshold, title, xlabel, range_values=None):
        plt.figure(figsize=(10, 5))
        plt.hist(data, bins=100, color='blue', range=range_values)
        plt.axvline(threshold, color='red', linestyle='--', label='Threshold')
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel('Frequency')
        plt.legend()
        plt.grid(True)
        plt.show()

