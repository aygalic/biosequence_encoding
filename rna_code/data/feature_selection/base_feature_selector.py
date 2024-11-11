"""Abstract class for feature selection"""

from abc import ABC, abstractmethod
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler


class BaseFeatureSelector(ABC):
    """Base Abstract class for FeatureSelectors

    Parameters
    ----------
    threshold : float | None, optional
        Selection threshold for given task, by default None
    n_features : int | None, optional
        Number of features to select for given task, by default None
    """

    def __init__(self, threshold: float | None = None, n_features: int | None = None):
        self.scaler = StandardScaler()
        self.scores: list[float] = []
        self.threshold: float | None = threshold
        self.n_features: int | None = n_features
        self._plot_title: str = ""
        self._plot_x_label: str = "Scores Value"
        self._plot_range_values: list[float] = [0, 0.01]
        assert self.threshold or self.n_features is not None, "Must specify either feature threshold or count"

    @abstractmethod
    def select_features(self, data_array: np.ndarray, **kwargs) -> np.ndarray:
        """Select features from data array according to self.threshold.

        Parameters
        ----------
        data_array : np.ndarray
            Data to select features from.

        Returns
        -------
        np.ndarray
            Filtered data.
        """

    def _plot_distribution(self):
        """Plot data distribution with given threshold."""
        plt.figure(figsize=(10, 5))
        plt.hist(self.scores, bins=100, color="blue", range=self._plot_range_values)
        plt.axvline(self.threshold, color="red", linestyle="--", label="Threshold")
        plt.title(self._plot_title)
        plt.xlabel(self._plot_x_label)
        plt.ylabel("Frequency")
        plt.legend()
        plt.grid(True)
        plt.show()
