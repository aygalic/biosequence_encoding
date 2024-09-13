import logging
from collections import Counter

import numpy as np
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn.utils.random import sample_without_replacement

from .base_feature_selector import BaseFeatureSelector


class LassoSelector(BaseFeatureSelector):

    def __init__(
            self,
            labels,
            threshold : float | None = 0,
            sgdc_params : dict | None = None,
            class_balancing=None
            
            ):
        super().__init__(threshold)
        self.class_balancing = class_balancing
        self.labels = labels
        if sgdc_params is None:
            self.sgdc_params = {
                'l1_ratio': np.linspace(0.1, 1, 10),
                'alpha': np.linspace(0.1, 0.5, 10),
            }
        else:
            self.sgdc_params = sgdc_params



    def select_features(self, data_array):
        """
        Selects features using LASSO regression.

        Parameters:
            data_array (numpy.ndarray): The dataset to process.
            labels (numpy.ndarray): The labels associated with the data.
            sgdc_params (dict, optional): Parameters for the SGDClassifier.
            class_balancing (str, optional): Method for class balancing.

        Returns:
            list: A list of boolean values indicating selected features.
        """
        scaled_data = self.scaler.fit_transform(data_array)
        if self.class_balancing == "match_smaller_sample":
            balanced_data, balanced_labels = self._balance_classes(scaled_data)
        else:
            balanced_data, balanced_labels = scaled_data, self.labels
        sgdc_gs = self._perform_grid_search(balanced_data, balanced_labels)
        predictions = sgdc_gs.predict(scaled_data)
        self._print_results(self.labels, predictions, sgdc_gs)
        self.scores = sgdc_gs.best_estimator_.coef_[0]
        selection = [abs(coef) > 0 for coef in self.scores]
        logging.info(
            "removing %i genes under the LASSO threshold from the dataset",
            len(selection)-sum(selection))
        return selection

    def _balance_classes(self, data):
        cts = Counter(self.labels)
        minimum = min(cts.values())
        balanced_data = np.empty((0, data.shape[1]), float)
        balanced_labels = np.array([])
        for key in cts:
            sample = sample_without_replacement(cts[key], minimum)
            patient_in_class = [label == key for label in self.labels]
            balanced_data = np.append(balanced_data, data[patient_in_class,:][sample,:], axis=0)
            balanced_labels = np.append(balanced_labels, np.repeat(key, minimum))
        return balanced_data, balanced_labels

    def _perform_grid_search(self, data, labels):
        sgdc = SGDClassifier(
            loss="modified_huber",
            penalty='elasticnet',
            max_iter=20000,
            class_weight="balanced" if self.class_balancing == "classic" else None
        )
        sgdc_gs = GridSearchCV(sgdc, self.sgdc_params, cv=5, verbose=3, n_jobs=4)
        sgdc_gs.fit(data, labels)
        return sgdc_gs

    def _print_results(self, labels, predictions, sgdc_gs):
        print("Best score:", sgdc_gs.best_score_)
        print("Best estimator:", sgdc_gs.best_estimator_)
        print("Error rate:", sum(predictions != labels) / len(labels))
        print(confusion_matrix(labels, predictions))

    def _plot_distribution(self):
        raise NotImplementedError