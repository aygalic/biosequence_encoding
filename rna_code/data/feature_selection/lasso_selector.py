from .base_feature_selector import BaseFeatureSelector
import numpy as np

from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.utils.random import sample_without_replacement
from sklearn.metrics import confusion_matrix
from collections import Counter


class LassoSelector(BaseFeatureSelector):
    def lasso_selection(self, data_array, labels, sgdc_params=None, class_balancing=None):
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
        
        if class_balancing == "match_smaller_sample":
            balanced_data, balanced_labels = self._balance_classes(scaled_data, labels)
        else:
            balanced_data, balanced_labels = scaled_data, labels
        
        sgdc_gs = self._perform_grid_search(balanced_data, balanced_labels, sgdc_params, class_balancing)
        
        predictions = sgdc_gs.predict(scaled_data)
        self._print_results(labels, predictions, sgdc_gs)
        
        return [coef != 0 for coef in sgdc_gs.best_estimator_.coef_[0]]

    def _balance_classes(self, data, labels):
        cts = Counter(labels)
        minimum = min(cts.values())
        balanced_data = np.empty((0, data.shape[1]), float)
        balanced_labels = np.array([])
        
        for key in cts:
            sample = sample_without_replacement(cts[key], minimum)
            patient_in_class = [label == key for label in labels]
            balanced_data = np.append(balanced_data, data[patient_in_class,:][sample,:], axis=0)
            balanced_labels = np.append(balanced_labels, np.repeat(key, minimum))
        
        return balanced_data, balanced_labels

    def _perform_grid_search(self, data, labels, sgdc_params, class_balancing):
        if sgdc_params is None:
            sgdc_params = {
                'l1_ratio': np.linspace(0.1, 1, 10),
                'alpha': np.linspace(0.1, 0.5, 10),
            }
        
        sgdc = SGDClassifier(
            loss="modified_huber",
            penalty='elasticnet',
            max_iter=20000,
            class_weight="balanced" if class_balancing == "classic" else None
        )
        
        sgdc_gs = GridSearchCV(sgdc, sgdc_params, cv=5, verbose=3, n_jobs=4)
        sgdc_gs.fit(data, labels)
        return sgdc_gs

    def _print_results(self, labels, predictions, sgdc_gs):
        print("Best score:", sgdc_gs.best_score_)
        print("Best estimator:", sgdc_gs.best_estimator_)
        print("Error rate:", sum(predictions != labels) / len(labels))
        print(confusion_matrix(labels, predictions))