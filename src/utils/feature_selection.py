# In this file, we want to build a feature selection pipeline
# the main approach we will focus on is LASSO regression


# How it works : We select the genes that are the most influencial in a given objective
# for instance : we want the top 1000 genes involved into which cohort our patient are part of

import numpy as np
import scipy

from sklearn.linear_model import SGDClassifier    
from sklearn.model_selection import GridSearchCV
from sklearn.utils.random import sample_without_replacement
from sklearn.metrics import confusion_matrix

from numpy import arange

from collections import Counter


from glmnet import LogitNet


from sklearn.preprocessing import StandardScaler

def MAD_selection(data_array, threshold):
        MAD = scipy.stats.median_abs_deviation(data_array)
        gene_selected = [True if val > threshold else False for val in MAD]
        return gene_selected

def LASSO_selection(data_array, labels, sgdc_params = None):

    ###########################################
    ############## normalization ##############
    ###########################################
    print("standardisation for LASSO regression...")

    scaler = StandardScaler()
    scaler.fit(data_array)

    scaled_data = scaler.transform(data_array)




    ###########################################
    ############# balacing classes ############
    ###########################################
    # we might want to use this approach, this way, we are guaranteed to use all the knowledge available from the 
    # least represented class, as opposed to a portion of it, and then rebalancing a posteriori, leaving some 
    # knowledge on the table.
    #
    # this is a nice comprimise between computational complexity and class balancing.
    #




    cts = Counter(labels)
    print("Classes : obs for a given class",cts)
    minimum = min(cts.values())
    print("minimum:",minimum)

    # initialize new empty balanced dataset
    balanced_data = np.empty((0,data_array.shape[1]), float)
    balanced_labels = np.array([])


    for key in cts:
        print(key, '->', cts[key])
        sample = sample_without_replacement(cts[key], minimum)
        patient_in_class = [True if label == key else False for label in labels]
        print(sum(patient_in_class))

        balanced_data = np.append(balanced_data, scaled_data[patient_in_class,:][sample,:], axis = 0)
        balanced_labels = np.append(balanced_labels, np.repeat(key, minimum))

    print(balanced_data.shape)
    print(balanced_labels.shape)

    # lets redesign the entire thing another way

    ###########################################
    ############### grid search ###############
    ###########################################

    # grid search LASSO classifier
    sgdc = SGDClassifier(loss="hinge", penalty='elasticnet')
    #sgdc = SGDClassifier(loss="hinge", penalty='elasticnet', class_weight = "balanced") # the easy way to balance classes



    if(sgdc_params is None):
        sgdc_params = {
            'l1_ratio':np.linspace(0.1, 1, 10),
            'alpha':np.linspace(0.1, 0.5, 10),
        }


    sgdc_gs = GridSearchCV(sgdc, sgdc_params, cv=5, verbose=3, n_jobs=4)

    # fit the model to the dataset
    sgdc_gs.fit(balanced_data, balanced_labels)

    predictions = sgdc_gs.predict(scaled_data) # predict on the unbalanced dataset
    # some debugging
    print("best score:",sgdc_gs.best_score_)
    print("best estimator:",sgdc_gs.best_estimator_)

    print("error rate :", sum(predictions!= labels)/len(labels) )

    print(confusion_matrix(labels, predictions))

    # genes to select:
    genes_selected = [True if coef!=0 else False for coef in sgdc_gs.best_estimator_.coef_[0]]


    return genes_selected