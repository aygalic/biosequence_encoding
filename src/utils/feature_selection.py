# In this file, we want to build a feature selection pipeline
# the main approach we will focus on is LASSO regression


# How it works : We select the genes that are the most influencial in a given objective
# for instance : we want the top 1000 genes involved into which cohort our patient are part of

import numpy as np
import scipy
import matplotlib.pyplot as plt
import seaborn as sns


from sklearn.linear_model import SGDClassifier    
from sklearn.model_selection import GridSearchCV
from sklearn.utils.random import sample_without_replacement
from sklearn.metrics import confusion_matrix

from scipy.spatial.distance import pdist, squareform
from sklearn.neighbors import NearestNeighbors

from numpy import arange

from collections import Counter


#from glmnet import LogitNet


from sklearn.preprocessing import StandardScaler

def MAD_selection(data_array, threshold, ceiling = 100, verbose = 0):
    MAD = scipy.stats.median_abs_deviation(data_array)
    
    if(verbose):
        print("min MAD",min(MAD))
        print("treshold:", threshold)
        print("max MAD",max(MAD))

    
    if(verbose):
        # Plot the distribution of MAD
        plt.figure(figsize=(10, 5))
        plt.hist(MAD, bins=100, color='blue', range = [0, ceiling + 20])

        plt.axvline(threshold, color='red', linestyle='--', label='Threshold')
        plt.axvline(ceiling,   color='red', linestyle='--', label='Cieling')

        plt.title('Distribution of Median Absolute Deviation (MAD)')
        plt.xlabel('MAD Value')
        plt.ylabel('Frequency')
        plt.legend()
        plt.grid(True)
        plt.show()
    
    gene_selected = [True if val > threshold and val < ceiling else False for val in MAD]
    return gene_selected



def laplacian_score(X, k=5):
    """
    Compute the Laplacian Score for each feature of dataset X.

    Parameters:
    X : numpy array : The dataset (number of samples x number of features)
    k : int : Number of neighbors for the KNN graph

    Returns:
    scores : list : Laplacian scores for each feature
    """

    # Step 1: Construct the adjacency matrix W using the heat kernel based on Euclidean distance
    dists = squareform(pdist(X, metric='euclidean'))
    dists_knn = np.sort(dists)[:, 1:k+1]  # Excluding the first column (distance to itself)
    sigma = np.mean(dists_knn)
    heat_kernel = np.exp(-dists ** 2 / (2 * sigma ** 2))

    # Step 2: Define the diagonal matrix D and compute the Laplacian matrix L
    W = heat_kernel
    D = np.diag(np.sum(W, axis=1))
    L = D - W

    # Step 3: Compute the pairwise fraternities for each feature
    fraternities = np.zeros(X.shape[1])  # number of features
    D_inverse_sqrt = np.diag(1 / np.sqrt(np.diag(D)))
    S = D_inverse_sqrt @ L @ D_inverse_sqrt

    for i in range(X.shape[1]):
        f = X[:, i]
        f = f - np.mean(f)  # centering the feature
        fraternities[i] = f.T @ S @ f / (f.T @ D @ f)

    return fraternities




def LS_selection(data_array, threshold, k = 5, verbose = 0):
    '''Laplacian score selection'''
    # Compute Laplacian scores
    scores = laplacian_score(data_array, k)

    if(verbose):
        print("min LS",min(scores))
        print("max LS",max(scores))

    # we also use a cieling to get rid of outliers.
    gene_selected = [True if val > threshold and val < 100 else False for val in scores]
    
    if(verbose):
        # Plot the distribution of MAD
        plt.figure(figsize=(10, 5))
        plt.hist(scores, bins=200, color='blue', range = [0, .01])
        plt.axvline(threshold, color='red', linestyle='--', label='Threshold')
        plt.title('Distribution of Laplacian Score (LS)')
        plt.xlabel('scores Value')
        plt.ylabel('Frequency')
        plt.legend()
        plt.grid(True)
        plt.show()
    return gene_selected


def expression_selection(data_array, threshold, verbose = 0):
    expr = np.count_nonzero(data_array, axis = 0)/data_array.shape[0]    
    if(verbose):
        print("min expression level",min(expr))
        print("max expression level",max(expr))

    # we also use a cieling to get rid of outliers.
    gene_selected = [True if val > (1-threshold) else False for val in expr]
    
    if(verbose):
        # Plot the distribution of MAD
        plt.figure(figsize=(10, 5))
        plt.hist(expr, bins=100, color='blue', range = [0,1])
        plt.axvline(threshold, color='red', linestyle='--', label='Threshold')
        plt.title('Distribution of Non Zero values per genes')
        plt.xlabel('Counts of Non Zero Values')
        plt.ylabel('Frequency')
        plt.legend()
        plt.grid(True)
        plt.show()
    return gene_selected

def LASSO_selection(data_array, labels, sgdc_params = None, class_balancing = None):

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

    if(class_balancing == "match_smaller_sample"):
        cts = Counter(labels)

        minimum = min(cts.values())
        print("Balancing data with:",minimum, "samples in each class")

        # initialize new empty balanced dataset
        balanced_data = np.empty((0,data_array.shape[1]), float)
        balanced_labels = np.array([])


        for key in cts:
            sample = sample_without_replacement(cts[key], minimum)
            patient_in_class = [True if label == key else False for label in labels]
            balanced_data = np.append(balanced_data, scaled_data[patient_in_class,:][sample,:], axis = 0)
            balanced_labels = np.append(balanced_labels, np.repeat(key, minimum))




    else:
         balanced_data = scaled_data
         balanced_labels = labels

    # lets redesign the entire thing another way
    # 
    # we could als create a dataset that always contain every minimum for each class as soon as the amount allow it
    # but this should be designed in the data_handler "subsampling" section....




    ###########################################
    ############### grid search ###############
    ###########################################



    # grid search LASSO classifier
    sgdc = SGDClassifier(loss="modified_huber", penalty='elasticnet', max_iter = 20000)
    if(class_balancing == "classic"):
        sgdc = SGDClassifier(loss="modified_huber", penalty='elasticnet', class_weight = "balanced", max_iter = 20000) # the easy way to balance classes



    if(sgdc_params is None):
        sgdc_params = {
            'l1_ratio':np.linspace(0.1, 1, 10),
            'alpha':np.linspace(0.1, 0.5, 10),
        }


    sgdc_gs = GridSearchCV(sgdc, sgdc_params, cv=5, verbose=3, n_jobs=4)

    # fit the model to the dataset
    if(class_balancing == "match_smaller_sample"):
        sgdc_gs.fit(balanced_data, balanced_labels)        
    else:
        sgdc_gs.fit(scaled_data, labels)


    predictions = sgdc_gs.predict(scaled_data) # predict on the unbalanced dataset
    # some debugging
    print("best score:",sgdc_gs.best_score_)
    print("best estimator:",sgdc_gs.best_estimator_)

    print("error rate :", sum(predictions!= labels)/len(labels) )

    print(confusion_matrix(labels, predictions))

    # genes to select:
    genes_selected = [True if coef!=0 else False for coef in sgdc_gs.best_estimator_.coef_[0]]


    return genes_selected