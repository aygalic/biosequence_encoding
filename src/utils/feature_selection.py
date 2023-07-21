# In this file, we want to build a feature selection pipeline
# the main approach we will focus on is LASSO regression


# How it works : We select the genes that are the most influencial in a given objective
# for instance : we want the top 1000 genes involved into which cohort our patient are part of

import numpy as np
import scipy

from sklearn.linear_model import SGDClassifier    
from sklearn.model_selection import GridSearchCV

from numpy import arange



from glmnet import LogitNet


from sklearn.preprocessing import StandardScaler

def MAD_selection(data_array, threshold):
        MAD = scipy.stats.median_abs_deviation(data_array)
        gene_selected = [True if val > threshold else False for val in MAD]
        return gene_selected

def LASSO_selection(data_array, labels, sgdc_params = None):
    print("standardisation for LASSO regression...")

    scaler = StandardScaler()
    scaler.fit(data_array)

    scaled_data = scaler.transform(data_array)

    # grid search LASSO classifier
    sgdc = SGDClassifier(loss="log", penalty='elasticnet')

    if(sgdc_params is None):
        sgdc_params = {
            'l1_ratio':np.linspace(0.1, 1, 10),
            'alpha':np.linspace(0.1, 0.5, 10),
        }

    sgdc_gs = GridSearchCV(sgdc, sgdc_params, cv=5, verbose=10, n_jobs=1)

    # fit the model to the dataset
    sgdc_gs.fit(scaled_data, labels)

    predictions = sgdc_gs.predict(scaled_data)
    # some debugging
    print("best score:",sgdc_gs.best_score_)
    print("best estimator:",sgdc_gs.best_estimator_)

    print("error rate :", sum(predictions!= labels)/len(labels) )
    print("Class 0 :", sum(predictions == 0) )
    print("Class 1 :", sum(predictions == 1) )

    # genes to select:
    genes_selected = [True if coef!=0 else False for coef in sgdc_gs.best_estimator_.coef_[0]]


    return genes_selected