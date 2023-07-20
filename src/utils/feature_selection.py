# In this file, we want to build a feature selection pipeline
# the main approach we will focus on is LASSO regression


# How it works : We select the genes that are the most influencial in a given objective
# for instance : we want the top 1000 genes involved into which cohort our patient are part of

import numpy as np
from sklearn.linear_model import LogisticRegression, Lasso, LassoCV, SGDClassifier     # TEST WHICH ONE IS FASTER. 
                                                    # (maybe there are some parallelisation differences behind the scene)
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RepeatedKFold
from numpy import arange



from glmnet import LogitNet


from sklearn.preprocessing import StandardScaler


def LASSO_selection(data_array, labels, n_genes):
    print("standardisation for LASSO regression...")

    scaler = StandardScaler()
    scaler.fit(data_array)

    scaled_data = scaler.transform(data_array)



    #
    # grid search LASSO classifier
    sgdc = SGDClassifier(loss="log", penalty='elasticnet')

    sgdc_params = {
        'l1_ratio':np.linspace(0.1, 1, 50),
        'alpha':np.linspace(0.1, 10, 100),
    }

    sgdc_gs = GridSearchCV(sgdc, sgdc_params, cv=5, verbose=1, n_jobs=1)

    # fit the model to the dataset
    sgdc_gs.fit(scaled_data, labels)


    # some debugging
    print("best score:",sgdc_gs.best_score_)
    print("best params:",sgdc_gs.best_params_)
    print("best estimator:",sgdc_gs.best_estimator_)
    print("best_estimator_.coef_:",sgdc_gs.best_estimator_.coef_)
    print("SUM best_estimator_.coef_:",sum(sgdc_gs.best_estimator_.coef_[0]))
    
    print("best_estimator_.intercept_:",sgdc_gs.best_estimator_.intercept_)
    print("SUM best_estimator_.intercept_:",sum(sgdc_gs.best_estimator_.intercept_))

    print("prediction :", sgdc_gs.predict(scaled_data))
    print("actual values :", labels)
    print("errors :", sgdc_gs.predict(scaled_data)!= labels)
    print("error rate :", sum(sgdc_gs.predict(scaled_data)!= labels)/len(labels) )

    # genes to select:
    genes_selected = [1 if coef!=0 else 0 for coef in sgdc_gs.best_estimator_.coef_[0]]
    print("genes_selected", genes_selected)
    print("sum(genes_selected)", sum(genes_selected))


    return genes_selected