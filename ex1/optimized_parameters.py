from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import cross_val_score
from typing import Optional,Tuple
import pandas as pd
import numpy as np

X =  pd.read_csv('X_val_train.csv')
y = pd.read_csv('label_train.csv')
y = y.values.ravel()
    #transform into 1d array


def testing_parameters_lr(C_list,max_iter_list,solver_list):
    """
    tests all combination of given parameters to optimize the accuracy of
    the logistic regression model.
    """
    
    results = []
        #array to save results
        
    for c in C_list:
        for max_iter in max_iter_list:
            for solver in solver_list:

                lr_model = LogisticRegression(C = c,
                               max_iter = max_iter,
                               solver = solver)

                cv_scores = cross_val_score(lr_model, X, y, cv=5, scoring='accuracy')
                results.append((c, max_iter, solver, cv_scores.mean()))

    results = np.array(results, dtype=object)
        #convert into np array to use np.argmax

    scores = results[:, 3].astype(float)
        #get the accuracy results
    best_index = np.argmax(scores)
    best_params = results[best_index]

    return best_params


def testing_parameters_rf(criterion_list,max_depth_list,
                          max_features_list,min_sample_split_list):
    """
    tests all combination of given parameters to optimize the accuracy of
    random forest model.
    """
    results = []

    for criterion in criterion_list:
        for max_depth in max_depth_list:
            for max_features in max_features_list:
                for min_sample in min_sample_split_list:
                    rf_model = RandomForestClassifier(criterion=criterion,
                                                      max_depth=max_depth,
                                                      max_features=max_features,
                                                      min_samples_split=min_sample)
                    

                    cv_scores = cross_val_score(rf_model, X, y, cv=5, scoring='accuracy')
                    results.append((criterion, max_depth, max_features,min_sample, cv_scores.mean()))

    results = np.array(results, dtype=object)
        #convert into np array to use np.argmax

    scores = results[:, 4].astype(float)
        #get the accuracy results
    best_index = np.argmax(scores)
    best_params = results[best_index]

    return best_params


def testing_parameters_knn(n_neighbors_list,weight_list,metric_list):
    """
    tests all combination of given parameters to optimize the accuracy of
    knn model.
    """

    results = []
    
    for n_neighbors in n_neighbors_list:
        for weight in weight_list:
            for metric in metric_list:
                knn_model = KNeighborsClassifier(n_neighbors=n_neighbors,
                                                 weights=weight,
                                                 metric=metric)

                cv_scores = cross_val_score(knn_model, X, y, cv=5, scoring='accuracy')
                results.append((n_neighbors, weight, metric, cv_scores.mean()))

    results = np.array(results, dtype=object)
        #convert into np array to use np.argmax

    scores = results[:, 3].astype(float)
        #get the accuracy results
    best_index = np.argmax(scores)
    best_params = results[best_index]

    return best_params

if __name__ == "__main__":

    lr = [0.01,0.05,0.005]
    max_iter = [1000,500]
    solvers = ['lbfgs', 'liblinear', 'newton-cg', 'newton-cholesky', 'sag', 'saga']

    print(testing_parameters_lr(lr,max_iter,solvers))
        #[0.05 1000 'lbfgs' np.float64(0.7538690476190475)]

    criterion = ['gini', 'entropy', 'log_loss']
    max_depth = [None,10,20,30]
    max_features = ['sqrt', 'log2']
    min_sample_split = [5,10,100]
    print(testing_parameters_rf(criterion,max_depth,max_features,min_sample_split))
        #['log_loss' None 'sqrt' 5 np.float64(0.7705357142857143)]
        

    n_neighbors = [3,5,10]
    weights = ['uniform','distance']
    metric = ['minkowski','euclidean','manhattan']
    print(testing_parameters_knn(n_neighbors,weights,metric))
        #[5 'distance' 'minkowski' np.float64(0.7848214285714284)]
