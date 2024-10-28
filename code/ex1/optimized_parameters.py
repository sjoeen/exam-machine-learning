from LogisticRegression import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import cross_val_score
from typing import Optional,Tuple
from sklearn.metrics import f1_score, make_scorer
import pandas as pd
import numpy as np

X =  pd.read_csv('X_val_train.csv')
y = pd.read_csv('label_train.csv')
y = y.values.ravel()
    #transform into 1d array

f1_scorer = make_scorer(f1_score, average='weighted')

def testing_parameters_lr(lr_list,max_iter_list,treshold_list):
    """
    tests all combination of given parameters to optimize the accuracy of
    the logistic regression model.
    """
    
    results = []
        #array to save results
        
    for lr in lr_list:
        for max_iter in max_iter_list:
                for treshold in treshold_list:

                    lr_model = LogisticRegression(lr, max_iter,treshold)
                    cv_scores = cross_val_score(lr_model, X, y, cv=5, scoring='accuracy')
                    results.append((lr, max_iter,treshold, cv_scores.mean()))

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

    lr = [0.01,0.05,0.1]
    max_iter = [1000,5000]
    treshold = [0.3,0.4,0.5,0.6,0.7]

    print(testing_parameters_lr(lr,max_iter,treshold))
        #[0.1 1000 0.3 np.float64(0.7002976190476191)]

    criterion = ['gini', 'entropy', 'log_loss']
    max_depth = [None,10,20,3]
    max_features = ['sqrt', 'log2']
    min_sample_split = [5,10,100]
    print(testing_parameters_rf(criterion,max_depth,max_features,min_sample_split))
        #['log_loss' None 'sqrt' 5 np.float64(0.7705357142857143)]
        

    n_neighbors = [3,5,10]
    weights = ['uniform','distance']
    metric = ['minkowski','euclidean','manhattan']
    print(testing_parameters_knn(n_neighbors,weights,metric))
        #[5 'distance' 'minkowski' np.float64(0.7848214285714284)]

    y = np.array(y)
    class_counts = np.bincount(y)  
    prior = class_counts / len(y)


    gnb_model = GaussianNB(priors=prior)


    cv_scores = cross_val_score(gnb_model, X, y, cv=5, scoring='accuracy')


    print(f"Gaussian Naive Bayes Cross-Validated Accuracy: {cv_scores.mean()}")
        #accuracy 0.68

