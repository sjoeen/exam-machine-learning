from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import cross_val_score
import pandas as pd
import numpy as np

X =  pd.read_csv('X_val_train.csv')
y = pd.read_csv('label_train.csv')
y = y.values.ravel()
    #transform into 1d array

"""
testing the dataset on logistic regression.
"""
lr_model = LogisticRegression()


cv_scores = cross_val_score(lr_model, X, y, cv=5, scoring='accuracy')


print(f"Logistic Regression Cross-Validated Accuracy: {cv_scores.mean()}") 
    #prints the average score across all points
    #accuracy = 0.769

"""
testing dataset on random forest.
"""


rf_model = RandomForestClassifier()
cv_scores = cross_val_score(rf_model,X,y,cv=5,scoring='accuracy')

print(f"random forest Cross-Validated Accuracy: {cv_scores.mean()}")
    #accuracy 0.745

knn_model = KNeighborsClassifier(
    n_neighbors = 10  # Number of neighbors to consider
)

cv_scores = cross_val_score(knn_model,X,y,cv=5,scoring='accuracy')

print(f"k nearest neighbors (KNN) Cross-Validated Accuracy: {cv_scores.mean()}")
    #accuracy 0.773

"""
training gaussian model
"""
y = np.array(y)
class_counts = np.bincount(y)  
prior = class_counts / len(y)


gnb_model = GaussianNB(priors=prior)


cv_scores = cross_val_score(gnb_model, X, y, cv=5, scoring='accuracy')


print(f"Gaussian Naive Bayes Cross-Validated Accuracy: {cv_scores.mean()}")
    #accuracy 0.746
