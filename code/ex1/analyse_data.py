from LogisticRegression import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import cross_val_score
from sklearn.metrics import f1_score, make_scorer
import pandas as pd
import numpy as np

X =  pd.read_csv('X_val_train.csv')
y = pd.read_csv('label_train.csv')
y = y.values.ravel()
    #transform into 1d array

f1_scorer = make_scorer(f1_score, average='weighted')

"""
testing the dataset on logistic regression.
"""
lr_model = LogisticRegression(lr=0.1,treshold=0.5,epochs=200)


cv_scores = cross_val_score(lr_model, X, y, cv=5, scoring='accuracy')
cv_f1 = cross_val_score(lr_model, X, y, cv=5, scoring=f1_scorer)

print(f"Logistic Regression Cross-Validated Accuracy: {cv_scores.mean()}") 
print(f"Logistic Regression Cross-Validated F1 Score: {cv_f1.mean()}")



"""
testing dataset on random forest.
"""

rf_model = RandomForestClassifier()
cv_scores = cross_val_score(rf_model,X,y,cv=5,scoring='accuracy')
cv_scores = cross_val_score(rf_model, X, y, cv=5, scoring='accuracy')
cv_f1 = cross_val_score(rf_model, X, y, cv=5, scoring=f1_scorer)

print(f"\nrandom forest Cross-Validated Accuracy: {cv_scores.mean()}")
print(f"random forest Cross-Validated F1 Score: {cv_f1.mean()}")

"""
testing dataset on knn
"""

knn_model = KNeighborsClassifier(n_neighbors = 10)


cv_scores = cross_val_score(knn_model,X,y,cv=5,scoring='accuracy')
cv_scores = cross_val_score(knn_model, X, y, cv=5, scoring='accuracy')
cv_f1 = cross_val_score(knn_model, X, y, cv=5, scoring=f1_scorer)

print(f"\nk nearest neighbors (KNN) Cross-Validated Accuracy: {cv_scores.mean()}")
print(f"k nearest neighbors (KNN) Cross-Validated F1 Score: {cv_f1.mean()}")


"""
testing data on gaussian model
"""

gnb_model = GaussianNB()


cv_scores = cross_val_score(gnb_model, X, y, cv=5, scoring='accuracy')
cv_scores = cross_val_score(gnb_model, X, y, cv=5, scoring='accuracy')
cv_f1 = cross_val_score(gnb_model, X, y, cv=5, scoring=f1_scorer)

print(f"\nGaussian Naive Bayes Cross-Validated Accuracy: {cv_scores.mean()}")
print(f"Gaussian Naive Bayes (KNN) Cross-Validated F1 Score: {cv_f1.mean()}")

