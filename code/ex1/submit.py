from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.model_selection import StratifiedKFold, cross_val_score
import pandas as pd
import numpy as np

train_X =  pd.read_csv('X_val_train.csv')
train_y = pd.read_csv('label_train.csv')
train_y = train_y.values.ravel()
    #transform into 1d array
test_x = pd.read_csv('test_preprocessed.csv')
test_id = pd.read_csv('test.csv')
test_id = test_id['Id']

"""
finding optimal amount of features on trainingset:
"""

knn = KNeighborsClassifier(n_neighbors=10,weights='distance',metric='manhattan')
cv = StratifiedKFold(n_splits=5)
best_score = 0
best_amount_of_features = None

for i in range(68):
    sfs = SequentialFeatureSelector(knn, n_features_to_select=i+1, direction='forward', cv=cv, n_jobs=-1)
        #forward selection
    sfs.fit(train_X, train_y)
        #train with current amount of feautures


    scores = cross_val_score(knn, sfs.transform(train_X), train_y, cv=cv, n_jobs=-1)
    mean_score = np.mean(scores)
        #avg of all 5 cross validations

    if mean_score > best_score:
        #keeps track of current best score
        best_score = mean_score
        best_amount_of_features = i+1

print(f"best score: {best_score}, with {best_amount_of_features} features")
    #0.8014880952380953 40

"""
model prediction
"""

knn.fit(train_X,train_y)
predictions = knn.predict(test_x)
submission = pd.DataFrame({'Id': test_id, 'lipophilicity': predictions})
submission.to_csv('submission.csv', index=False)

