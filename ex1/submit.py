from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
import pandas as pd

train_X =  pd.read_csv('X_val_train.csv')
train_y = pd.read_csv('label_train.csv')
train_y = train_y.values.ravel()
    #transform into 1d array
test_x = pd.read_csv('test_preprocessed.csv')
test_id = pd.read_csv('test.csv')
test_id = test_id['Id']


#knn = KNeighborsClassifier(n_neighbors=10,weights='distance',metric='manhattan')
#knn.fit(train_X,train_y)
#predictions = knn.predict(test_x)

lr = LogisticRegression(C=0.05,
                        max_iter=1000,
                        solver='lbfgs')
lr.fit(train_X,train_y)
predictions = lr.predict(test_x)


submission = pd.DataFrame({'Id': test_id, 'lipophilicity': predictions})

# Lagre prediksjonene i en CSV-fil for innsending
submission.to_csv('submission.csv', index=False)

print("Prediksjoner lagret i submission.csv!")