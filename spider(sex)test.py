from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.model_selection import train_test_split

X = pd.read_csv('train_cleaned.csv')
id = pd.read_csv('id_cleaned.csv')
y = pd.read_csv('lipophilicity_cleaned.csv')
y = y.values.ravel()
test = pd.read_csv('test_cleaned.csv')
id_test = pd.read_csv('test.csv')
id_test = id_test.iloc[:, 0]
print(id_test)

print(test.shape)

# Opprett Logistic Regression-modellen
lr_model = LogisticRegression(
    solver='lbfgs',            # Standard solver for logistisk regresjon
    max_iter=1000,             # Øk antall iterasjoner for å sikre konvergens
    random_state=42
)

lr_model.fit(X,y)

y_pred = lr_model.predict(test)

# Prepare the submission file
submission = pd.DataFrame({
    'Id': id_test,              # Use the ID column from the test set
    'lipophilicity': y_pred     # Model predictions
})

# Save to CSV file with the correct format
submission.to_csv('submission.csv', index=False)

print("Submission file created successfully.")
