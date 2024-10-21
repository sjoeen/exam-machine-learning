from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.model_selection import train_test_split

X = pd.read_csv('train_cleaned.csv')
print(X.head())
id = pd.read_csv('id_cleaned.csv')
y = pd.read_csv('lipophilicity_cleaned.csv')
y = y.values.ravel()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

lda = LDA(n_components=1)  # For to klasser, maks antall komponenter er 1
X_train_lda = lda.fit_transform(X_train, y_train)
X_test_lda = lda.transform(X_test)

class_counts = np.bincount(y_train)  # Teller forekomsten av hver klasse
prior = class_counts / len(y_train)
# Bygg og tren en Gaussian Naive Bayes-modell
gnb = GaussianNB(priors=prior)
gnb.fit(X_train_lda, y_train)

# Gjør prediksjoner på testsettet
y_pred = gnb.predict(X_test_lda)

# Beregn nøyaktigheten på testsettet
accuracy = accuracy_score(y_test, y_pred)
print(f"Modellens nøyaktighet: {accuracy * 100:.2f}%")

rf_model = RandomForestClassifier(
    n_estimators=100,            # Antall trær i skogen
    max_depth=15,                # Maks dybde på trærne (unngå overfitting)
    min_samples_leaf=2,          # Minimum antall prøver for å splitte på en node
    max_features='sqrt',         # Antall funksjoner som vurderes for hver split
    random_state=42,             # Sikre reproduserbare resultater
    n_jobs=-1                    # Bruk alle tilgjengelige CPU-kjerner for raskere trening
)

# Tren modellen
rf_model.fit(X_train_lda, y_train)

# Gjør prediksjoner på testsettet
y_pred = rf_model.predict(X_test_lda)

# Beregn nøyaktigheten på testsettet
accuracy = accuracy_score(y_test, y_pred)
print(f"Random Forest-modellens nøyaktighet: {accuracy * 100:.2f}%")

# Opprett Logistic Regression-modellen
lr_model = LogisticRegression(
    solver='lbfgs',            # Standard solver for logistisk regresjon
    max_iter=1000,             # Øk antall iterasjoner for å sikre konvergens
    random_state=42
)

lr_model.fit(X_train,y_train)
y_pred = lr_model.predict(X_test)
acc = accuracy_score(y_test,y_pred)
print(f"logistic regression acc:{acc}")