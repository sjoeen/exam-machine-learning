import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler


train_data = pd.read_csv('train.csv')
train_data = train_data.iloc[:, 1:-1] 
    #remove label and id
id = pd.read_csv('train.csv')
id_value = id.iloc[:, 0]
print(id_value.shape)
y = id.iloc[:, -1].values


"""
removes all features that have the same values across the board
"""
constant_value_features = []
    #NOTE also remove from train!!
for i in train_data.columns:
    if train_data[i].nunique() == 1:  
        constant_value_features.append(i)



train_data_cleaned = train_data.drop(columns=constant_value_features)

# Beregn Q1 (25. percentil) og Q3 (75. percentil)
Q1 = train_data_cleaned.quantile(0.025)
Q3 = train_data_cleaned.quantile(0.975)

# Beregn interkvartilavstanden (IQR)
IQR = Q3 - Q1

# Identifiser outliers
outliers = (train_data_cleaned < (Q1 - 1.5 * IQR)) | (train_data_cleaned > (Q3 + 1.5 * IQR))

# Print hvilke kolonner som har outliers og hvor mange
outlier_columns = outliers.sum()
print(f"Kolonner med outliers og antall outliers i hver kolonne:\n{outlier_columns[outlier_columns > 0]}")

# Finn rader som har minst én outlier
samples_with_outliers = train_data_cleaned[outliers.any(axis=1)]

# Print ut hvilke samples som har outliers
print("Samples som har outliers:")
print(samples_with_outliers)
train_data_cleaned = train_data_cleaned[~outliers.any(axis=1)]
y = y[~outliers.any(axis=1)]
id = id[~outliers.any(axis=1)]


# Print ut resultatet
print(f"Antall samples etter fjerning: {train_data_cleaned.shape[0]}")

"""
next up: replace all zeros (assumed as missing values) with mean values of thir
respective coloumns.
"""

for i in train_data_cleaned.columns:
    train_data_cleaned[i] = train_data_cleaned[i].replace(0, train_data_cleaned[i].median())


"""
scale all the features so they have the same input on the ML model(s)
"""
scaler = StandardScaler()
train_data_scaled = scaler.fit_transform(train_data_cleaned)

#Konverter tilbake til DataFrame og behold kolonnenavnene
train_data_scaled = pd.DataFrame(train_data_scaled, columns=train_data_cleaned.columns)

# Sjekk de skalerte dataene
print("\nhead and shape afterwards:")
print(train_data_scaled.head())
print(train_data_scaled.shape)

train_data_scaled.to_csv('train_cleaned.csv', index=False)
    #NOTE, is missing id and labels
y = pd.DataFrame(id, columns=["lipophilicity"])
y.to_csv('lipophilicity_cleaned.csv', index=False)
id.to_csv('id_cleaned.csv', index=False)

print(id.shape)
print(train_data_scaled.shape)