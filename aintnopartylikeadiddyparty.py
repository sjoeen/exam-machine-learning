import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler


test_data = pd.read_csv('test.csv')
print("test data set before cleaning:")
print(test_data.head())
print(test_data.shape)

test_data = test_data.iloc[:, 1:] 
    #remove id, for the scaling. 

removed_features = ['NumRadicalElectrons', 'SMR_VSA8', 'SlogP_VSA9', 'fr_azide', 'fr_diazo',
                    'fr_isocyan', 'fr_isothiocyan', 'fr_nitroso', 'fr_prisulfonamd',
                      'fr_thiocyan']
    #this is the features that was decided to remove from the train dataset. 

test_data_cleaned = test_data.drop(columns=removed_features)

"""
next up: replace all zeros (assumed as missing values) with mean values of thir
respective coloumns.
"""

for i in test_data_cleaned.columns:
    test_data_cleaned[i] = test_data_cleaned[i].replace(0, test_data_cleaned[i].median())
"""
scale all the features so they have the same input on the ML model(s)
"""
scaler = StandardScaler()
test_data_scaled = scaler.fit_transform(test_data_cleaned)

# Konverter tilbake til DataFrame og behold kolonnenavnene
test_data_scaled = pd.DataFrame(test_data_scaled, columns=test_data_cleaned.columns)

# Sjekk de skalerte dataene
print("\nhead and shape afterwards:")
print(test_data_scaled.head())
print(test_data_scaled.shape)

test_data_scaled.to_csv('test_cleaned.csv', index=False)
    #NOTE, is missing id and labels
