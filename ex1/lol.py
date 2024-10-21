import pandas as pd

X = pd.read_csv('train.csv')


for column in X.columns:

    print(column)