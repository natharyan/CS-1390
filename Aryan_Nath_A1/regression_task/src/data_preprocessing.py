import pandas as pd
import numpy as np

def preprocess_data_regression(filepath):
    df = pd.read_csv(filepath)

    # filtered the datafram to remove rows with missing engine size or fuel consumption values
    df = df.sample(frac=1).reset_index(drop=True)

    df = df.dropna(subset=['ENGINE SIZE','COEMISSIONS ','FUEL CONSUMPTION'])

    # using engine size and coemissions
    X = df[['ENGINE SIZE','COEMISSIONS ']]
    y = df['FUEL CONSUMPTION']

    # standardize the data
    for i in range(X.shape[1]):
        mean = sum(X.iloc[:,i])/len(X.iloc[:,i])
        X.iloc[:,i] = (X.iloc[:,i] - mean)/np.std(X.iloc[:,i])

    return df, X, y