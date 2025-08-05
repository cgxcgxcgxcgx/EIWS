import pandas as pd
import numpy as np

def load_data(path):
    data = pd.read_csv(path)
    X = data.iloc[:, :-1].values
    y = data.iloc[:, -1].values
    return X, y
