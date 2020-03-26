from variables import *
import numpy as np
import pandas as pd

def read_csv_files(csv_file):
        data = pd.read_csv(csv_file)
        data = data.dropna(axis = 0, how ='any')
        Y = data.iloc[:,0]
        X = data.iloc[:,1:] / 255.0
        return X, Y

def get_data():
    Xtrain, Ytrain = read_csv_files(train_path)
    Xtest , Ytest  = read_csv_files(test_path)
    Xtrain, Ytrain = Xtrain[:5000], Ytrain[:5000]
    return Xtrain, Ytrain, Xtest , Ytest