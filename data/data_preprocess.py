import numpy as np
import pandas as pd
import torch
from sklearn import preprocessing
import random

def data_preprocess(dataset_name, random_seed = 0):
    random.seed(random_seed)
    if dataset_name == 'mpg':
        dat = pd.read_csv('dataset/auto-mpg.csv', na_values = '?')
        dat = dat.drop(columns = ['car name'])
        y = dat['mpg']
        x = dat.drop(columns = ['mpg'])
        origin = x.pop('origin')
        x['USA'] = (origin == 1) * 1.0
        x['Europe'] = (origin == 2) * 1.0
        x['Japan'] = (origin == 3) * 1.0
        x = x.dropna()
        scaler = preprocessing.StandardScaler().fit(x)
        x = scaler.transform(x)

        ind = np.arange(x.shape[0])
        random.shuffle(ind)
        x = x[ind, :]
        y = y[ind]

        y_tensor = torch.tensor(y.values).float().reshape(-1,1)
        x_tensor = torch.tensor(x).float()
        
        y_train = y_tensor[:300]
        y_test = y_tensor[300:]
        x_train = x_tensor[:300]
        x_test = x_tensor[300:]
    
    return x_train, y_train, x_test, y_test
    