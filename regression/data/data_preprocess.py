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
        dat = dat.dropna()
        dat = dat.reset_index(drop=True)
        y = dat['mpg']
        x = dat.drop(columns = ['mpg'])
        origin = x.pop('origin')
        x['USA'] = (origin == 1) * 1.0
        x['Europe'] = (origin == 2) * 1.0
        x['Japan'] = (origin == 3) * 1.0
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
        
    elif dataset_name == 'crimes':
        dat = pd.read_csv('dataset/crimes.csv', na_values = '?', header = None)
        dat = dat.loc[:, dat.isnull().mean() < .10]
        dat = dat.drop(dat.columns[[0,1,2]], axis = 1)
        dat = dat.dropna()
        dat = dat.reset_index(drop=True)
        y = dat.iloc[:, dat.shape[1] - 1]
        x = dat.iloc[:, :(dat.shape[1] - 1)]
        scaler = preprocessing.StandardScaler().fit(x)
        x = scaler.transform(x)
        
        ind = np.arange(x.shape[0])
        random.shuffle(ind)
        x = x[ind, :]
        y = y[ind]

        y_tensor = torch.tensor(y.values).float().reshape(-1,1)
        x_tensor = torch.tensor(x).float()

        y_train = y_tensor[:1500]
        y_test = y_tensor[1500:]
        x_train = x_tensor[:1500]
        x_test = x_tensor[1500:]
        
    elif dataset_name == 'cpu':
        dat = pd.read_csv('dataset/cpu.csv', na_values = '?', header = None)
        dat = dat.drop(dat.columns[[0,1,9]], axis = 1)
        y = dat.iloc[:, dat.shape[1] - 1]
        x = dat.iloc[:, :(dat.shape[1] - 1)]
        scaler = preprocessing.StandardScaler().fit(x)
        x = scaler.transform(x)

        ind = np.arange(x.shape[0])
        random.shuffle(ind)
        x = x[ind, :]
        y = y[ind]

        y_tensor = torch.tensor(y.values).float().reshape(-1,1)
        x_tensor = torch.tensor(x).float()

        y_train = y_tensor[:150]
        y_test = y_tensor[150:]
        x_train = x_tensor[:150]
        x_test = x_tensor[150:]
        
    elif dataset_name == 'housing':
        dat = pd.read_csv('dataset/housing.csv', header = None)
        y = dat.iloc[:, dat.shape[1] - 1]
        x = dat.iloc[:, :(dat.shape[1] - 1)]
        scaler = preprocessing.StandardScaler().fit(x)
        x = scaler.transform(x)

        ind = np.arange(x.shape[0])
        random.shuffle(ind)
        x = x[ind, :]
        y = y[ind]

        y_tensor = torch.tensor(y.values).float().reshape(-1,1)
        x_tensor = torch.tensor(x).float()

        y_train = y_tensor[:380]
        y_test = y_tensor[380:]
        x_train = x_tensor[:380]
        x_test = x_tensor[380:]
        
    elif dataset_name == 'concrete':
        dat = pd.read_csv('dataset/concrete.csv', header = None)
        y = dat.iloc[:, dat.shape[1] - 1]
        x = dat.iloc[:, :(dat.shape[1] - 1)]
        scaler = preprocessing.StandardScaler().fit(x)
        x = scaler.transform(x)

        ind = np.arange(x.shape[0])
        random.shuffle(ind)
        x = x[ind, :]
        y = y[ind]

        y_tensor = torch.tensor(y.values).float().reshape(-1,1)
        x_tensor = torch.tensor(x).float()

        y_train = y_tensor[:750]
        y_test = y_tensor[750:]
        x_train = x_tensor[:750]
        x_test = x_tensor[750:]
        
    elif dataset_name == 'kin8nm':
        dat = pd.read_csv('dataset/kin8nm.csv')
        y = dat.iloc[:, dat.shape[1] - 1]
        x = dat.iloc[:, :(dat.shape[1] - 1)]
        scaler = preprocessing.StandardScaler().fit(x)
        x = scaler.transform(x)

        ind = np.arange(x.shape[0])
        random.shuffle(ind)
        x = x[ind, :]
        y = y[ind]

        y_tensor = torch.tensor(y.values).float().reshape(-1,1)
        x_tensor = torch.tensor(x).float()

        y_train = y_tensor[:6100]
        y_test = y_tensor[6100:]
        x_train = x_tensor[:6100]
        x_test = x_tensor[6100:]
        
    elif dataset_name == 'yacht':
        dat = pd.read_csv('dataset/yacht.csv')
        y = dat.iloc[:, dat.shape[1] - 1]
        x = dat.iloc[:, :(dat.shape[1] - 1)]
        scaler = preprocessing.StandardScaler().fit(x)
        x = scaler.transform(x)

        ind = np.arange(x.shape[0])
        random.shuffle(ind)
        x = x[ind, :]
        y = y[ind]

        y_tensor = torch.tensor(y.values).float().reshape(-1,1)
        x_tensor = torch.tensor(x).float()

        y_train = y_tensor[:230]
        y_test = y_tensor[230:]
        x_train = x_tensor[:230]
        x_test = x_tensor[230:]
    
    return x_train, y_train, x_test, y_test
    