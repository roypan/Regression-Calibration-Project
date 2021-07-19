import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import math
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch import optim
from torch.autograd import Variable
from sklearn.isotonic import IsotonicRegression

class MLP(nn.Module):
    """ Multilayer perceptron (MLP) with tanh/sigmoid activation functions implemented in PyTorch for regression tasks.

    Attributes:
        inputs (int): inputs of the network
        outputs (int): outputs of the network
        hidden_layers (list): layer structure of MLP: [5, 5] (2 hidden layer with 5 neurons)
        activation (string): activation function used ('relu', 'tanh' or 'sigmoid')

    """

    def __init__(self, inputs=1, outputs=1, hidden_layers=[100], activation='relu'):
        super(MLP, self).__init__()
        self.inputs = inputs
        self.outputs = outputs
        self.hidden_layers = hidden_layers
        self.nLayers = len(hidden_layers)
        self.net_structure = [inputs, *hidden_layers, outputs]
        
        if activation == 'relu':
            self.act = torch.relu
        elif activation == 'tanh':
            self.act = torch.tanh
        elif activation == 'sigmoid':
            self.act = torch.sigmoid
        else:
            assert('Use "relu","tanh" or "sigmoid" as activation.')
        # create linear layers y = Wx + b

        for i in range(self.nLayers + 1):
            setattr(self, 'layer_'+str(i), nn.Linear(self.net_structure[i], self.net_structure[i+1]))

    def forward(self, x):
        # connect layers
        for i in range(self.nLayers):
            layer = getattr(self, 'layer_'+str(i))
            x = self.act(layer(x))
        layer = getattr(self, 'layer_' + str(self.nLayers))
        x = layer(x)
        return x
        
class GaussianMLP(MLP):
    """ Gaussian MLP which outputs are mean and variance.

    Attributes:
        inputs (int): number of inputs
        outputs (int): number of outputs
        hidden_layers (list of ints): hidden layer sizes

    """

    def __init__(self, inputs=1, outputs=1, hidden_layers=[100], activation='relu', tanh=False):
        super(GaussianMLP, self).__init__(inputs=inputs, outputs=2*outputs, hidden_layers=hidden_layers, activation=activation)
        self.outputs = outputs
        self.tanh = tanh
    def forward(self, x):
        # connect layers
        for i in range(self.nLayers):
            layer = getattr(self, 'layer_'+str(i))
            x = self.act(layer(x))
        layer = getattr(self, 'layer_' + str(self.nLayers))
        x = layer(x)
        mean, variance = torch.split(x, self.outputs, dim=1)
        variance = torch.log(1 + torch.exp(variance))
        if self.tanh:
            mean = (torch.tanh(mean) + 1.0) / 2.0
        return mean, variance

def NLLloss(y, mean, var):
    """ Negative log-likelihood loss function. """
    return (torch.log(var) + torch.pow(y - mean, 2)/var).mean()

def pcdf(mean_pred, var_pred, val):
    n = len(mean_pred)
    pcdf = torch.zeros(n)
    for i in range(n):
        pcdf[i] = torch.distributions.normal.Normal(loc = mean_pred[i], scale = torch.sqrt(var_pred[i])).cdf(val[i])
    return pcdf
    
def ecdf(predicted_cdf):
    empirical_cdf = np.zeros(len(predicted_cdf))
    for i, p in enumerate(predicted_cdf):
        empirical_cdf[i] = np.sum(predicted_cdf <= p) / len(predicted_cdf)
    return empirical_cdf

def isotonic_regression(predicted_cdf, empirical_cdf):
    isotonic_model = IsotonicRegression(out_of_bounds='clip')
    isotonic_model.fit(predicted_cdf, empirical_cdf)
    return isotonic_model

def calibration_error(pcdf, step = 0.1):
    p = np.arange(0,1,step) + step
    cumulative_pcdf = np.zeros(len(p))
    for i in range(len(p)):
        cumulative_pcdf[i] = np.mean(pcdf <= p[i])
    return np.sum((cumulative_pcdf - p) ** 2)
    
def train_model_nllk(X, Y, n_epoch = 1000, num_models = 5, hidden_layers = [20, 20], learning_rate = 0.003, tanh = False, calibration_threshold = .05, exp_decay = 1, decay_stepsize = 1):
    N, input_size = X.shape
    gmm = GaussianMLP(inputs = input_size, hidden_layers=hidden_layers, tanh = tanh)

    optimizer = torch.optim.RMSprop(params=gmm.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=decay_stepsize, gamma=exp_decay)
    
    # in the first half of the training epochs, train the model without the calibration loss
    for epoch in range(n_epoch):
        optimizer.zero_grad()
        mean, var = gmm(X)
        nllk_loss = NLLloss(Y, mean, var) #NLL loss
        if epoch == 0:
            print('initial loss: ',nllk_loss.item())
        nllk_loss.backward()
        optimizer.step()
        scheduler.step()
        
    print('final loss: ', nllk_loss.item())
    
    return gmm

def train_isotonic_regression(model, X_eval, Y_eval):
    mean_eval, var_eval = model(X_eval)
    n = X_eval.shape[0]
    predicted_cdf = torch.zeros(n)
    for i in range(n):
        predicted_cdf[i] = torch.distributions.normal.Normal(loc = mean_eval[i], scale = torch.sqrt(var_eval[i])).cdf(Y_eval[i])
    
    predicted_cdf = predicted_cdf.detach().numpy()
    empirical_cdf = ecdf(predicted_cdf)
    isotonic_model = isotonic_regression(predicted_cdf, empirical_cdf)
    
    return isotonic_model

