import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch import optim
from torch.autograd import Variable

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


def train_model(X, Y, n_epoch = 1000, num_models = 5, hidden_layers = [20, 20], learning_rate = 0.003, tanh = False, exp_decay = 1, decay_stepsize = 1):
    N, input_size = X.shape
    gmm = GaussianMLP(inputs = input_size * num_models, outputs = num_models, hidden_layers=hidden_layers, tanh = tanh)

    optimizer = torch.optim.RMSprop(params=gmm.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=decay_stepsize, gamma=exp_decay)
    
    # in the first half of the training epochs, train the model without the calibration loss
    for epoch in range(n_epoch):
        optimizer.zero_grad()
        X_train = X
        Y_train = Y
        for i in range(num_models - 1):
            ind = torch.randperm(N)
            X_train = torch.cat([X_train, X[ind]], dim = 1)
            Y_train = torch.cat([Y_train, Y[ind]], dim = 1)
        mean, var = gmm(X_train)
        nllk_loss = NLLloss(Y_train, mean, var) #NLL loss
        if epoch == 0:
            print('initial loss: ',nllk_loss.item())
        nllk_loss.backward()
        optimizer.step()
    print('final loss: ', nllk_loss.item())
    
    return gmm
    
def test_model(model, X, num_models = 5):
    X_test = X
    for i in range(num_models - 1):
        X_test = torch.cat([X_test, X], dim = 1)
    mean_pred, var_pred = model(X_test)
    mean = mean_pred.mean(dim = 1)
    variance = (var_pred + mean_pred.pow(2)).mean(dim=1) - mean.pow(2)
    return mean.unsqueeze(1), variance.unsqueeze(1)