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


class GaussianMixtureMLP(nn.Module):
    """ Gaussian mixture MLP which outputs are mean and variance.

    Attributes:
        models (int): number of models
        inputs (int): number of inputs
        outputs (int): number of outputs
        hidden_layers (list of ints): hidden layer sizes

    """
    def __init__(self, num_models=5, inputs=1, outputs=1, hidden_layers=[100], activation='relu', tanh=False):
        super(GaussianMixtureMLP, self).__init__()
        self.num_models = num_models
        self.inputs = inputs
        self.outputs = outputs
        self.hidden_layers = hidden_layers
        self.activation = activation
        self.tanh = tanh
        for i in range(self.num_models):
            model = GaussianMLP(inputs=self.inputs, 
                                outputs=self.outputs, 
                                hidden_layers=self.hidden_layers,
                                activation=self.activation,
                                tanh=self.tanh)
            setattr(self, 'model_'+str(i), model)
            
    def forward(self, x):
        # connect layers
        means = []
        variances = []
        for i in range(self.num_models):
            model = getattr(self, 'model_' + str(i))
            mean, var = model(x)
            means.append(mean)
            variances.append(var)
        means = torch.stack(means)
        mean = means.mean(dim=0)
        variances = torch.stack(variances)
        variance = (variances + means.pow(2)).mean(dim=0) - mean.pow(2)
        return mean, variance

def NLLloss(y, mean, var):
    """ Negative log-likelihood loss function. """
    return (torch.log(var) + torch.pow(y - mean, 2)/var).mean()


def train_model(X, Y, n_epoch = 1000, num_models = 5, hidden_layers = [20, 20], learning_rate = 0.003, tanh = False):
    N, input_size = X.shape
    gmm = GaussianMixtureMLP(num_models=num_models, inputs = input_size, hidden_layers=hidden_layers, tanh = tanh)

    for i in range(gmm.num_models):
        model = getattr(gmm, 'model_' + str(i))
        gmlp_optimizer = torch.optim.RMSprop(params=model.parameters(), lr=learning_rate)
        
        for epoch in range(n_epoch):
            gmlp_optimizer.zero_grad()
            mean, var = model(X)
            gmlp_loss = NLLloss(Y, mean, var) #NLL loss
            if epoch == 0:
                print('initial loss: ',gmlp_loss.item())
            gmlp_loss.backward()
            gmlp_optimizer.step()
        print('final loss: ',gmlp_loss.item())
    
    return gmm