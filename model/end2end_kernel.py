import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import math
import random
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
        feature = x
        layer = getattr(self, 'layer_' + str(self.nLayers))
        x = layer(x)
        mean, variance = torch.split(x, self.outputs, dim=1)
        variance = torch.log(1 + torch.exp(variance))
        if self.tanh:
            mean = (torch.tanh(mean) + 1.0) / 2.0
        return mean, variance, feature

class CRPSMetric:
    """
    Compute the Continuous Ranked Probability Score (CRPS) is one of the most widely used error metrics to evaluate
    the quality of probabilistic regression tasks.
    Original paper: Gneiting, T. and Raftery, A.E., 2007. Strictly proper scoring rules, prediction, and estimation.
    Journal of the American statistical Association, 102(477), pp.359-378.
    https://sites.stat.washington.edu/raftery/Research/PDF/Gneiting2007jasa.pdf
    Many distributions have closed form solution:
    https://mran.microsoft.com/snapshot/2017-12-13/web/packages/scoringRules/vignettes/crpsformulas.html
    """
    def __init__(self, x, loc, scale):
        self.value = x
        self.loc = loc
        self.scale = scale
    def gaussian_pdf(self, x):
        """Probability density function of a univariate standard Gaussian
            distribution with zero mean and unit variance.
        """
        _normconst = 1.0 / math.sqrt(2.0 * math.pi)
        return _normconst * torch.exp(-(x * x) / 2.0)
    def gaussian_cdf(self, x):
        return 0.5 * (1 + torch.erf(x / math.sqrt(2)))
    def laplace_crps(self):
        """
        Compute the CRPS of observations x relative to laplace distributed forecasts with mean and b.
        Formula taken from Equation 2.1 Laplace distribution
        https://mran.microsoft.com/snapshot/2017-12-13/web/packages/scoringRules/vignettes/crpsformulas.html
        Returns:
        ----------
        crps: torch tensor
        The CRPS of each observation x relative to loc and scale.
        """
        # standadized value
        sx = (self.value - self.loc) / self.scale
        crps = self.scale * (sx.abs() + torch.exp(-sx.abs()) - 0.75)
        return crps
    def gaussian_crps(self):
        """
        Compute the CRPS of observations x relative to gaussian distributed forecasts with mean, sigma.
        CRPS(N(mu, sig^2); x)
        Formula taken from Equation (5):
        Calibrated Probablistic Forecasting Using Ensemble Model Output
        Statistics and Minimum CRPS Estimation. Gneiting, Raftery,
        Westveld, Goldman. Monthly Weather Review 2004
        http://journals.ametsoc.org/doi/pdf/10.1175/MWR2904.1
        Returns:
        ----------
        crps:
        The CRPS of each observation x relative to loc and scale.
        """
        # standadized value
        sx = (self.value - self.loc) / self.scale
        pdf = self.gaussian_pdf(sx)
        cdf = self.gaussian_cdf(sx)
        pi_inv = 1.0 / math.sqrt(math.pi)
        # the actual crps
        crps = self.scale * (sx * (2 * cdf - 1) + 2 * pdf - pi_inv)
        return crps

def NLLloss(y, mean, var):
    """ Negative log-likelihood loss function. """
    return (torch.log(var) + torch.pow(y - mean, 2)/var).mean()

def pcdf(mean_pred, var_pred, val):
    n = len(mean_pred)
    pcdf = torch.zeros(n)
    for i in range(n):
        pcdf[i] = torch.distributions.normal.Normal(loc = mean_pred[i], scale = torch.sqrt(var_pred[i])).cdf(val[i])
    return pcdf

def calibration_error(pcdf, step = 0.1):
    p = np.arange(0,1,step) + step
    cumulative_pcdf = np.zeros(len(p))
    for i in range(len(p)):
        cumulative_pcdf[i] = np.mean(pcdf <= p[i])
    return np.sum((cumulative_pcdf - p) ** 2)
    
def calibration_loss(y, p, mean_pred, var_pred):
    loss = 0
    m = len(p)
    n = len(y)
    for i in range(m):
        pq = torch.zeros(n)
        for j in range(n):
            pq[j] = torch.distributions.normal.Normal(loc = mean_pred[j], scale = torch.sqrt(var_pred[j])).icdf(p[i])
        average_coverage = torch.mean((y <= pq).float())
        if average_coverage < p[i]:
            loss += 1/m * torch.mean((y - pq) ** 2 * (y > pq))
        else:
            loss += 1/m * torch.mean((pq - y) ** 2 * (y < pq))
    return loss
    
def contrastive_loss_proportional(x, feature):
    """
    x: k by 3 by d tensor
    """
    distance1 = (x[:, 0, :] - x[:, 1, :]).norm(dim = 1) / (feature[:, 0, :] - feature[:, 1, :]).norm(dim = 1)
    distance2 = (x[:, 0, :] - x[:, 2, :]).norm(dim = 1) / (feature[:, 0, :] - feature[:, 2, :]).norm(dim = 1)
    loss = (torch.abs(distance1 - distance2) ** 2).mean()
    return loss

def contrastive_loss(x, feature):
    """
    x: k by 3 by d tensor
    """
    loss = 0
    distance_x01 = (x[:, 0, :] - x[:, 1, :]).norm(dim = 1)
    distance_x02 = (x[:, 0, :] - x[:, 2, :]).norm(dim = 1)
    distance_x12 = (x[:, 1, :] - x[:, 2, :]).norm(dim = 1)
    distance_f01 = (feature[:, 0, :] - feature[:, 1, :]).norm(dim = 1)
    distance_f02 = (feature[:, 0, :] - feature[:, 2, :]).norm(dim = 1)
    distance_f12 = (feature[:, 1, :] - feature[:, 2, :]).norm(dim = 1)
    loss += torch.sum(((distance_x01 - distance_x02) * (distance_f01 - distance_f02) < 0) * torch.abs(distance_f01 - distance_f02))
    loss += torch.sum(((distance_x01 - distance_x12) * (distance_f01 - distance_f12) < 0) * torch.abs(distance_f01 - distance_f12))
    loss += torch.sum(((distance_x12 - distance_x02) * (distance_f12 - distance_f02) < 0) * torch.abs(distance_f12 - distance_f02))
    return loss

def train_model_kernel(X, Y, n_epoch = 1000, num_models = 5, hidden_layers = [20, 20], n_unif = 10, n_pairs = 100, learning_rate = 0.003, tanh = False, calibration_threshold = .05, exp_decay = 1, decay_stepsize = 1):
    N, input_size = X.shape
    gmm = GaussianMLP(inputs = input_size, hidden_layers=hidden_layers, tanh = tanh)

    optimizer = torch.optim.RMSprop(params=gmm.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=decay_stepsize, gamma=exp_decay)
    
    for epoch in range(n_epoch):
        optimizer.zero_grad()
        mean, var, feature = gmm(X)
        nllk_loss = NLLloss(Y, mean, var) #NLL loss
        
        #predicted_cdf = pcdf(mean.squeeze(dim = 1), var.squeeze(dim = 1), Y)
        #p = torch.FloatTensor(n_unif).uniform_(0, 1)
        #cal_loss = calibration_loss(Y.squeeze(dim = 1), p, mean.squeeze(dim = 1), var.squeeze(dim = 1)) # calibration loss
        
        index = [random.sample(range(N), 3) for _ in range(n_pairs)]
        kernel_loss = contrastive_loss(X[index, :], feature[index, :])
        
        loss = nllk_loss + 0.25 * kernel_loss
        if epoch == 0:
            print('initial loss: ',loss.item())
        #print('cal loss: ', cal_loss.item(), 'cal error:', calibration_error(predicted_cdf.detach().numpy()), 'nllk loss: ', nllk_loss, 'kernel loss:', kernel_loss)
        loss.backward()
        optimizer.step()
        scheduler.step()
        
    print('final loss: ', loss.item())
    
    return gmm