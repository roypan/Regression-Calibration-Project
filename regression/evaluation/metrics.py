import numpy as np
import math
import torch
from scipy.stats import norm
import matplotlib.pyplot as plt

def mape(val_true, val_pred):
    ind = (np.abs(val_true) > 1e-5)
    return np.mean(np.abs((val_pred[ind] - val_true[ind]) / val_true[ind]))

def rmse(val_true, val_pred):
    return np.sqrt(np.mean((val_pred - val_true) ** 2))

def nllk(val_true, mean_pred, var_pred):
    """ Negative log-likelihood loss function. """
    return (np.log(var_pred) + np.power(val_true - mean_pred, 2)/var_pred).mean()

def pcdf(mean_pred, var_pred, val):
    n = len(mean_pred)
    pcdf = np.zeros(n)
    for i in range(n):
        pcdf[i] = norm.cdf(val[i], loc = mean_pred[i], scale = np.sqrt(var_pred[i]))
    return pcdf

def abline(slope, intercept):
    """Plot a line from slope and intercept"""
    axes = plt.gca()
    x_vals = np.array(axes.get_xlim())
    y_vals = intercept + slope * x_vals
    plt.plot(x_vals, y_vals, '--')

def draw_pcdf(pcdf, step = 0.01):
    p = np.arange(0,1,step) + step
    cumulative_pcdf = np.zeros(len(p))
    for i in range(len(p)):
        cumulative_pcdf[i] = np.mean(pcdf <= p[i])

    fig, ax = plt.subplots()
    ax.scatter(cumulative_pcdf, p, c='black')
    abline(1, 0)
    plt.grid()
    plt.xlabel('Predicted cumulative distribution')
    plt.ylabel('Empirical cumulative distribution')
    plt.show()

def calibration_error(pcdf, step = 0.01):
    p = np.arange(0,1,step) + step
    cumulative_pcdf = np.zeros(len(p))
    for i in range(len(p)):
        cumulative_pcdf[i] = np.mean(pcdf <= p[i])
    return np.sum((cumulative_pcdf - p) ** 2)
    
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