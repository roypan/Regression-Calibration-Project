import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt

def mape(val_true, val_pred):
    ind = (np.abs(val_true) > 1e-5)
    return np.mean(np.abs((val_pred[ind] - val_true[ind]) / val_true[ind]))

def rmse(val_true, val_pred):
    return np.sqrt(np.mean((val_pred - val_true) ** 2))

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