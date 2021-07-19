import numpy as np


def metric_mse(forecast_samples, targets):
    forecast_mean = np.mean(forecast_samples, -1)
    return np.mean((forecast_mean - targets) ** 2)


def metric_rmse(forecast_samples, targets):
    # definition from p9 in
    # "DeepAR: Probabilistic forecasting with autoregressive recurrent networks"
    # https://arxiv.org/abs/1704.04110
    numerator = np.sqrt(metric_mse(forecast_samples, targets))
    denom = np.abs(targets).mean()
    return numerator / denom


def metric_nd(forecast_samples, targets):
    # definition from p9 in
    # "DeepAR: Probabilistic forecasting with autoregressive recurrent networks"
    # https://arxiv.org/abs/1704.04110
    forecast_median = np.median(forecast_samples, -1)
    numerator = np.abs(forecast_median - targets).sum()
    denom = np.abs(targets).sum()
    return numerator / denom


def quantile_loss(rho=0.9):
    def q_loss(z, z_hat):
        # definition from p7 in
        # "Deep state space models for time series forecasting"
        # https://papers.nips.cc/paper/8004-deep-state-space-models-for-time-series-forecasting
        diff = z - z_hat
        return ((diff > 0) * rho + (diff <= 0) * (rho - 1)) * diff

    def fn(forecast_samples, targets):
        forecast_quantile = np.percentile(forecast_samples, rho * 100.0, -1)
        numerator = 2 * q_loss(targets, forecast_quantile).sum()
        denom = np.abs(targets).sum()

        return numerator / denom

    return fn


METRIC_FN = {
    "MSE": metric_mse,
    "RMSE": metric_rmse,
    "ND": metric_nd,
    "P50": quantile_loss(0.5),
    "P90": quantile_loss(0.9),
}


class Metrics:
    def __init__(self, metric_list=["MSE", "RMSE", "ND", "P50", "P90"]):
        self.metric_list = metric_list

    def __call__(self, forecast_samples, targets):
        """
        Compute the metrics in self.metric_list

        Args:
            forecast_samples: shape (batch, sample_len, num_samples)
            targets: shape (batch, sample_len)

        Returns:
            metric_dict: a dictionary, keys are the metric names and values
                are the mean of metrics across the batch                
        """
        return {
            metric: METRIC_FN[metric](forecast_samples, targets)
            for metric in self.metric_list
        }
