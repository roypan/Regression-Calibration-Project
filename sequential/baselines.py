import argparse
import logging

import os
import gluonts
import mxnet as mx
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from gluonts.dataset.common import ListDataset
from gluonts.mx.trainer import Trainer
from gluonts.evaluation.backtest import make_evaluation_predictions

from dataset import get_dataset
from metrics import Metrics

parser = argparse.ArgumentParser()
parser.add_argument(
    "--model",
    type=str,
    default="deepar",
    help="type of model (default: deepar), deepar, deepstate, rnn",
)
parser.add_argument("--epochs", type=int, default=10, help="number of training epochs")
parser.add_argument("--cxt_len", type=int, default=336, help="number of context length")
parser.add_argument(
    "--pred_len", type=int, default=168, help="number of prediction length"
)
parser.add_argument(
    "--save_plots", action="store_true", default=True, help="save the results"
)

args = parser.parse_args()

mx.random.seed(0)
np.random.seed(0)

LOG_FILENAME = r'logging.log'

def plot_prob_forecasts(
    ts_entry, forecast_entry, save_fig, time_series_index, model_dir
):
    plot_length = 336
    prediction_intervals = (50.0, 90.0)
    legend = ["observations", "median prediction"] + [
        f"{k}% prediction interval" for k in prediction_intervals
    ][::-1]

    fig, ax = plt.subplots(1, 1, figsize=(10, 7))
    ts_entry[-plot_length:].plot(ax=ax)  # plot the time series
    forecast_entry.plot(prediction_intervals=prediction_intervals, color="g")
    plt.grid(which="both")
    plt.legend(legend, loc="upper left")

    if save_fig == True:
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)

        fig_name = os.path.join(model_dir, str(time_series_index) + ".png")
        print("save fig_name: ", fig_name)

        plt.savefig(fig_name)
    plt.show()


if __name__ == "__main__":
    _, _, data_np = get_dataset(name="electricity", window_len=1, cxt_len=1, pred_len=1)

    freq = "1H"
    start = pd.Timestamp("2012-01-01", freq=freq)

    train_ds = ListDataset(
        [{"target": x, "start": start} for x in data_np[:, : -args.pred_len]], freq=freq
    )
    test_ds = ListDataset([{"target": x, "start": start} for x in data_np], freq=freq)

    if args.model == "deepar":
        from gluonts.model.deepar import DeepAREstimator

        estimator = DeepAREstimator(
            freq=freq,
            prediction_length=args.pred_len,
            context_length=args.cxt_len,
            num_layers=1,
            num_cells=32,
            cell_type="lstm",
            trainer=Trainer(ctx=mx.cpu(), epochs=args.epochs),
        )
    elif args.model == "deepstate":
        from gluonts.model.deepstate import DeepStateEstimator

        estimator = DeepStateEstimator(
            freq=freq,
            prediction_length=args.pred_len,
            past_length=args.cxt_len,
            num_layers=1,
            num_cells=32,
            cell_type="lstm",
            cardinality=[],
            use_feat_static_cat=False,
            # trainer = Trainer(ctx=mx.gpu(0),epochs=args.epochs)
            trainer=Trainer(ctx=mx.cpu(), epochs=args.epochs),
        )
    elif args.model == "rnn":
        from gluonts.model.canonical import CanonicalRNNEstimator

        estimator = CanonicalRNNEstimator(
            freq="1H",
            prediction_length=168,
            context_length=336,
            num_layers=1,
            num_cells=32,
            cell_type="lstm",
            trainer=Trainer(ctx=mx.cpu(), epochs=args.epochs),
        )
    elif args.model == "transformer":
        from gluonts.model.transformer import TransformerEstimator

        estimator = TransformerEstimator(
            freq="1H",
            prediction_length=168,
            context_length=336,
            trainer=Trainer(ctx=mx.cpu(), epochs=args.epochs),
        )

    predictor = estimator.train(train_ds)
    forecast_it, ts_it = make_evaluation_predictions(
        dataset=test_ds,  # test dataset
        predictor=predictor,  # predictor
        num_samples=100,  # number of sample paths we want for evaluation
    )
    forecasts = list(forecast_it)
    tss = list(ts_it)
    forecasts_np = np.stack([np.transpose(entry.samples) for entry in forecasts])

    metrics = Metrics()
    metric_dict = metrics(
        forecast_samples=forecasts_np, targets=data_np[:, -args.pred_len :]
    )

    logging.info("====> Test metrics:")
    for key, val in metric_dict.items():
        logging.info(f"\t{key}: {val:.3f}")
    logging.basicConfig(filename=LOG_FILENAME, level=logging.info)
    print(metric_dict)

    if args.save_plots:
        plot_index = 0  ## Choose which time series to plot
        plot_prob_forecasts(
            tss[plot_index],
            forecasts[plot_index],
            args.save_plots,
            plot_index,
            args.model,
        )
