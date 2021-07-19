import logging
import os

import matplotlib.pyplot as plt
import numpy as np

plt.switch_backend("agg")


def plot_forecast(sample_seqs, context, target, log_path, level=0.9, num_plots=10):
    context_len = context.shape[1]
    target_len = target.shape[1]

    pred_med = np.median(sample_seqs, -1)
    pred_ub = np.percentile(sample_seqs, 100.0 * (1.0 + level) / 2, -1)
    pred_lb = np.percentile(sample_seqs, 100.0 * (1.0 - level) / 2, -1)

    plt.style.use("ggplot")
    colors = [d["color"] for d in iter(plt.rcParams["axes.prop_cycle"])]

    logging.info(
        f"Creating plots: log_path={log_path}, level={level}, num_plots={num_plots}"
    )
    for i in range(num_plots):
        plt.axvline(
            x=context_len, linestyle="dashed", linewidth=0.5, color="black", alpha=0.8
        )
        for seq_idx in range(sample_seqs.shape[-1]):
            plt.plot(
                np.arange(context_len, context_len + target_len),
                sample_seqs[i, :, seq_idx],
                alpha=0.05,
                color=colors[3],
            )
        plt.plot(
            np.arange(context_len + target_len),
            np.concatenate((context[i], target[i])),
            color=colors[0],
            alpha=0.8,
        )
        plt.plot(
            np.arange(context_len, context_len + target_len),
            pred_med[i],
            alpha=0.8,
            color=colors[1],
        )
        plt.plot(
            np.arange(context_len, context_len + target_len),
            pred_ub[i],
            alpha=0.8,
            color=colors[1],
        )
        plt.plot(
            np.arange(context_len, context_len + target_len),
            pred_lb[i],
            alpha=0.8,
            color=colors[1],
        )

        plt.savefig(os.path.join(log_path, f"forecast_{i}.png"))
        plt.clf()


def create_log_path(expr_name):
    log_path = os.path.join("results", expr_name)
    if not os.path.exists(log_path):
        os.makedirs(log_path)

    return log_path


def create_logger(log_path):
    logger = logging.getLogger()
    log_level = logging.INFO
    logger.setLevel(log_level)

    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

    file_handler = logging.FileHandler(os.path.join(log_path, "log.txt"), mode="w")
    file_handler.setLevel(log_level)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    return logger
