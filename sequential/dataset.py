import os

import numpy as np
from torch.utils.data import Dataset


def get_dataset(name, window_len, cxt_len, pred_len):
    """ 
    Get train and test datasets

    Args:
        name: name of the dataset
        window_len: training window length at training time
        cxt_len: context length at test time
        pred_len: prediction length at test time

    Returns:
        train_dataset: train dataset
        test_dataset: test dataset
    """
    if name == "electricity":
        data_np = np.genfromtxt(
            os.path.join("data", "electricity.txt"), delimiter=",", dtype=np.float32
        )
        data_np = np.transpose(data_np)
        assert data_np.shape == (321, 26304)
        train_dataset = TrainDataset(data_np, window_len, pred_len, seasonality=24)
        test_dataset = TestDataset(data_np, cxt_len, pred_len, seasonality=24)
    elif name == "synth_linear":
        data_np = gen_synth_data(
            "linear", seq_len=10 * window_len + pred_len, num_seqs=1000
        )
        train_dataset = TrainDataset(data_np, window_len, pred_len, seasonality=None)
        test_dataset = TestDataset(data_np, cxt_len, pred_len, seasonality=None)
    elif name == "synth_seasonal":
        data_np = gen_synth_data(
            "seasonal", seq_len=10 * window_len + pred_len, num_seqs=1000
        )
        train_dataset = TrainDataset(data_np, window_len, pred_len, seasonality=5)
        test_dataset = TestDataset(data_np, cxt_len, pred_len, seasonality=5)
    else:
        raise NotImplementedError

    return train_dataset, test_dataset, data_np


class TrainDataset(Dataset):
    def __init__(self, data_np, window_len, pred_len, seasonality):
        """
        Constructor

        Args:
            data_np: a numpy ndarray, each row is a sequence.
            window_len: training window length at training time
            pred_len: prediction length at test time
            seasonality: seasonality
        """
        self.num_rows, self.num_cols = data_np.shape
        self.num_seqs_per_row = self.num_cols - pred_len - window_len + 1
        self.data_np = data_np
        self.window_len = window_len

        self.num_series = self.num_rows
        self.seq_len = self.num_cols - pred_len
        self.seasonality = seasonality

    def __getitem__(self, index):
        """
        Args:
            index

        Returns:
            idx: index of the sequence, int
            init_ts: initial time step, int
            seq: the sequence, shape (window_len, )
        """
        idx = index // self.num_seqs_per_row
        init_ts = index % self.num_seqs_per_row
        return idx, init_ts, self.data_np[idx, init_ts : init_ts + self.window_len]

    def __len__(self):
        return self.num_seqs_per_row * self.num_series


class TestDataset(Dataset):
    def __init__(self, data_np, cxt_len, pred_len, seasonality):
        """
        Constructor

        Args:  
            data_np: a numpy ndarray, each row is a sequence.
            cxt_len: context length at test time
            pred_len: prediction length at test time
            seasonality: seasonality
        """
        self.num_rows, self.num_cols = data_np.shape
        self.data_np = data_np
        self.cxt_len = cxt_len
        self.pred_len = pred_len

        self.num_series = self.num_rows
        self.seq_len = cxt_len + pred_len
        self.seasonality = seasonality

    def __getitem__(self, index):
        """
        Args:
            index

        Returns:
            idx: index of the sequence, int
            init_ts: initial time step, int
            context: context sequence, shape (cxt_len, )
            prediction: prediction sequence, shape (pred_len, )
        """
        init_ts = self.num_cols - self.seq_len
        context = self.data_np[index, -self.seq_len : -self.pred_len]
        prediction = self.data_np[index, -self.pred_len :]
        return index, init_ts, context, prediction

    def __len__(self):
        return self.num_series


def gen_synth_data(name, seq_len=100, num_seqs=50, seasonality=5):
    if name == "linear":
        slope = np.sqrt(seq_len) * np.random.standard_normal([num_seqs, 1])
        time = np.arange(seq_len)
        linear = slope * time
        noise = seq_len * np.random.standard_normal([num_seqs, seq_len])
        data_np = linear + noise
    elif name == "seasonal":
        slope = np.sqrt(seq_len) * np.random.standard_normal([num_seqs, 1])
        time = np.arange(seq_len)
        linear = slope * time
        noise = seq_len * np.random.standard_normal([num_seqs, seq_len])
        seasonal = 10.0 * seq_len * np.sin(time * 2.0 * np.pi / seasonality)
        data_np = linear + noise + seasonal
    else:
        raise NotImplementedError

    data_np = data_np.astype(np.float32)
    return data_np
