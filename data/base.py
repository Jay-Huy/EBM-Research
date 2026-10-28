"""Shared dataset interface and registry `get_dataset(name, split)`.

Every dataset subclasses `BaseDataset` and defines:
    - SPLITS      : its splits, e.g. ("train", "test_id", "test_ood")
    - input_len   : number of input tokens (N_x)
    - output_len  : number of label tokens (N)
    - __getitem__ : {"inputs": Tensor(N_x), "labels": Tensor(N)}
                    -> after DataLoader: [B, N_x], [B, N]
    - metric(pred, labels) -> dict : metric used for comparison
    - sample_scores(pred, labels) -> dict of (B,) tensors, metric_from_scores(scores) -> dict : the same metric from
                    per-sample sums, so inference.py gets the exact split metric at every step without keeping the
                    predictions of every step (Phase C.5 Task 8)
    - BEST_METRIC : (key, "max" | "min"), test_id/<key> that picks best_checkpoint.pt: a metric key or "loss"
                    (Phase C.5 Task 6; Sudoku: loss, Task 9.8)
    - PLOT_METRIC : ({metric key: line label}, panel title, axis label, y limits or None), the metric lines drawn by
                    inference.py (curves.png); the first key is the main one (drawn on top, bold)
    - format_sample(inputs, pred, labels) -> dict | None : one sample as text for the W&B sample table
                    (Phase C.5 Task 4.3, 5.8); None (default): the task has no sample table yet
    - download()  : optional; only for datasets that must be downloaded (into data/<task>/raw/)
"""
import torch
from torch.utils.data import Dataset

_REGISTRY = {}


def register_dataset(name):
    """Class decorator: register a dataset under `name`."""
    def deco(cls):
        _REGISTRY[name] = cls
        return cls
    return deco


def get_dataset(name, split, **kwargs):
    """Build the dataset registered as `name` for the given `split`."""
    if name not in _REGISTRY:
        raise KeyError(f"dataset '{name}' is not registered; available: {sorted(_REGISTRY)}")
    return _REGISTRY[name](split, **kwargs)


def list_datasets():
    return sorted(_REGISTRY)


class BaseDataset(Dataset):
    SPLITS = ()
    BEST_METRIC = None
    PLOT_METRIC = None
    input_len = None
    output_len = None

    def __init__(self, split):
        if split not in self.SPLITS:
            raise ValueError(f"{type(self).__name__}: invalid split '{split}'; available: {self.SPLITS}")
        self.split = split

    def metric(self, pred, labels):
        raise NotImplementedError

    def sample_scores(self, pred, labels):
        raise NotImplementedError

    def metric_from_scores(self, scores):
        raise NotImplementedError

    def format_sample(self, inputs, pred, labels):
        """One sample (inputs (N_x), pred and labels (N), in the metric format) as a row of text; None: not written."""
        return None

    def download(self):
        pass


def mse(pred, labels):
    """Element-wise MSE, averaged over the whole batch."""
    return {"mse": torch.mean((pred - labels) ** 2).item()}
