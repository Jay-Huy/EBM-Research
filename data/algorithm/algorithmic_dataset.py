"""Three matrix toy tasks: Inverse, LowRank, Addition.

Source:
    - IRED: Learning Iterative Reasoning through Energy Diffusion (Du, Mao, Tenenbaum, ICML 2024)
    - URL: https://energy-based-model.github.io/ired/ired.pdf
    - repo https://github.com/yilundu/ired_code_release, file dataset.py (classes Inverse, LowRankDataset, Addition)

Notes:
    - The data-generation logic is kept as in the original (ID / OOD, how inputs / labels are built);
      only the structure is changed to follow BaseDataset.
    - Data is sampled at random on every call, without a seed, as in the original code.
    - Splits: "train" and "test_id" are generated identically; "test_ood" is the harder distribution.
"""
import numpy as np
import torch

from data.base import BaseDataset, mse, register_dataset

SPLITS = ("train", "test_id", "test_ood")


class _MatrixDataset(BaseDataset):
    SPLITS = SPLITS
    BEST_METRIC = ("mse", "min")
    PLOT_METRIC = ({"mse": "MSE"}, "MSE", "MSE", None)

    def __init__(self, split, rank=20):
        super().__init__(split)
        self.h = self.w = rank
        self.ood = split == "test_ood"

    def __len__(self):
        return int(1e7)                       # as in the original code

    def metric(self, pred, labels):
        return mse(pred, labels)

    def sample_scores(self, pred, labels):
        """Per sample: mean squared error (every sample has the same number of entries, so the mean over samples is mse)."""
        return {"sq_err": ((pred - labels) ** 2).flatten(1).mean(1)}

    def metric_from_scores(self, scores):
        return {"mse": float(scores["sq_err"].mean())}

    @staticmethod
    def _pack(inputs, labels):
        return {"inputs": torch.from_numpy(inputs).float(), "labels": torch.from_numpy(labels).float()}


@register_dataset("inverse")
class Inverse(_MatrixDataset):
    """Matrix inverse.

    - inputs: M = R R^T + (R R^T)^T + c I, flattened (400)
    - labels: M^-1, flattened (400)
    - c = 0.5 (ID), 0.1 (OOD: less well-conditioned)
    """

    def __init__(self, split, rank=20):
        super().__init__(split, rank)
        self.input_len = self.output_len = self.h * self.w

    def __getitem__(self, index):
        R = np.random.uniform(-1, 1, (self.h, self.w))
        R = R.dot(R.transpose())
        c = 0.1 if self.ood else 0.5
        M = R + R.transpose() + c * np.eye(self.h)
        return self._pack(M.flatten(), np.linalg.inv(M).flatten())


@register_dataset("lowrank")
class LowRank(_MatrixDataset):
    """Low-rank matrix completion.

    - labels: R = 0.1 N + U V^T / s, flattened (400); U, V ~ N(0, 1), rank 10
    - inputs: R with ~50% of the entries masked to 0, flattened (400)
    - s = 20 (ID), 5 (OOD: larger values)
    """

    def __init__(self, split, rank=20):
        super().__init__(split, rank)
        self.input_len = self.output_len = self.h * self.w

    def __getitem__(self, index):
        U = np.random.randn(self.h, 10)
        V = np.random.randn(self.w, 10)
        s = 5 if self.ood else 20
        R = 0.1 * np.random.randn(self.h, self.w) + np.dot(U, V.T) / s
        mask = np.round(np.random.rand(self.h, self.w))
        return self._pack((R * mask).flatten(), R.flatten())


@register_dataset("addition")
class Addition(_MatrixDataset):
    """Matrix addition.

    - inputs: [A, B] concatenated, flattened (800)
    - labels: A + B, flattened (400)
    - entries in [-1, 1] (ID), [-2.5, 2.5] (OOD)
    """

    def __init__(self, split, rank=20):
        super().__init__(split, rank)
        if self.h == 2:                       # as in the original code
            self.w = 1
        self.input_len = 2 * self.h * self.w
        self.output_len = self.h * self.w

    def __getitem__(self, index):
        scale = 2.5 if self.ood else 1.0
        A = np.random.uniform(-scale, scale, (self.h, self.w)).flatten()
        B = np.random.uniform(-scale, scale, (self.h, self.w)).flatten()
        return self._pack(np.concatenate([A, B], axis=0), A + B)
