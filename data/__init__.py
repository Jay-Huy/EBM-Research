"""Data: dataset classes, builders, task metrics (Phase C.2)."""
from data.base import BaseDataset, get_dataset, list_datasets
import data.algorithm  # noqa: F401  (registers inverse, lowrank, addition)
import data.sudoku  # noqa: F401  (registers sudoku_rrn)
import data.cifar  # noqa: F401  (registers cifar10, cifar100)
