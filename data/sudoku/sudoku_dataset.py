"""Sudoku-RRN: hard 9x9 Sudoku boards (17-34 given digits).

Source:
    - data: Recurrent Relational Networks (Palm, Paquet, Winther, NeurIPS 2018), file sudoku-hard.zip
    - set up: Hyper-SET (ICLR 2026), repo hyper-set, file sudoku/dataset.py (class Sudoku_Dataset_Palm)

Notes:
    - The same sudoku-hard.zip is used by IRED (data/download-rrn.sh).
    - Raw files: data/sudoku/raw/sudoku-hard/{train,valid,test}.csv
      (2 columns, no header: board with 0 = empty, solution).
    - Splits, as in Hyper-SET (configs/task/sudoku.yaml: n_train 9000, n_test 1000, seed 0):
        - "train": permutation of train.csv with RandomState(0), first 9000 boards
        - "test_id": permutation of test.csv with RandomState(0), first 1000 boards
          (named test_id like the matrix tasks; Sudoku has no OOD split, decided in Phase C.5 Task 5.1)
    - valid.csv is not used, as in Hyper-SET.
    - The data seed is fixed to 0; it is independent of the training seed.
"""
import csv
import shutil
import urllib.request
import zipfile

import numpy as np
import torch

from data.base import BaseDataset, register_dataset
from utils.logger import REPO_ROOT

URL = "https://www.dropbox.com/s/rp3hbjs91xiqdgc/sudoku-hard.zip?dl=1"   # as in Hyper-SET and IRED
RAW_DIR = REPO_ROOT / "data" / "sudoku" / "raw"
CSV_DIR = RAW_DIR / "sudoku-hard"
CSV_FILES = ("train.csv", "valid.csv", "test.csv")

SPLIT_FILE = {"train": "train.csv", "test_id": "test.csv"}
SPLIT_SIZE = {"train": 9000, "test_id": 1000}
DATA_SEED = 0
IGNORE = -100                                 # label of given cells (ignore_index of cross-entropy)


@register_dataset("sudoku_rrn")
class SudokuRRN(BaseDataset):
    """Sudoku-RRN.

    - inputs: 81 board cells (long) in {0, ..., 9}, 0 = empty
    - labels: 81 cells (long), solution digit - 1 in {0, ..., 8}; given cells = -100
    - metric: board accuracy (exact match of the whole board) and cell accuracy (empty cells only), in %
    """

    SPLITS = ("train", "test_id")
    BEST_METRIC = ("board_acc", "max")
    PLOT_METRIC = ({"board_acc": "Board", "cell_acc": "Cell"}, "Accuracy", "Acc (%)", (0, 100))
    input_len = 81
    output_len = 81

    def __init__(self, split):
        super().__init__(split)
        self.download()
        with open(CSV_DIR / SPLIT_FILE[split], newline="") as f:
            rows = list(csv.reader(f))
        indices = np.random.RandomState(DATA_SEED).permutation(len(rows))[:SPLIT_SIZE[split]]
        boards = np.array([[int(c) for c in rows[i][0]] for i in indices], dtype=np.int64)
        solutions = np.array([[int(c) for c in rows[i][1]] for i in indices], dtype=np.int64)
        self.inputs = torch.from_numpy(boards)
        self.labels = torch.from_numpy(solutions) - 1
        self.labels[self.inputs != 0] = IGNORE

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, index):
        return {"inputs": self.inputs[index], "labels": self.labels[index]}

    def metric(self, pred, labels):
        """Board and cell accuracy, as in Hyper-SET (sudoku/utils.py, eval_vis_func).

        - pred: class indices [B, 81] in {0, ..., 8}, or logits [B, 81, 9]
        - labels: [B, 81], given cells = -100 (counted as correct)
        - board_acc: % of boards with every empty cell correct (exact match)
        - cell_acc: % of empty cells that are correct
        """
        if pred.dim() == 3:
            pred = pred.argmax(dim=-1)
        blank = labels != IGNORE
        correct = (pred == labels) | ~blank
        return {
            "board_acc": 100.0 * correct.all(dim=1).float().mean().item(),
            "cell_acc": 100.0 * correct[blank].float().mean().item(),
        }

    def sample_scores(self, pred, labels):
        """Per board: board_ok (every empty cell right), cells_ok, cells_empty; metric_from_scores gives metric()."""
        if pred.dim() == 3:
            pred = pred.argmax(dim=-1)
        blank = labels != IGNORE
        correct = (pred == labels) & blank
        return {"board_ok": (correct.sum(1) == blank.sum(1)).float(), "cells_ok": correct.sum(1).float(),
                "cells_empty": blank.sum(1).float()}

    def metric_from_scores(self, scores):
        """Scores summed or concatenated over the split -> board_acc, cell_acc (as metric)."""
        return {"board_acc": 100.0 * float(scores["board_ok"].sum() / scores["board_ok"].numel()),
                "cell_acc": 100.0 * float(scores["cells_ok"].sum() / scores["cells_empty"].sum())}

    def format_sample(self, inputs, pred, labels):
        """A board as text for the W&B sample table (Phase C.5 Task 4.3).

        - inputs (81,) digits, 0 = empty; pred (81,) in {0..8}; labels (81,) in {0..8}, given cells = -100
        - returns {"input", "prediction", "answer"} (9x9 grids of digits, '.' = empty in the input; the prediction
          keeps the given cells) and "correct" = "k/m" correct empty cells
        """
        empty = inputs == 0
        answer = torch.where(empty, labels + 1, inputs)
        filled = torch.where(empty, pred + 1, inputs)
        n_ok = int((pred[empty] == labels[empty]).sum())
        return {"input": grid(inputs), "prediction": grid(filled), "answer": grid(answer),
                "correct": f"{n_ok}/{int(empty.sum())}"}

    @classmethod
    def download(cls, log=print):
        """Download and unzip sudoku-hard.zip into data/sudoku/raw/.

        - Skipped if all csv files already exist.
        - Removes the zip, the __MACOSX folder and .DS_Store files afterwards.
        """
        if all((CSV_DIR / f).exists() for f in CSV_FILES):
            return
        RAW_DIR.mkdir(parents=True, exist_ok=True)
        zip_path = RAW_DIR / "sudoku-hard.zip"
        log(f"downloading {URL} -> {zip_path.relative_to(REPO_ROOT)}")
        urllib.request.urlretrieve(URL, zip_path)
        with zipfile.ZipFile(zip_path) as z:
            z.extractall(RAW_DIR)
        zip_path.unlink()
        shutil.rmtree(RAW_DIR / "__MACOSX", ignore_errors=True)
        for junk in RAW_DIR.rglob(".DS_Store"):
            junk.unlink()
        log(f"extracted to {CSV_DIR.relative_to(REPO_ROOT)}")


def grid(cells):
    """(81,) digits -> 9 lines of a 9x9 board, 0 shown as '.', boxes separated."""
    rows = []
    for r in range(9):
        if r in (3, 6):
            rows.append("------+-------+------")
        d = ["." if int(c) == 0 else str(int(c)) for c in cells[9 * r:9 * r + 9]]
        rows.append(" ".join(d[0:3]) + " | " + " ".join(d[3:6]) + " | " + " ".join(d[6:9]))
    return "\n".join(rows)
