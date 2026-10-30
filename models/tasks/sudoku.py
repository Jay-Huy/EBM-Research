"""Sudoku-RRN (data/sudoku/sudoku_dataset.py), d_model = 512.

- one token per cell: inputs (B, 81) long in {0, ..., 9} (0 = empty) -> X (B, 81, d)
- Z* (B, 81, d) -> logits (B, 81, 9); labels = digit - 1 in {0, ..., 8}, given cells = -100
- loss: cross-entropy, ignore_index -100 (only empty cells count), as Hyper-SET
"""
import math

import torch.nn as nn
import torch.nn.functional as F

from models.modeling_base import EnergyLoopModel, register_model

N_DIGITS = 9
IGNORE = -100


class SudokuModel(EnergyLoopModel):
    """Encoder nn.Embedding(10, d) (0 = empty has its own vector), decoder nn.Linear(d, 9) no bias, loss CE."""

    def build_encoder(self):
        return nn.Embedding(N_DIGITS + 1, self.config.d_model)

    def build_decoder(self):
        return nn.Linear(self.config.d_model, N_DIGITS, bias=False)

    def init_encoder_decoder(self, c=0.25):
        """- embedding ~ N(0, 1) (PyTorch default), the scale of Z0 ~ N(0, 1)
        - decoder: trunc_normal, std = c / sqrt(d_model), truncated at +-3 std, as the core latent
        """
        nn.init.normal_(self.encoder.weight, std=1.0)
        std = c / math.sqrt(self.config.d_model)
        nn.init.trunc_normal_(self.decoder.weight, mean=0.0, std=std, a=-3 * std, b=3 * std)

    def compute_loss(self, outputs, labels):
        return F.cross_entropy(outputs.reshape(-1, N_DIGITS), labels.reshape(-1), ignore_index=IGNORE)

    def predict(self, outputs):
        """Digit - 1 of every cell, (B, 81), the format of SudokuRRN.metric."""
        return outputs.argmax(dim=-1)


register_model("sudoku_rrn", d_model=768, n_tokens=81, n_context=81)(SudokuModel)
