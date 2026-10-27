"""Positional encoding: one learned vector per position (as in BERT).

Notes:
    - P_Z (state) and P_X (context) are both created and added in the core latent (models/modeling_base.py).
    - P_X is added once, before the loop (X is fixed).
    - P_Z is added at every step: the self, cross and mlp blocks receive Z_hat = Z + P_Z; the term 1/2 ||Z||^2 keeps Z.
"""
import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    """Learned positional encoding.

    - weight: (n_positions, d_model), init trunc_normal (std 1.0, truncated at +-3 std)
    - forward returns the table, (n_positions, d_model); it is added to a (B, n_positions, d_model) tensor by broadcast
    """

    def __init__(self, n_positions, d_model):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(n_positions, d_model))
        self.init_weights()

    def init_weights(self):
        """trunc_normal init, std 1.0, truncated at +-3 std (Phase C.5 Task 9.6).

        - the same scale as the token embedding (std 1, models/tasks/sudoku.py) and Z0 (N(0, 1)): position and
          content of X_tilde = X + P_X are 1 : 1, as in BERT (where both are std 0.02); std 0.02 here with an
          embedding of std 1 made the position 50 times smaller than the content
        """
        nn.init.trunc_normal_(self.weight, std=1.0, a=-3.0, b=3.0)

    def forward(self):
        """Return the positional table, (n_positions, d_model)."""
        return self.weight
