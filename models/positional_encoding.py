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

    - weight: (n_positions, d_model), init trunc_normal (std 0.02)
    - forward returns the table, (n_positions, d_model); it is added to a (B, n_positions, d_model) tensor by broadcast
    """

    def __init__(self, n_positions, d_model):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(n_positions, d_model))
        self.init_weights()

    def init_weights(self):
        """trunc_normal init, std 0.02 (BERT / ViT)."""
        nn.init.trunc_normal_(self.weight, std=0.02)

    def forward(self):
        """Return the positional table, (n_positions, d_model)."""
        return self.weight
