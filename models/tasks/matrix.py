"""Matrix toy tasks (inverse, lowrank, addition; data/algorithm/algorithmic_dataset.py), d_model = 256.

- each matrix entry is one token: inputs (B, N_x) float -> X (B, N_x, d); Z* (B, N, d) -> outputs (B, N)
- N = 400 (a 20 x 20 output); N_x = 400 (inverse, lowrank) or 800 (addition: A and B concatenated)
- loss: MSE, the metric of the datasets
"""
import math

import torch
import torch.nn as nn

from models.modeling_base import EnergyLoopModel, register_model


class ScalarEmbedding(nn.Module):
    """One float per token -> d_model: (B, L) -> (B, L, d), x_l * w (nn.Linear(1, d), no bias)."""

    def __init__(self, d_model):
        super().__init__()
        self.proj = nn.Linear(1, d_model, bias=False)

    def forward(self, x):
        return self.proj(x.unsqueeze(-1))


class ScalarHead(nn.Module):
    """d_model -> one float per token: (B, N, d) -> (B, N), nn.Linear(d, 1), no bias."""

    def __init__(self, d_model):
        super().__init__()
        self.proj = nn.Linear(d_model, 1, bias=False)

    def forward(self, z):
        return self.proj(z).squeeze(-1)


class MatrixModel(EnergyLoopModel):
    """Encoder ScalarEmbedding, decoder ScalarHead, loss MSE."""

    def build_encoder(self):
        return ScalarEmbedding(self.config.d_model)

    def build_decoder(self):
        return ScalarHead(self.config.d_model)

    def init_encoder_decoder(self, c=0.25):
        """- encoder weight ~ N(0, 1): a token x_l w has the scale of x_l, like Z0 ~ N(0, 1)
        - decoder: trunc_normal, std = c / sqrt(d_model), truncated at +-3 std, as the core latent
        """
        nn.init.normal_(self.encoder.proj.weight, std=1.0)
        std = c / math.sqrt(self.config.d_model)
        nn.init.trunc_normal_(self.decoder.proj.weight, mean=0.0, std=std, a=-3 * std, b=3 * std)

    def compute_loss(self, outputs, labels):
        return torch.mean((outputs - labels) ** 2)

    def predict(self, outputs):
        return outputs


register_model("inverse", d_model=256, n_tokens=400, n_context=400)(MatrixModel)
register_model("lowrank", d_model=256, n_tokens=400, n_context=400)(MatrixModel)
register_model("addition", d_model=256, n_tokens=400, n_context=800)(MatrixModel)
