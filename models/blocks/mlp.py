"""MLP branch (family 3b): closed-form gradient of the MLP potential.

Theory: models/theory/mlp.md
"""
import torch
import torch.nn as nn


def silu_prime(x):
    """Closed-form derivative of SiLU.

    - SiLU(x) = x * sigmoid(x)
    - SiLU'(x) = sigmoid(x) * (1 + x * (1 - sigmoid(x)))
    """
    s = torch.sigmoid(x)
    return s * (1 + x * (1 - s))


class MLPBlock(nn.Module):
    """MLP branch, family 3b (mirror by weight sharing).

    - W_up:   nn.Linear(d, ratio * d), lifts d -> ratio * d
    - W_down: nn.Linear(ratio * d, d), lowers ratio * d -> d inside the potential
    - no bias; ratio = 4 gives 8 d^2 parameters; weights are initialised by the core latent (LatentEnergyLayer)
    - row convention as in the theory: h W_up = W_up(h), and W_up^T acting on a row = (row) @ W_up.weight
    - forward returns the closed-form gradient of the potential (one row = one token):
        grad = [ SiLU'(h W_up) * ( sigmoid(SiLU(h W_up) W_down) W_down^T ) ] W_up^T
        SiLU'(x) = sigmoid(x) * (1 + x * (1 - sigmoid(x)))
    - the coefficient alpha is not here (set in models/modeling_base.py)
    """

    def __init__(self, d_model, ratio=4):
        super().__init__()
        self.W_up = nn.Linear(d_model, ratio * d_model, bias=False)
        self.W_down = nn.Linear(ratio * d_model, d_model, bias=False)

    def forward(self, hidden_states):
        """Closed-form gradient of the MLP potential.

        - hidden_states: (B, N, d), already z + P_Z
        - returns: (B, N, d), the gradient for each token
        """
        a = self.W_up(hidden_states)                                       # h W_up               (B, N, 4d)
        q = self.W_down(nn.functional.silu(a))                             # SiLU(a) W_down        (B, N, d)
        g = (torch.sigmoid(q) @ self.W_down.weight) * silu_prime(a)        # (sigmoid(q) W_down^T) * SiLU'(a)
        return g @ self.W_up.weight                                        # (.) W_up^T            (B, N, d)


def energy(block, hidden_states):
    """MLP potential, one value per sample (to be moved into the core latent's get_energy).

    - phi_mlp(h) = sum over tokens and over k of softplus( [SiLU(h W_up) W_down]_k )
    - hidden_states: (B, N, d)
    - returns: (B,)
    """
    q = block.W_down(nn.functional.silu(block.W_up(hidden_states)))
    return nn.functional.softplus(q).sum(dim=(-2, -1))
