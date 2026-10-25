"""Cross-attention branch: closed-form gradient of the cross potential.

Theory: models/theory/cross_attn.md
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.blocks.heads import init_tie_heads, merge_heads, split_heads, tie_penalty_heads


class CrossAttnBlock(nn.Module):
    """Cross-attention branch (shared by ET and Forced), n_heads heads of size d_h = d / n_heads.

    - W_q, W_k, W_v, W_o: nn.Linear(d, d, bias=False); head h = a slice of d_h columns (models/blocks/heads.py)
    - row convention as in the theory: Q = h W_q = W_q(h), K = X W_k = W_k(X)
    - potential (one token h, context X), a sum over heads:
        phi_cross(h) = sum_h 1/beta * log sum_m exp( beta * h W_q^h W_k^h^T x_m^T )
    - forward returns the closed-form gradient of the potential (cross_attn.md §6):
        grad = sum_h softmax( beta * Q_h K_h^T ) X W_k^h W_q^h^T
    - tie_mode:
        - 'orbit' (default): value path X W_v^h W_o^h; tie W_v^h W_o^h = W_k^h W_q^h^T per head, exact at init
          (init_tie), soft during training (tie_penalty); the theory assumes it exact
        - 'hard' (ablation): no W_v, W_o; value = K_h W_q^h^T, the exact gradient
    - inputs already carry the positional encoding: h = z + P_Z, X = X + P_X (added in the core latent)
    - beta = 1 / sqrt(d_h) by default; the coefficient gamma_2 is not here (set in models/modeling_base.py)
    """

    def __init__(self, d_model, n_heads=1, beta=None, tie_mode="orbit"):
        super().__init__()
        assert tie_mode in ("orbit", "hard"), f"tie_mode is {tie_mode}"
        assert d_model % n_heads == 0, f"d_model {d_model} is not divisible by n_heads {n_heads}"
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.beta = 1.0 / math.sqrt(self.d_head) if beta is None else beta
        self.tie_mode = tie_mode
        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        if tie_mode != "hard":
            self.W_v = nn.Linear(d_model, d_model, bias=False)
            self.W_o = nn.Linear(d_model, d_model, bias=False)

    def init_tie(self):
        """Put W_v, W_o on the tie manifold of every head (tie_mode 'orbit'), see heads.init_tie_heads.

        - called by the core latent after its weight init (otherwise the weight init would overwrite W_v, W_o)
        """
        if self.tie_mode != "hard":
            init_tie_heads(self)

    def tie_penalty(self):
        """Relative tie error, mean over heads (heads.tie_penalty_heads); 0 if the tie is exact."""
        return tie_penalty_heads(self)

    def forward(self, hidden_states, context):
        """Closed-form gradient of the cross potential.

        - hidden_states: (B, N, d), already z + P_Z
        - context: (B, N_x, d), already X + P_X
        - returns: (B, N, d) = sum_h P_h V_h, with V_h = X W_v^h W_o^h (orbit) or K_h W_q^h^T (hard)
        """
        Q = split_heads(self.W_q(hidden_states), self.n_heads)                 # (B, H, N, d_h)
        K = split_heads(self.W_k(context), self.n_heads)                       # (B, H, N_x, d_h)
        attn_weights = torch.softmax(self.beta * (Q @ K.transpose(-2, -1)), dim=-1)   # (B, H, N, N_x)

        if self.tie_mode == "hard":
            # exact gradient: concat_h(P_h K_h) W_q^T = sum_h P_h K_h W_q^h^T (nn.Linear weight of W_q is W_q^T)
            return F.linear(merge_heads(attn_weights @ K), self.W_q.weight.T)
        V = split_heads(self.W_v(context), self.n_heads)                       # X W_v^h  (B, H, N_x, d_h)
        return self.W_o(merge_heads(attn_weights @ V))                         # sum_h P_h X W_v^h W_o^h  (B, N, d)


def energy(block, hidden_states, context):
    """Cross potential, one value per sample.

    - phi_cross = sum_h sum_n 1/beta * log sum_m exp( beta * Q_n^h K_m^h^T ), Q = h W_q, K = X W_k
    - uses W_q W_k^T (the theory), not W_v, W_o
    - hidden_states: (B, N, d); context: (B, N_x, d)
    - returns: (B,)
    """
    Q = split_heads(block.W_q(hidden_states), block.n_heads)                   # (B, H, N, d_h)
    K = split_heads(block.W_k(context), block.n_heads)                         # (B, H, N_x, d_h)
    attn_scores = Q @ K.transpose(-2, -1)                                      # (B, H, N, N_x)
    return (torch.logsumexp(block.beta * attn_scores, dim=-1) / block.beta).sum(dim=(-2, -1))
