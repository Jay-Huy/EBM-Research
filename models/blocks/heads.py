"""Head helpers shared by the self- and cross-attention blocks (Phase C.4, Task 6).

Weights stay four nn.Linear(d, d); head h is a slice of d_h = d / n_heads columns of the math matrices:
  - W_q^h, W_k^h, W_v^h = columns h*d_h:(h+1)*d_h of W_q, W_k, W_v    (d, d_h)
  - W_o^h = rows h*d_h:(h+1)*d_h of W_o                                  (d_h, d)
  nn.Linear stores the transpose, so W_q^h = W_q.weight[sl].T and W_o^h = W_o.weight[:, sl].T
Theory: models/theory/cross_attn.md §6, et_self_attn.md §7, forced_self_attn.md §7.
"""
import torch


def split_heads(x, n_heads):
    """(B, L, d) -> (B, n_heads, L, d_h)."""
    B, L, d = x.shape
    return x.view(B, L, n_heads, d // n_heads).transpose(1, 2)


def merge_heads(x):
    """(B, n_heads, L, d_h) -> (B, L, d): concatenation of the heads."""
    B, H, L, d_h = x.shape
    return x.transpose(1, 2).reshape(B, L, H * d_h)


@torch.no_grad()
def init_tie_heads(block):
    """Put W_v, W_o on the tie manifold of every head, at a random gauge point.

    - per head: W_v^h = W_k^h G_h, W_o^h = G_h^T W_q^h^T, G_h a random orthogonal (d_h, d_h) matrix (QR)
      => W_v^h W_o^h = W_k^h W_q^h^T exactly
    - nn.Linear stores the transpose: W_v.weight[sl] = G_h^T W_k.weight[sl], W_o.weight[:, sl] = W_q.weight[sl]^T G_h
    - n_heads = 1: the same as the one-head init_tie (one G of size (d, d))
    """
    d_h = block.d_head
    w = block.W_q.weight
    for h in range(block.n_heads):
        sl = slice(h * d_h, (h + 1) * d_h)
        G, _ = torch.linalg.qr(torch.randn(d_h, d_h, dtype=w.dtype, device=w.device))
        block.W_v.weight[sl].copy_(G.T @ block.W_k.weight[sl])
        block.W_o.weight[:, sl].copy_(block.W_q.weight[sl].T @ G)


def tie_penalty_heads(block, eps=1e-12):
    """Relative tie error, mean over heads (as tie_penalty in docs/samples/hopfield_models.py).

    - R = mean_h ||W_v^h W_o^h - W_k^h W_q^h^T||_F^2 / ||W_k^h W_q^h^T||_F^2
    - each head must match on its own: the sum over heads multiplies different A_h, so a global
      W_v W_o = W_k W_q^T is not enough
    - 0 when the tie is exact; ~2 at a random init; 0 for tie_mode 'hard' (no W_v, W_o)
    - how it enters the training loss is decided in Phase C.5
    """
    if block.tie_mode == "hard":
        return torch.zeros((), dtype=block.W_q.weight.dtype, device=block.W_q.weight.device)
    d_h = block.d_head
    Wq, Wk, Wv, Wo = block.W_q.weight, block.W_k.weight, block.W_v.weight, block.W_o.weight
    terms = []
    for h in range(block.n_heads):
        sl = slice(h * d_h, (h + 1) * d_h)
        got = Wv[sl].T @ Wo[:, sl].T                                       # W_v^h W_o^h        (d, d)
        want = Wk[sl].T @ Wq[sl]                                           # W_k^h W_q^h^T      (d, d)
        terms.append((got - want).pow(2).sum() / (want.pow(2).sum() + eps))
    return torch.stack(terms).mean()
