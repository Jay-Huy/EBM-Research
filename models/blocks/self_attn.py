"""Self-attention branch: closed-form gradient of the self potential, ET and Forced.

Theory: models/theory/et_self_attn.md, models/theory/forced_self_attn.md
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.blocks.heads import init_tie_heads, merge_heads, split_heads, tie_penalty_heads


class ETSelfAttnBlock(nn.Module):
    """Self-attention branch, ET version: LSE per row, A_h = W_q^h W_k^h^T free; n_heads heads, d_h = d / n_heads.

    - W_q, W_k, W_v, W_o: nn.Linear(d, d, bias=False); head h = a slice of d_h columns (models/blocks/heads.py)
    - the potential and the gradient below are per head, summed over heads (et_self_attn.md §7)
    - row convention as in the theory: Q = Z_hat W_q = W_q(Z_hat), K = Z_hat W_k = W_k(Z_hat)
    - potential (Z_hat: N tokens):
        phi_self(Z_hat) = 1/beta * sum_n log sum_m exp( beta * z_n W_q W_k^T z_m^T )
    - forward returns the closed-form gradient of the potential, two terms (the two slots of Z_hat in S = Z_hat A Z_hat^T):
        P = softmax_row( beta * Q K^T )                                 -> attn_weights
        (1) query slot:  P   Z_hat W_k W_q^T  ~  P   (Z_hat W_v)   W_o     -> attn_weights   @ V_query, then W_o
        (2) key slot:    P^T Z_hat W_q W_k^T  ~  P^T (Z_hat W_o^T) W_v^T   -> attn_weights^T @ V_key,   then W_v^T
      term (2) reuses W_v, W_o transposed, since W_q W_k^T = (W_k W_q^T)^T ~ (W_v W_o)^T = W_o^T W_v^T
    - nn.Linear W applies x W; the transposed map x W^T is F.linear(x, W.weight.T)
    - tie_mode:
        - 'orbit' (default): tie W_v W_o = W_k W_q^T exact at init (init_tie), soft (MSE) during training;
          the theory assumes it exact
        - 'hard' (ablation): no W_v, W_o; value of (1) is K W_q^T, of (2) is Q W_k^T, the exact gradient
    - input already carries the positional encoding: Z_hat = Z + P_Z (added in the core latent)
    - beta = 1 / sqrt(d_h) by default; self enters E with a + sign, set in models/modeling_base.py
    - sums over heads use concatenation: concat_h(Y_h) M = sum_h Y_h M^h, with M^h the row block h of M
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

        - W_v^h W_o^h = W_k^h W_q^h^T (term (1)) and W_o^h^T W_v^h^T = W_q^h W_k^h^T (term (2))
        - called by the core latent after its weight init (otherwise the weight init would overwrite W_v, W_o)
        """
        if self.tie_mode != "hard":
            init_tie_heads(self)

    def tie_penalty(self):
        """Relative tie error, mean over heads (heads.tie_penalty_heads); 0 if the tie is exact."""
        return tie_penalty_heads(self)

    def forward(self, hidden_states):
        """Closed-form gradient of the ET self potential.

        - hidden_states: (B, N, d), already Z + P_Z
        - returns: (B, N, d) = sum_h [ (1) query slot + (2) key slot ]
        """
        Q = split_heads(self.W_q(hidden_states), self.n_heads)                        # (B, H, N, d_h)
        K = split_heads(self.W_k(hidden_states), self.n_heads)                        # (B, H, N, d_h)
        attn_weights = torch.softmax(self.beta * (Q @ K.transpose(-2, -1)), dim=-1)   # P_h, rows sum to 1 (B, H, N, N)
        attn_t = attn_weights.transpose(-2, -1)                                       # P_h^T

        if self.tie_mode == "hard":
            # exact gradient: (1) sum_h P_h K_h W_q^h^T, (2) sum_h P_h^T Q_h W_k^h^T
            query_slot = F.linear(merge_heads(attn_weights @ K), self.W_q.weight.T)
            key_slot = F.linear(merge_heads(attn_t @ Q), self.W_k.weight.T)
            return query_slot + key_slot

        # (1) query slot: sum_h P_h (Z_hat W_v^h) W_o^h
        V_query = split_heads(self.W_v(hidden_states), self.n_heads)                  # Z_hat W_v^h    (B, H, N, d_h)
        query_slot = self.W_o(merge_heads(attn_weights @ V_query))

        # (2) key slot: sum_h P_h^T (Z_hat W_o^h^T) W_v^h^T
        V_key = split_heads(F.linear(hidden_states, self.W_o.weight.T), self.n_heads)  # Z_hat W_o^h^T (B, H, N, d_h)
        key_slot = F.linear(merge_heads(attn_t @ V_key), self.W_v.weight.T)
        return query_slot + key_slot


def et_energy(block, hidden_states):
    """ET self potential, one value per sample (to be moved into the core latent's get_energy).

    - phi_self = sum_h 1/beta * sum_n log sum_m exp( beta * Q_n^h K_m^h^T ), Q = Z_hat W_q, K = Z_hat W_k
    - uses W_q W_k^T (the theory), not W_v, W_o
    - hidden_states: (B, N, d)
    - returns: (B,)
    """
    Q = split_heads(block.W_q(hidden_states), block.n_heads)               # (B, H, N, d_h)
    K = split_heads(block.W_k(hidden_states), block.n_heads)               # (B, H, N, d_h)
    attn_scores = Q @ K.transpose(-2, -1)                                  # S_h (B, H, N, N)
    return (torch.logsumexp(block.beta * attn_scores, dim=-1) / block.beta).sum(dim=(-2, -1))


class ForcedSelfAttnBlock(nn.Module):
    """Self-attention branch, Forced version: global LSE, every A_h = W_q^h W_k^h^T symmetric; n_heads heads.

    - W_q, W_k, W_v, W_o: nn.Linear(d, d, bias=False); head h = a slice of d_h columns (models/blocks/heads.py)
    - the potential and the gradient below are per head, summed over heads (forced_self_attn.md §7)
    - A_h symmetric <=> W_k^h = W_q^h S_h with S_h = S_h^T; enforced per head by retract() after each optimizer step
      (models/theory/retraction.md), assumed exact here
    - row convention as in the theory: Q = Z_hat W_q = W_q(Z_hat), K = Z_hat W_k = W_k(Z_hat)
    - potential (Z_hat: N tokens), the factor N compensates the 1/N scale of the global softmax:
        phi_self(Z_hat) = N/beta * log sum_{n,m} exp( beta * z_n W_q W_k^T z_m^T )
    - forward returns the closed-form gradient of the potential; with A and P symmetric the two slots merge:
        P = softmax_global( beta * Q K^T )      (the whole N x N matrix sums to 1)  -> attn_weights
        2N P Z_hat W_k W_q^T  ~  2N P (Z_hat W_v) W_o                               -> 2N * (attn_weights @ V_query), then W_o
    - tie_mode:
        - 'orbit' (default): tie W_v W_o = W_k W_q^T exact at init (init_tie), soft (MSE) during training;
          the theory assumes it exact
        - 'hard' (ablation): no W_v, W_o; value = K W_q^T, the exact gradient
    - input already carries the positional encoding: Z_hat = Z + P_Z (added in the core latent)
    - beta = 1 / sqrt(d_h) by default; self enters E with a + sign, set in models/modeling_base.py
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
        self.register_buffer("retract_count", torch.zeros((), dtype=torch.long))   # number of retractions done
        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        if tie_mode != "hard":
            self.W_v = nn.Linear(d_model, d_model, bias=False)
            self.W_o = nn.Linear(d_model, d_model, bias=False)

    def init_tie(self):
        """Put W_v, W_o on the tie manifold of every head (tie_mode 'orbit'), see heads.init_tie_heads.

        - called by the core latent after its weight init and after every A_h is made symmetric
        """
        if self.tie_mode != "hard":
            init_tie_heads(self)

    def tie_penalty(self):
        """Relative tie error, mean over heads (heads.tie_penalty_heads); 0 if the tie is exact."""
        return tie_penalty_heads(self)

    @torch.no_grad()
    def retract(self):
        """Pull W_q^h, W_k^h back so that every A_h = W_q^h W_k^h^T is symmetric (retraction.md §0, §2).

        - each head on its own, all heads at the same call
        - 1st, 3rd, 5th, ... call: keep W_q^h, fix W_k^h <- W_q^h S_h
        - 2nd, 4th, 6th, ... call: keep W_k^h, fix W_q^h <- W_k^h S_h'
        - S_h, S_h' symmetric (d_h, d_h), from symmetric_factor (retraction.md §3)
        - W_v, W_o are not touched (the soft tie follows during training)
        - nn.Linear stores the transpose: (W_q^h S)^T = S W_q^h^T, so W_k.weight[sl] = S @ W_q.weight[sl]
        - called after each optimizer step (Phase C.5) and once at init (core latent)
        """
        d_h = self.d_head
        for h in range(self.n_heads):
            sl = slice(h * d_h, (h + 1) * d_h)
            W_q, W_k = self.W_q.weight[sl].T, self.W_k.weight[sl].T        # math matrices of head h (d, d_h)
            if self.retract_count % 2 == 0:                                # keep W_q^h, fix W_k^h
                S = symmetric_factor(W_q, W_k)
                self.W_k.weight[sl].copy_(S @ self.W_q.weight[sl])
            else:                                                          # keep W_k^h, fix W_q^h
                S = symmetric_factor(W_k, W_q)
                self.W_q.weight[sl].copy_(S @ self.W_k.weight[sl])
        self.retract_count += 1

    def forward(self, hidden_states):
        """Closed-form gradient of the Forced self potential.

        - hidden_states: (B, N, d), already Z + P_Z
        - returns: (B, N, d) = sum_h 2N P_h Z_hat W_v^h W_o^h (orbit) or sum_h 2N P_h K_h W_q^h^T (hard)
        """
        B, N, _ = hidden_states.shape
        H = self.n_heads
        Q = split_heads(self.W_q(hidden_states), H)                        # (B, H, N, d_h)
        K = split_heads(self.W_k(hidden_states), H)                        # (B, H, N, d_h)
        attn_scores = self.beta * (Q @ K.transpose(-2, -1))                # (B, H, N, N)
        attn_weights = torch.softmax(attn_scores.reshape(B, H, N * N), dim=-1).reshape(B, H, N, N)   # P_h, each sums to 1

        if self.tie_mode == "hard":
            # exact gradient: sum_h 2N P_h (K_h W_q^h^T)
            return 2 * N * F.linear(merge_heads(attn_weights @ K), self.W_q.weight.T)

        # sum_h 2N P_h (Z_hat W_v^h) W_o^h
        V_query = split_heads(self.W_v(hidden_states), H)                  # Z_hat W_v^h (B, H, N, d_h)
        return 2 * N * self.W_o(merge_heads(attn_weights @ V_query))


def symmetric_factor(base, target):
    """Symmetric S closest in the sense min_{S = S^T} ||base S - target||_F (retraction.md §3).

    - base, target: (d, d_h) math matrices of one head; base of full column rank (d_h = d: invertible)
    - optimality condition, a Lyapunov equation:
        G = base^T base,  C = base^T target + target^T base
        G S + S G = C
    - solve with G = U diag(lam) U^T:
        S~_ij = C~_ij / (lam_i + lam_j),  C~ = U^T C U,  S = U S~ U^T
    - computed in float64, returned in the dtype of base: G = base^T base squares the condition number of base
      (smallest eigenvalue of G ~1e-9 at d_h = 512, float32 eps ~1e-7), so in float32 eigh gives eigenvalues of
      the wrong sign, 1 / (lam_i + lam_j) explodes, W_q, W_k grow at each retraction and training diverges
      (Phase C.5 Task 5.2: Forced, CE 5e4 at step ~100)
    - returns: (d_h, d_h), symmetric
    """
    dtype = base.dtype
    base, target = base.double(), target.double()
    G = base.T @ base
    C = base.T @ target + target.T @ base
    lam, U = torch.linalg.eigh(G)
    S_tilde = (U.T @ C @ U) / (lam[:, None] + lam[None, :])
    S = U @ S_tilde @ U.T
    return (0.5 * (S + S.T)).to(dtype)                                     # exact symmetry against round-off


def forced_energy(block, hidden_states):
    """Forced self potential, one value per sample (to be moved into the core latent's get_energy).

    - phi_self = sum_h N/beta * log sum_{n,m} exp( beta * Q_n^h K_m^h^T ), Q = Z_hat W_q, K = Z_hat W_k
    - uses W_q W_k^T (the theory), not W_v, W_o
    - hidden_states: (B, N, d)
    - returns: (B,)
    """
    B, N, _ = hidden_states.shape
    H = block.n_heads
    Q = split_heads(block.W_q(hidden_states), H)                           # (B, H, N, d_h)
    K = split_heads(block.W_k(hidden_states), H)                           # (B, H, N, d_h)
    attn_scores = Q @ K.transpose(-2, -1)                                  # S_h (B, H, N, N)
    return (N / block.beta * torch.logsumexp(block.beta * attn_scores.reshape(B, H, N * N), dim=-1)).sum(dim=-1)
