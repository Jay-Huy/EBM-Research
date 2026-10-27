"""RMSNorm without learnable weight, and its vector-Jacobian product (Phase C.5 v2, pre-norm energy).

- per token (last dim): g = sqrt(d) x / s,  s = sqrt(||x||^2 + d eps)   (= x / sqrt(mean(x^2) + eps))
- Jacobian M = dg/dx = sqrt(d) / s (I - x x^T / s^2), symmetric, so M^T v = M v
- written here (not torch's RMSNorm) so that the closed-form step (rms_norm_vjp) is the exact gradient of the
  energy that get_energy computes with the same rms_norm, eps included
- trap: the factor sqrt(d) / s; a token near 0 amplifies the branches (plans/Phase-C.5.md, Task 9.6)
"""
import math

EPS = 1e-6


def rms_norm(x):
    """(..., d) -> (..., d): each token on the sphere of radius sqrt(d) (up to eps)."""
    d = x.shape[-1]
    s = (x.pow(2).sum(-1, keepdim=True) + d * EPS).sqrt()
    return math.sqrt(d) * x / s


def rms_norm_vjp(x, v):
    """M^T v with M = d rms_norm(x) / dx, per token: sqrt(d) / s * (v - x (x . v) / s^2); x, v: (..., d)."""
    d = x.shape[-1]
    s2 = x.pow(2).sum(-1, keepdim=True) + d * EPS
    return math.sqrt(d) / s2.sqrt() * (v - x * (x * v).sum(-1, keepdim=True) / s2)
