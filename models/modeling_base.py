"""Core latent (LatentEnergyLayer) and the base model (encoder -> core latent -> decoder, Task 4).

- LatentEnergyLayer: one energy layer; one update step = one gradient step on E
- EnergyLoopConfig: hyper-parameters of a whole model (saved with the checkpoint)
- EnergyLoopModel: base model; subclasses (models/tasks/) give the encoder, the decoder and the loss
- register_model / build_model / list_models: registry, as get_dataset in data/

Theory: models/theory/layer.md, models/theory/retraction.md, models/theory/ift.md
"""
import math
from dataclasses import asdict, dataclass, replace

import torch
import torch.nn as nn

from models.blocks.cross_attn import CrossAttnBlock
from models.blocks.cross_attn import energy as cross_energy
from models.blocks.mlp import MLPBlock
from models.blocks.mlp import energy as mlp_energy
from models.blocks.self_attn import ETSelfAttnBlock, ForcedSelfAttnBlock, et_energy, forced_energy
from models.positional_encoding import PositionalEncoding


class LatentEnergyLayer(nn.Module):
    """Core latent: one energy layer, one update step = one gradient-descent step on E.

    - energy (Z_hat = Z + P_Z at every step, X_tilde = X + P_X once):
        E(Z; X) = rho/2 ||Z||^2 + phi_self(Z_hat) - gamma2 phi_cross(Z_hat; X_tilde) - alpha phi_mlp(Z_hat)
      self enters with +: minimising E pushes similar tokens apart (anti-collapse); only rho/2 ||Z||^2 keeps the plain Z
    - one step Z <- Z - step_size * grad E:
        Z_next = (1 - step_size * rho) Z + step_size * [ - self(Z_hat) + gamma2 cross(Z_hat, X_tilde) + alpha mlp(Z_hat) ]
      each branch forward is the closed-form gradient of its potential
    - no projection: E is bounded below and has a minimum when ||W_q|| ||W_k|| < 1/2 (summed over heads),
      which the trunc_normal init gives (init_weights)
    - mode: 'et' (ETSelfAttnBlock) or 'forced' (ForcedSelfAttnBlock); cross and mlp are the same in both
    - tie_mode: 'orbit' (default) or 'hard' (ablation), passed to the self and cross blocks
    - n_heads: heads of the self and cross blocks, d_h = d_model / n_heads, beta = 1 / sqrt(d_h) (Task 6);
      self and cross potentials are sums over heads, gamma2 shared by all heads
    - coefficients rho, gamma2, alpha and step_size are set here, not in the blocks
    """

    def __init__(self, d_model, n_tokens, n_context, mode="et", tie_mode="orbit",
                 rho=1.0, gamma2=1.0, alpha=1.0, step_size=0.5, mlp_ratio=4, beta=None, n_heads=1):
        super().__init__()
        assert mode in ("et", "forced"), f"mode is {mode}"
        self.mode = mode
        self.d_model = d_model
        self.n_heads = n_heads
        self.rho, self.gamma2, self.alpha = rho, gamma2, alpha
        self.step_size = step_size

        if mode == "et":
            self.self_attn = ETSelfAttnBlock(d_model, n_heads=n_heads, beta=beta, tie_mode=tie_mode)
        else:
            self.self_attn = ForcedSelfAttnBlock(d_model, n_heads=n_heads, beta=beta, tie_mode=tie_mode)
        self.cross_attn = CrossAttnBlock(d_model, n_heads=n_heads, beta=beta, tie_mode=tie_mode)
        self.mlp = MLPBlock(d_model, ratio=mlp_ratio)
        self.pe_z = PositionalEncoding(n_tokens, d_model)       # P_Z
        self.pe_x = PositionalEncoding(n_context, d_model)      # P_X
        self.init_weights()

    def init_weights(self, c=0.25):
        """Init the whole core latent, in this order.

        - trunc_normal for every nn.Linear: std = c / sqrt(d_model), truncated at +-3 std
          (c = 1/4: ||W_q|| ||W_k|| ~ 4 c^2 = 1/4 < 1/2 by RMT, the condition of Theorem 3.1)
        - trunc_normal (std 0.02) for P_Z, P_X
        - Forced: retract once, so every A_h = W_q^h W_k^h^T is symmetric (models/theory/retraction.md); no-op for ET
        - init_tie of the self and cross blocks, last (so the weight init does not overwrite W_v, W_o)
        """
        std = c / math.sqrt(self.d_model)
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.trunc_normal_(module.weight, mean=0.0, std=std, a=-3 * std, b=3 * std)
        self.pe_z.init_weights()
        self.pe_x.init_weights()
        self.retract()
        self.self_attn.init_tie()
        self.cross_attn.init_tie()

    @torch.no_grad()
    def retract(self):
        """Retract every Forced self-attention block (models/theory/retraction.md); no-op for ET.

        - training: optimizer.step(), then model.retract()
        """
        for module in self.modules():
            if isinstance(module, ForcedSelfAttnBlock):
                module.retract()

    def tie_penalty(self):
        """Relative tie error of the self and cross blocks, mean over heads (models/blocks/heads.py).

        - returns {"self": scalar, "cross": scalar}; 0 when W_v^h W_o^h = W_k^h W_q^h^T exactly
        - how it enters the training loss is decided in Phase C.5
        """
        return {"self": self.self_attn.tie_penalty(), "cross": self.cross_attn.tie_penalty()}

    def add_context_pe(self, context):
        """X_tilde = X + P_X, computed once before the loop (X is fixed).

        - context: (B, N_x, d) -> (B, N_x, d)
        """
        return context + self.pe_x()

    def forward_step(self, z, context_pe):
        """One update step Z <- Z - step_size * grad E.

        - z: (B, N, d), the state
        - context_pe: (B, N_x, d), X_tilde = X + P_X (from add_context_pe)
        - returns: (B, N, d)
        """
        z_hat = z + self.pe_z()                                            # Z_hat = Z + P_Z   (B, N, d)
        self_term = self.self_attn(z_hat)                                  # grad phi_self(Z_hat)
        cross_term = self.cross_attn(z_hat, context_pe)                    # grad phi_cross(Z_hat; X_tilde)
        mlp_term = self.mlp(z_hat)                                         # grad phi_mlp(Z_hat)
        branches = - self_term + self.gamma2 * cross_term + self.alpha * mlp_term          # self: + in E, so - here
        return (1 - self.step_size * self.rho) * z + self.step_size * branches

    def get_energy(self, z, context_pe):
        """Energy E(Z; X), one value per sample (models/theory/layer.md §2).

        - E = rho/2 ||Z||^2 + phi_self(Z_hat) - gamma2 phi_cross(Z_hat; X_tilde) - alpha phi_mlp(Z_hat)
        - the potentials use W_q W_k^T (the theory), not W_v, W_o
        - z: (B, N, d); context_pe: (B, N_x, d), X_tilde = X + P_X
        - returns: (B,)
        """
        z_hat = z + self.pe_z()
        self_energy = et_energy if self.mode == "et" else forced_energy
        return (0.5 * self.rho * z.pow(2).sum(dim=(-2, -1))
                + self_energy(self.self_attn, z_hat)
                - self.gamma2 * cross_energy(self.cross_attn, z_hat, context_pe)
                - self.alpha * mlp_energy(self.mlp, z_hat))

    @torch.no_grad()
    def descend(self, z0, context_pe, tol=1e-5, max_steps=100):
        """Repeat forward_step until convergence, per sample.

        - local dist  ||Z_{t+1} - Z_t||_F; a sample has converged at step t+1 when its local dist < tol,
          then it is frozen (no more updates)
        - global dist ||Z_t - Z_0||_F, t >= 1
        - stops when every sample has converged, or after max_steps steps
        - z0: (B, N, d); context_pe: (B, N_x, d)
        - returns a dict:
            z_star      (B, N, d)
            steps       (B,)  number of steps to converge; -1 if not converged within max_steps
            energy      (T + 1, B)  E(Z_t), t = 0..T
            local_dist  (T, B)
            global_dist (T, B)
        """
        z = z0
        steps = torch.full((z0.shape[0],), -1, dtype=torch.long, device=z0.device)
        energy, local_dist, global_dist = [self.get_energy(z, context_pe)], [], []
        for t in range(max_steps):
            z_next = self.forward_step(z, context_pe)
            active = (steps < 0).view(-1, 1, 1)
            z_next = torch.where(active, z_next, z)                        # converged samples stay frozen
            local = (z_next - z).flatten(1).norm(dim=-1)
            z = z_next
            energy.append(self.get_energy(z, context_pe))
            local_dist.append(local)
            global_dist.append((z - z0).flatten(1).norm(dim=-1))
            steps = torch.where((steps < 0) & (local < tol), torch.full_like(steps, t + 1), steps)
            if (steps >= 0).all():
                break
        return {"z_star": z, "steps": steps, "energy": torch.stack(energy),
                "local_dist": torch.stack(local_dist), "global_dist": torch.stack(global_dist)}


# ---------------------------------------------------------------------------------------------------------------------
# Base model
# ---------------------------------------------------------------------------------------------------------------------

_REGISTRY = {}


def register_model(name, **defaults):
    """Class decorator: register a model class under `name`, with its default config values (d_model, N, N_x, ...)."""
    def deco(cls):
        _REGISTRY[name] = (cls, defaults)
        return cls
    return deco


def build_model(name, mode="et", **overrides):
    """Build the model registered as `name`.

    - config = registered defaults, then mode, then overrides
    - example: build_model("sudoku_rrn", mode="forced", step_size=0.5)
    """
    if name not in _REGISTRY:
        raise KeyError(f"model '{name}' is not registered; available: {sorted(_REGISTRY)}")
    cls, defaults = _REGISTRY[name]
    config = EnergyLoopConfig(**{**defaults, "mode": mode, **overrides})
    return cls(config)


def list_models():
    return sorted(_REGISTRY)


@dataclass
class EnergyLoopConfig:
    """Hyper-parameters of a whole model (HF-style config, saved with the checkpoint).

    - sizes: d_model, n_tokens (N, tokens of Z = output side), n_context (N_x, tokens of X = input side)
    - core latent: mode ('et' | 'forced'), tie_mode ('orbit' | 'hard'), n_heads, rho, gamma2, alpha,
      step_size, beta (None -> 1 / sqrt(d_model / n_heads))
    - new E (self with +, no gamma1, no projection); step_size default 0.75 (< 2/L at init, 1 head, c = 1/4)
    - backward ('unroll' | 'ift'): how the gradient goes through the loop when grad is enabled (Task 5.2)
        unroll: exactly max_gradient_steps steps with grad, backprop through every step
        ift:    forward without grad to a fixed point (fwd_tol, fwd_max_steps), one step with grad,
                then (I - J^T) u = grad_{Z*} L solved by fixed-point iteration (bwd_tol, bwd_max_steps)
      switch at any time with model.config.backward = ...
    - forward without grad (inference, and the forward of 'ift'): a sample stops when its local dist
      ||Z_{t+1} - Z_t||_F < fwd_tol, at most fwd_max_steps steps
    - backward of 'ift': stops when ||u_{k+1} - u_k|| < bwd_tol ||u_{k+1}|| for every sample, at most bwd_max_steps
    - z0_init: 'trunc_normal' N(0, 1) truncated at +-3 std (decided, Task 3.3), or 'uniform' U(-1, 1)
    """
    d_model: int
    n_tokens: int
    n_context: int
    mode: str = "et"
    tie_mode: str = "orbit"
    n_heads: int = 1
    rho: float = 1.0
    gamma2: float = 1.0
    alpha: float = 1.0
    step_size: float = 0.75
    beta: float | None = None
    mlp_ratio: int = 4
    backward: str = "unroll"
    max_gradient_steps: int = 100
    fwd_tol: float = 1e-5                    # 1e-5, not 1e-7: training is float32, local dist floor ~5e-7 (C.5 Task 6)
    fwd_max_steps: int = 200                 # 200, not 2000: inference to the fixed point (C.5 Task 9)
    bwd_tol: float = 1e-5
    bwd_max_steps: int = 200
    z0_init: str = "trunc_normal"

    def to_dict(self):
        return asdict(self)


class EnergyLoopModel(nn.Module):
    """Base model: encoder -> core latent (loop to a fixed point) -> decoder.

    - forward(inputs, labels=None):
        X       = encoder(inputs)                       (B, N_x, d)
        X_tilde = X + P_X                                (once, core.add_context_pe)
        Z0      = init_state(B)                          (B, N, d)
        Z*      = loop Z <- step(Z) = core.forward_step(Z, X_tilde)
                    grad + 'unroll': solve_unroll; grad + 'ift': solve_fixed_point then implicit_step;
                    no grad: loop='fixed_steps' (default, solve_unroll) or 'fixed_point' (solve_fixed_point)
        outputs = decoder(Z*)
        loss    = compute_loss(outputs, labels)
    - subclasses (models/tasks/) define build_encoder, build_decoder, compute_loss and predict
    - init: the core latent inits itself (trunc_normal, PE, retraction, tie); init_encoder_decoder for the rest
    - training with Forced: optimizer.step(), then model.retract()
    - after loss.backward() with 'ift': model.backward_info = {"iters", "residual" (B,), "converged" (B,)}
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        c = config
        self.core = LatentEnergyLayer(c.d_model, c.n_tokens, c.n_context, mode=c.mode, tie_mode=c.tie_mode,
                                      rho=c.rho, gamma2=c.gamma2, alpha=c.alpha,
                                      step_size=c.step_size, mlp_ratio=c.mlp_ratio, beta=c.beta,
                                      n_heads=c.n_heads)
        self.encoder = self.build_encoder()
        self.decoder = self.build_decoder()
        self.init_encoder_decoder()
        self.backward_info = None

    # --- subclass hooks ---------------------------------------------------------------------------------------------
    def build_encoder(self):
        """nn.Module: inputs -> X (B, N_x, d), without positional encoding (P_X is added by the core)."""
        raise NotImplementedError

    def build_decoder(self):
        """nn.Module: Z* (B, N, d) -> outputs."""
        raise NotImplementedError

    def init_encoder_decoder(self):
        """Init of the encoder and decoder weights (the core latent inits itself)."""
        raise NotImplementedError

    def compute_loss(self, outputs, labels):
        raise NotImplementedError

    def predict(self, outputs):
        """outputs -> prediction in the format of the dataset metric."""
        raise NotImplementedError

    # --- shared -----------------------------------------------------------------------------------------------------
    def init_state(self, batch_size):
        """Z0, (B, N, d): N(0, 1) truncated at +-3 std, or U(-1, 1)."""
        w = self.core.pe_z.weight
        shape = (batch_size, self.config.n_tokens, self.config.d_model)
        if self.config.z0_init == "trunc_normal":
            z = torch.empty(shape, dtype=w.dtype, device=w.device)
            return nn.init.trunc_normal_(z, mean=0.0, std=1.0, a=-3.0, b=3.0)
        if self.config.z0_init == "uniform":
            return torch.rand(shape, dtype=w.dtype, device=w.device) * 2 - 1
        raise ValueError(f"z0_init is {self.config.z0_init}")

    def step(self, z, x_tilde):
        """One update step T(Z) = core.forward_step(Z, X_tilde) = Z - step_size grad E(Z), (B, N, d)."""
        return self.core.forward_step(z, x_tilde)

    @staticmethod
    def _info(z, z0, steps, local):
        return {"steps": steps, "converged": steps >= 0, "local_dist": local.detach(),
                "global_dist": (z - z0).flatten(1).norm(dim=-1).detach()}

    def solve_unroll(self, x_tilde, z0):
        """Exactly max_gradient_steps steps for every sample; with grad, the gradient goes back through every step.

        - also used without grad (loop='fixed_steps', the default at inference / test): same steps, no graph

        - returns z (B, N, d) and info:
            steps       (B,)  first step with local dist < fwd_tol, -1 if never (recorded only, no early stop)
            converged   (B,)  steps >= 0
            local_dist  (B,)  local dist of the last step
            global_dist (B,)  ||Z_end - Z0||_F
        """
        c = self.config
        z = z0
        steps = torch.full((z0.shape[0],), -1, dtype=torch.long, device=z0.device)
        local = torch.zeros(z0.shape[0], dtype=z0.dtype, device=z0.device)
        for t in range(c.max_gradient_steps):
            z_next = self.step(z, x_tilde)
            local = (z_next - z).flatten(1).norm(dim=-1).detach()
            z = z_next
            steps = torch.where((steps < 0) & (local < c.fwd_tol), torch.full_like(steps, t + 1), steps)
        return z, self._info(z, z0, steps, local)

    @torch.no_grad()
    def solve_fixed_point(self, x_tilde, z0):
        """Without grad: loop to a fixed point.

        - a sample stops (frozen) once its local dist ||Z_{t+1} - Z_t||_F < fwd_tol
        - the loop ends when every sample has stopped, or after fwd_max_steps
        - returns z (B, N, d), no graph, and info as solve_unroll (steps = -1: not converged, V4 of ift.md)
        """
        c = self.config
        z = z0
        steps = torch.full((z0.shape[0],), -1, dtype=torch.long, device=z0.device)
        local = torch.zeros(z0.shape[0], dtype=z0.dtype, device=z0.device)
        for t in range(c.fwd_max_steps):
            z_next = self.step(z, x_tilde)
            z_next = torch.where((steps < 0).view(-1, 1, 1), z_next, z)
            local = (z_next - z).flatten(1).norm(dim=-1)
            z = z_next
            steps = torch.where((steps < 0) & (local < c.fwd_tol), torch.full_like(steps, t + 1), steps)
            if bool((steps >= 0).all()):
                break
        return z, self._info(z, z0, steps, local)

    def implicit_step(self, z_star, x_tilde):
        """One step with grad at Z*, whose backward gives the IFT gradient (models/theory/ift.md).

        - Y = T(Z*) with Z* detached: the graph goes only into theta and X_tilde
        - a hook on Y replaces the incoming gradient g = grad_{Z*} L by u, the solution of (I - J^T) u = g,
          J = dT/dZ at Z*; then autograd continues: dL/dtheta = (dT/dtheta)^T u, dL/dX_tilde = (dT/dX_tilde)^T u
        - u by fixed-point iteration u <- J^T u + g, u_0 = g; J^T v = one VJP through a second copy of the step
          F = T(Z_s) with Z_s = Z* (requires grad) and X_tilde detached; never builds J
        - stops when ||u_{k+1} - u_k|| < bwd_tol ||u_{k+1}|| for every sample, or after bwd_max_steps;
          writes self.backward_info = {"iters", "residual" (B,), "converged" (B,)}
        - returns Y (B, N, d), to feed the decoder
        """
        c = self.config
        z_star = z_star.detach()
        y = self.step(z_star, x_tilde)
        if not y.requires_grad:
            return y
        z_s = z_star.clone().requires_grad_()
        f = self.step(z_s, x_tilde.detach())

        def solve_u(g):
            u = g
            iters, residual = 0, torch.zeros(g.shape[0], dtype=g.dtype, device=g.device)
            done = torch.zeros(g.shape[0], dtype=torch.bool, device=g.device)
            for k in range(c.bwd_max_steps):
                u_next = torch.autograd.grad(f, z_s, u, retain_graph=True)[0] + g
                residual = (u_next - u).flatten(1).norm(dim=-1) / u_next.flatten(1).norm(dim=-1).clamp_min(1e-300)
                u = u_next
                iters = k + 1
                done = residual < c.bwd_tol
                if bool(done.all()):
                    break
            self.backward_info = {"iters": iters, "residual": residual.detach(), "converged": done}
            return u

        y.register_hook(solve_u)
        return y

    def solve(self, x_tilde, z0, loop="fixed_steps"):
        """Z* and info.

        - grad enabled: by config.backward ('unroll' -> solve_unroll; 'ift' -> solve_fixed_point then implicit_step);
          loop is ignored
        - no grad: by loop ('fixed_steps' -> solve_unroll, max_gradient_steps steps, the number used in training;
          'fixed_point' -> solve_fixed_point, to fwd_tol, at most fwd_max_steps)
        """
        if not torch.is_grad_enabled():
            if loop == "fixed_steps":
                return self.solve_unroll(x_tilde, z0)
            if loop == "fixed_point":
                return self.solve_fixed_point(x_tilde, z0)
            raise ValueError(f"loop is {loop}")
        if self.config.backward == "unroll":
            return self.solve_unroll(x_tilde, z0)
        if self.config.backward == "ift":
            z_star, info = self.solve_fixed_point(x_tilde, z0)
            return self.implicit_step(z_star, x_tilde), info
        raise ValueError(f"backward is {self.config.backward}")

    def get_energy(self, inputs, z):
        """E(Z; X) per sample, (B,), for the inputs of a batch."""
        x_tilde = self.core.add_context_pe(self.encoder(inputs))
        return self.core.get_energy(z, x_tilde)

    def retract(self):
        """Forced: pull A = W_q W_k^T back to symmetric (call after optimizer.step()); no-op for ET."""
        self.core.retract()

    def forward(self, inputs, labels=None, loop="fixed_steps"):
        """inputs (B, N_x) -> dict: loss (None without labels), outputs, z_star, steps, converged, local_dist,
        global_dist (see solve); loop ('fixed_steps' | 'fixed_point') is used only without grad."""
        x_tilde = self.core.add_context_pe(self.encoder(inputs))
        z_star, info = self.solve(x_tilde, self.init_state(inputs.shape[0]), loop=loop)
        outputs = self.decoder(z_star)
        loss = self.compute_loss(outputs, labels) if labels is not None else None
        return {"loss": loss, "outputs": outputs, "z_star": z_star, **info}
