"""Save / load checkpoints (Phase C.5, Task 6).

- files under results/<run>/checkpoints/:
    latest_checkpoint.pt       overwritten every epoch (schedule 'epoch') / every test (schedule 'step'); for --resume
    best_checkpoint.pt         when the test_id metric beats the best so far (Sudoku: board_acc, higher; toy: mse, lower)
    convergence_checkpoint.pt  once, the first test where >= converge_frac of test_id converges within converge_steps
                               (unroll runs only); start of run B (--init_from, backward 'ift')
- a checkpoint is a dict:
    model, optimizer   state_dict (the Forced retract_count is a buffer, so it is in the model state_dict)
    x_name, x          'epoch' | 'step' and the last finished one
    cfg                full run config (with seed); the run folder also has config.yaml to read directly
    best               best test_id metric so far
    wandb_id           W&B run id, to continue the same W&B run on --resume
    rng                python, numpy, torch (cpu, cuda) RNG states, so a resumed run draws the same data
"""
import argparse
import random

import numpy as np
import torch

CHECKPOINT_DIR = "checkpoints"
LATEST, BEST, CONVERGENCE = "latest_checkpoint.pt", "best_checkpoint.pt", "convergence_checkpoint.pt"


def rng_state():
    return {"python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}


def set_rng_state(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if torch.cuda.is_available() and state["cuda"]:
        torch.cuda.set_rng_state_all(state["cuda"])


def save_checkpoint(path, model, optimizer, x_name, x, cfg, best=None, wandb_id=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(), "x_name": x_name, "x": x,
                "cfg": cfg, "best": best, "wandb_id": wandb_id, "rng": rng_state()}, path)


def load_checkpoint(path, device="cpu"):
    """The checkpoint dict; argparse.Namespace allowed for checkpoints of older code that saved args."""
    torch.serialization.add_safe_globals([argparse.Namespace])
    return torch.load(path, map_location=device, weights_only=False)
