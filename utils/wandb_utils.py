"""Weights & Biases helpers (Phase C.5, Task 3): init, startup info, finish.

- one W&B project per task (config: wandb.project, e.g. 'ebm-inverse'), personal account (wandb.entity: null)
- run name '{mode}-{backward}-{n_heads}h' (utils.config.run_name); no group, no tags
- wandb.mode: 'online' (default) | 'offline' (no network; sync later with `wandb sync`) |
  'disabled' (test runs: every call is a no-op, nothing is sent or saved) — set with --set wandb.mode=disabled
- login once per machine: `wandb login` (the API key is never written in the repo)
- startup info logged into the run config: full config, seed, environment (GPU, torch, python, git commit),
  model size, data — see startup_info
- resume of the same run (continue training, Task 6): pass the W&B run id saved in the checkpoint;
  run B (IFT from converge_model.pt) is a new run whose config names the parent run and iteration

Example:
    run = init_run(cfg, run_dir, info=startup_info(cfg, seed, model, datasets))
    run.log({"train/loss": loss}, step=step)
    run.finish()
"""
import math
import platform
import sys

import torch

from utils.config import run_name
from utils.logger import git_commit


def environment():
    """GPU, torch, python, platform, git commit."""
    return {
        "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "torch": torch.__version__,
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "git_commit": git_commit(),
    }


def startup_info(cfg, seed, model=None, datasets=None):
    """What a run logs at start, besides its config.

    - seed: the training seed of this run (utils.config.pick_seed)
    - model: parameter count, total and per group (encoder, decoder, core latent); d_model, n_tokens, n_context
    - data, by train.schedule:
        step  (data drawn on the fly, toy tasks):  {"Batch size", "Training Iteration"}
        epoch (fixed train set, Sudoku, ...):      {"Batch size", "Num Batch", "Epochs"},
              Num Batch = batches per epoch = ceil(len(train set) / batch size), needs datasets["train"]
    - datasets: {split name: dataset}
    """
    info = {"seed": seed, "env": environment()}
    if model is not None:
        groups = {"encoder": getattr(model, "encoder", None), "decoder": getattr(model, "decoder", None),
                  "core": getattr(model, "core", None)}
        mc = getattr(model, "config", None)
        if mc is not None:
            info["model_dims"] = {k: getattr(mc, k) for k in ("d_model", "n_tokens", "n_context")}
        info["model_size"] = {"total": sum(p.numel() for p in model.parameters()),
                              **{k: sum(p.numel() for p in g.parameters()) for k, g in groups.items() if g is not None}}
    train = cfg.get("train", {})
    batch = cfg.get("data", {}).get("batch_size")
    if train.get("schedule") == "epoch":
        n_train = len(datasets["train"]) if datasets and "train" in datasets else None
        n_batch = math.ceil(n_train / batch) if n_train is not None and batch else None
        info["data"] = {"Batch size": batch, "Num Batch": n_batch, "Epochs": train.get("epochs")}
    else:
        info["data"] = {"Batch size": batch, "Training Iteration": train.get("iterations")}
    return info


def init_run(cfg, run_dir, info=None, resume_id=None):
    """Start (or resume) the W&B run of this training run.

    - project = cfg['wandb']['project'], entity = cfg['wandb']['entity'] (None: personal account)
    - name = '{mode}-{backward}-{n_heads}h'; files of W&B go to run_dir/wandb
    - config = the full run config + info (startup_info)
    - resume_id: the W&B run id of an interrupted run, to continue the same run (resume='must')
    """
    import wandb

    w = cfg["wandb"]
    return wandb.init(
        project=w["project"],
        entity=w.get("entity"),
        name=run_name(cfg),
        mode=w.get("mode", "online"),
        dir=str(run_dir),
        config={**cfg, **(info or {})},
        id=resume_id,
        resume="must" if resume_id else None,
    )
