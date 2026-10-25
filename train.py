"""Train a model on a dataset; periodic test (ID, OOD); checkpoints; W&B (Phase C.5, Task 5, 6).

Run:
    python train.py --config config/<task>.yaml [--set key=value ...]                 new run
    python train.py --config config/<task>.yaml --set model.backward=ift --init_from <run>/checkpoints/convergence_checkpoint.pt
        run B (Task 6.1): a new run that starts from the checkpoint (model, optimizer, epoch / step); the same code
        as an ift run from scratch, only the starting state differs
    python train.py --resume results/<run>                                             continue an interrupted run
        (Task 6.3): reads <run>/config.yaml and <run>/checkpoints/latest_checkpoint.pt, continues the same W&B run

Done (Task 5.1-5.8, 6):
    - build_optimizer(model, cfg): adam | adamw from cfg["train"] (lr, betas, weight_decay)
    - train_step(model, optimizer, batch, dataset, device, dtype, tie_self, tie_cross): one optimizer step on one batch,
      loss = task loss + tie_self * R_self + tie_cross * R_cross (tie coefficients from cfg["train"]);
      Forced: model.retract() after the step
    - main(): config, seed, device, datasets, model, optimizer, run folder (config.yaml, train.log), W&B, then the loop
      by train.schedule (Task 5.6):
        epoch: one pass over the train set per epoch (DataLoader shuffle); x axis = epoch; train logged every epoch
        step:  new batches drawn on the fly up to train.iterations; x axis = step; train logged every LOG_EVERY steps
      test every train.test_every (epochs or steps) on every split except "train" (Task 5.5), loop by backward
      (engine.test_loop); 4 fixed samples per split to a wandb.Table at each test (Task 4.3)
    - logged (Task 4.1, 4.2): train/loss (total), train/<metric>, train/grad_norm, tie/self_mse, tie/cross_mse,
      ift/* (backward 'ift' only), test_id/loss, test_id/<metric>, test_ood/<metric>; s/step on the terminal only (5.7)
    - checkpoints (Task 6, utils/checkpoint.py), in <run>/checkpoints/:
        latest_checkpoint.pt every epoch (schedule 'epoch') / every test (schedule 'step'), overwritten
        best_checkpoint.pt when the test_id metric (dataset BEST_METRIC: Sudoku board_acc, toy mse) beats the best
        convergence_checkpoint.pt once (backward 'unroll' only): test_id runs one pass of train.converge_steps steps
          (loss / metric read at max_gradient_steps); saved when >= converge_frac of test_id has local dist
          < converge_tol within those steps; the % is printed at every test
"""
import argparse
import random
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

import data  # noqa: F401  (registers the datasets)
import models
from data.base import get_dataset
from engine import run_inference, test_loop, to_device
from utils.checkpoint import (BEST, CHECKPOINT_DIR, CONVERGENCE, LATEST, load_checkpoint, save_checkpoint,
                              set_rng_state)
from utils.config import load_config, make_run_dir, pick_seed, save_config
from utils.logger import Logger
from utils.wandb_utils import init_run, startup_info

LOG_EVERY = 10            # schedule 'step': train logged every 10 steps, mean over them (Task 4.2)
N_SAMPLES = 4             # fixed samples per split for the W&B table (Task 4.3)


def build_optimizer(model, cfg):
    """Optimizer from cfg["train"]: optimizer (adam | adamw), lr, betas, weight_decay."""
    t = cfg["train"]
    optimizers = {"adam": torch.optim.Adam, "adamw": torch.optim.AdamW}
    if t["optimizer"] not in optimizers:
        raise ValueError(f"optimizer is {t['optimizer']}; available: {sorted(optimizers)}")
    return optimizers[t["optimizer"]](model.parameters(), lr=t["lr"], betas=tuple(t["betas"]),
                                      weight_decay=t["weight_decay"])


def train_step(model, optimizer, batch, dataset, device, dtype, tie_self, tie_cross):
    """One optimizer step on one batch.

    - forward with grad: Z* by config.backward (unroll: max_gradient_steps steps; ift: fixed point, then IFT backward)
    - loss = task loss (model.compute_loss: CE for Sudoku, MSE for the matrix tasks)
             + tie_self * R_self + tie_cross * R_cross,   R = model.core.tie_penalty() (relative tie error, heads.py)
      the tie term is added here, not in model.forward: out["loss"] stays the task loss (test loss without tie)
    - loss.backward(), total L2 norm of the gradient (before the step), optimizer.step()
    - Forced: model.retract() after the step (every A_h back to symmetric); no-op for ET
    - returns dict:
        loss        float, the total loss (task + tie), logged as train/loss
        tie         dict {"self", "cross"}: R before the step (the values in the loss), logged as tie/self, tie/cross
        metric      dict, dataset.metric(model.predict(outputs), labels)   (Sudoku: board_acc, cell_acc; toy: mse)
        grad_norm   float
        steps, converged, local_dist, global_dist   (B,) tensors of the forward (see EnergyLoopModel.solve)
        backward    model.backward_info (iters, residual (B,), converged (B,)) with backward 'ift', else None
    """
    model.train()
    batch = to_device(batch, device, dtype)
    out = model(batch["inputs"], batch["labels"])
    tie = model.core.tie_penalty()
    loss = out["loss"] + tie_self * tie["self"] + tie_cross * tie["cross"]
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    grad_norm = torch.sqrt(sum(p.grad.pow(2).sum() for p in model.parameters() if p.grad is not None))
    optimizer.step()
    model.retract()
    with torch.no_grad():
        metric = dataset.metric(model.predict(out["outputs"].detach()), batch["labels"])
    return {
        "loss": loss.item(),
        "tie": {k: v.item() for k, v in tie.items()},
        "metric": metric,
        "grad_norm": grad_norm.item(),
        **{k: out[k] for k in ("steps", "converged", "local_dist", "global_dist")},
        "backward": model.backward_info if model.config.backward == "ift" else None,
    }


def resolve_device(name):
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class TrainWindow:
    """Train results of the steps since the last train log; mean() gives the W&B / terminal values (Task 4.1, 4.2)."""

    def __init__(self, bwd_max_steps):
        self.bwd_max_steps = bwd_max_steps
        self.reset()

    def reset(self):
        self.values, self.ift, self.t0 = defaultdict(list), defaultdict(list), time.time()

    def add(self, r):
        self.values["train/loss"].append(r["loss"])
        for k, v in r["metric"].items():
            self.values[f"train/{k}"].append(v)
        self.values["train/grad_norm"].append(r["grad_norm"])
        self.values["tie/self_mse"].append(r["tie"]["self"])
        self.values["tie/cross_mse"].append(r["tie"]["cross"])
        if r["backward"] is not None:
            bw = r["backward"]
            self.ift["ift/mean_backward_iterations"].append(bw["iters"])
            self.ift["ift/mean_backward_residual"].append(bw["residual"].mean().item())
            self.ift["ift/percent_steps_hitting_bwd_max_steps"].append(100.0 * (bw["iters"] >= self.bwd_max_steps))

    def mean(self):
        """(dict of means, seconds per step); resets the window."""
        n = len(self.values["train/loss"])
        out = {k: float(np.mean(v)) for k, v in {**self.values, **self.ift}.items()}
        sec = (time.time() - self.t0) / max(n, 1)
        self.reset()
        return out, sec


def fixed_samples(dataset, n=N_SAMPLES):
    """The first n samples of a split, stacked; drawn once at start (toy tasks draw new data at every call)."""
    items = [dataset[i] for i in range(n)]
    return {k: torch.stack([it[k] for it in items]) for k in items[0]}


@torch.no_grad()
def sample_table(model, dataset, samples, loop):
    """wandb.Table of the fixed samples (Task 4.3), or None if the dataset has no format_sample."""
    import wandb

    model.eval()
    p = next(model.parameters())
    b = to_device(samples, p.device, p.dtype)
    pred = model.predict(model(b["inputs"], loop=loop)["outputs"]).cpu()
    rows = [dataset.format_sample(samples["inputs"][i], pred[i], samples["labels"][i]) for i in range(len(pred))]
    if rows[0] is None:
        return None
    columns = ["sample"] + list(rows[0])
    return wandb.Table(columns=columns, data=[[i] + list(r.values()) for i, r in enumerate(rows)])


def fmt(values):
    return "  ".join(f"{k.split('/', 1)[1] if k.startswith('train/') else k} {v:.4g}" for k, v in values.items())


def parse_train_args():
    ap = argparse.ArgumentParser(description="Train a loop model (Phase C.5)")
    ap.add_argument("--config", help="config/<task>.yaml (not with --resume)")
    ap.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE",
                    help="override config keys, e.g. model.mode=forced model.backward=ift wandb.mode=disabled")
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--init_from", help="checkpoint to start a new run from (run B: convergence_checkpoint.pt)")
    g.add_argument("--resume", help="run folder results/<run> to continue (latest_checkpoint.pt)")
    args = ap.parse_args()
    if args.resume and (args.config or args.set):
        ap.error("--resume reads <run>/config.yaml; no --config / --set")
    if not args.resume and not args.config:
        ap.error("--config is required (or --resume)")
    return args


def main():
    args = parse_train_args()
    ckpt = None
    if args.resume:
        run_dir = Path(args.resume)
        cfg = yaml.safe_load((run_dir / "config.yaml").read_text(encoding="utf-8"))    # as saved, with seed
        ckpt = load_checkpoint(run_dir / CHECKPOINT_DIR / LATEST)
    else:
        cfg = load_config(args.config, args.set)
        cfg["seed"] = pick_seed()
        if args.init_from:
            ckpt = load_checkpoint(args.init_from)
            cfg["init_from"] = {"path": str(Path(args.init_from).resolve()), ckpt["x_name"]: ckpt["x"]}
    t = cfg["train"]
    for key in ("tie_self", "tie_cross"):
        if t[key] is None:
            raise ValueError(f"train.{key} is not set (config/{cfg['task']}.yaml or --set train.{key}=...)")
    seed = cfg["seed"]
    seed_everything(seed)
    device, dtype = resolve_device(cfg["device"]), getattr(torch, cfg["dtype"])
    x_name = "epoch" if t["schedule"] == "epoch" else "step"

    train_set = get_dataset(cfg["task"], "train")
    datasets = {split: train_set if split == "train" else get_dataset(cfg["task"], split)
                for split in type(train_set).SPLITS}
    test_splits = [s for s in datasets if s != "train"]
    test_size, test_batch = cfg["data"]["test_size"], cfg["data"]["test_batch_size"]
    test_loaders = {s: DataLoader(datasets[s] if test_size is None
                                  else torch.utils.data.Subset(datasets[s], range(test_size)),
                                  batch_size=test_batch, shuffle=False) for s in test_splits}
    samples = {s: fixed_samples(datasets[s]) for s in datasets}

    model = models.build_model(cfg["task"], **cfg["model"]).to(device, dtype)
    optimizer = build_optimizer(model, cfg)
    start, best = 1, None
    if ckpt is not None:
        if ckpt["x_name"] != x_name:
            raise ValueError(f"checkpoint counts {ckpt['x_name']}, this run counts {x_name}")
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start = ckpt["x"] + 1
        if args.resume:
            best = ckpt["best"]
    loop = test_loop(model)
    best_key, best_mode = type(train_set).BEST_METRIC

    if not args.resume:
        run_dir = make_run_dir(cfg)
        save_config(cfg, run_dir / "config.yaml")
    ckpt_dir = run_dir / CHECKPOINT_DIR
    converge_saved = (ckpt_dir / CONVERGENCE).exists()
    log = Logger("train", run_dir / "train.log", config=cfg, mode="a" if args.resume else "w")
    mc = model.config
    log.kv("run folder", run_dir)
    log.kv("device", f"{device}, {cfg['dtype']}")
    log.kv("model", f"{cfg['task']}, {mc.mode}, {mc.backward}, {mc.n_heads} head{'s' if mc.n_heads > 1 else ''}")
    log.kv("size", f"d_model {mc.d_model}, n_tokens {mc.n_tokens}, n_context {mc.n_context}, "
                   f"params {sum(p.numel() for p in model.parameters()):,}")
    log.kv("loop", f"max_gradient_steps {mc.max_gradient_steps}, step_size {mc.step_size}")
    log.kv("splits", ", ".join(f"{s} {len(d):,}" for s, d in datasets.items()))
    log.kv("test", f"{', '.join(test_splits)} ({'all samples' if test_size is None else f'{test_size} samples'}), "
                   f"loop {loop}, every {t['test_every']} {x_name}")
    if loop == "fixed_steps":
        log.kv("converge", f"local dist < {t['converge_tol']:g} within {t['converge_steps']} steps, save at "
                           f">= {100 * t['converge_frac']:g}% of test_id")
    if ckpt is not None:
        log.kv("resume" if args.resume else "init from", f"{args.resume or args.init_from}, start at {x_name} "
                                                         f"{start}, best {best_key} {best}")
    run = init_run(cfg, run_dir, info=startup_info(cfg, seed, model, datasets),
                   resume_id=ckpt["wandb_id"] if args.resume else None)
    run.define_metric(x_name)
    run.define_metric("*", step_metric=x_name)

    window = TrainWindow(model.config.bwd_max_steps)

    def save(name, x):
        save_checkpoint(ckpt_dir / name, model, optimizer, x_name, x, cfg, best=best, wandb_id=run.id)

    def log_train(x):
        values, sec = window.mean()
        run.log({x_name: x, **values})
        log.info(f"{x_name} {x} | train  {fmt(values)} | {sec:.3f} s/step")

    def test(x):
        nonlocal best, converge_saved
        values, converged = {}, None
        for s in test_splits:
            converge = (t["converge_steps"], t["converge_tol"]) if s == "test_id" and loop == "fixed_steps" else None
            r = run_inference(model, test_loaders[s], datasets[s], loop=loop, converge=converge)
            if s != "test_ood":                                      # test OOD: metric only (Task 4.1)
                values[f"{s}/loss"] = r["loss"]
            values.update({f"{s}/{k}": v for k, v in r["metric"].items()})
            if s == "test_id":
                converged = r["converged"]
        tables = {}
        for s in datasets:
            table = sample_table(model, datasets[s], samples[s], loop)
            if table is not None:
                tables[f"samples/{s}"] = table
        run.log({x_name: x, **values, **tables})
        within = f"local dist < {t['converge_tol']:g} within {t['converge_steps']} steps" if loop == "fixed_steps" \
            else f"fixed point, local dist < {model.config.fwd_tol:g} within {model.config.fwd_max_steps} steps"
        log.info(f"{x_name} {x} | test   {fmt(values)} | test_id converged {100 * converged:.1f}% ({within})")
        v = values[f"test_id/{best_key}"]
        if best is None or (v > best if best_mode == "max" else v < best):
            best = v
            save(BEST, x)
            log.info(f"{x_name} {x} | saved {BEST} (test_id/{best_key} {v:.4g})")
        if model.config.backward == "unroll" and not converge_saved and converged >= t["converge_frac"]:
            save(CONVERGENCE, x)
            converge_saved = True
            log.info(f"{x_name} {x} | saved {CONVERGENCE} (test_id converged {100 * converged:.1f}% "
                     f">= {100 * t['converge_frac']:g}%)")

    def step(batch):
        window.add(train_step(model, optimizer, batch, train_set, device, dtype, t["tie_self"], t["tie_cross"]))

    if ckpt is not None and args.resume:
        set_rng_state(ckpt["rng"])                                   # same data order as without the interruption
    batch_size = cfg["data"]["batch_size"]
    if t["schedule"] == "epoch":
        loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
        for epoch in range(start, t["epochs"] + 1):
            for batch in loader:
                step(batch)
            log_train(epoch)
            if t["test_every"] and epoch % t["test_every"] == 0:
                test(epoch)
            save(LATEST, epoch)
    elif t["schedule"] == "step":
        batches = iter(DataLoader(train_set, batch_size=batch_size))   # data drawn on the fly, len 1e7
        for it in range(start, t["iterations"] + 1):
            step(next(batches))
            if it % LOG_EVERY == 0:
                log_train(it)
            if t["test_every"] and it % t["test_every"] == 0:
                test(it)
                save(LATEST, it)
    else:
        raise ValueError(f"train.schedule is {t['schedule']}")
    run.finish()
    log.info(f"done; run folder {run_dir}")
    log.close()


if __name__ == "__main__":
    main()
