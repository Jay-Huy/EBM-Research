"""Run a trained model on a test split to the fixed point, save its outputs and the per-step curves (Phase C.5, Task 7, 8).

Run:
    python inference.py --run results/<run> --split test_id [--checkpoint best] [--n_z0 1] [--seed 0] [--save_z]
    - always to the fixed point (local dist < fwd_tol, at most fwd_max_steps), whatever the run was trained with;
      the result at max_gradient_steps (fixed steps) is a point on the same curve, so one inference covers both loops
    - --checkpoint: best (default) | latest | convergence   (results/<run>/checkpoints/<name>_checkpoint.pt)
    - --n_z0 K: K runs from K random Z0 (seeds seed, seed + 1, ...); the same input can give another output (C.6)
    - --save_z: also save Z* (off by default: large)
Out: results/<run>/inference/<split>-<checkpoint>/
    inference.pt   dict: cfg, checkpoint (path, epoch / step), split, max_gradient_steps, fwd_tol, seeds;
                   per Z0: metric and loss at the fixed point, metric_at_T (at max_gradient_steps);
                   metric_curve {key: (K, S)} the split metric at every step; pred (K, N, ...) at the fixed point,
                   labels (N, ...), steps_to_converge, final_local, energy_star (K, N); local, glob, energy (K, N, S)
                   per step t = 1..S; z_star (K, N, N_tok, d) with --save_z   (eval.py reads this file, Task 8)
    curves.png     4 panels per step (first Z0), short titles (a figure for the paper): the metric (dataset
                   PLOT_METRIC, e.g. Sudoku: Accuracy, board_acc bold on top, cell_acc thin) with its value at max_gradient_steps (dotted) and at
                   the fixed point (dash-dot); local distance (log), global distance, energy (median over the split in
                   red, 16 samples in blue); a dashed vertical line at max_gradient_steps in every panel. The numbers
                   and the legend of the figure are written to inference.log
    inference.log  header (run, checkpoint, options) and the results
Not logged to W&B (Task 4.1). Input = a split of the run's task; other inputs: later.
"""
import argparse
import sys
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader, Subset

import data  # noqa: F401  (registers the datasets)
import models
from data.base import get_dataset
from engine import pad_steps, run_inference
from utils.checkpoint import CHECKPOINT_DIR, load_checkpoint
from utils.logger import Logger

N_THIN = 16               # samples drawn as thin lines in curves.png


def parse_args():
    ap = argparse.ArgumentParser(description="Inference of a trained run (Phase C.5 Task 7)")
    ap.add_argument("--run", required=True, help="results/<run>")
    ap.add_argument("--split", required=True, help="a split of the task, e.g. test_id, test_ood")
    ap.add_argument("--checkpoint", default="best", choices=["best", "latest", "convergence"])
    ap.add_argument("--n_z0", type=int, default=1, help="number of random Z0")
    ap.add_argument("--seed", type=int, default=0, help="seed of the first Z0")
    ap.add_argument("--save_z", action="store_true", help="also save Z*")
    ap.add_argument("--batch_size", type=int, help="default data.test_batch_size of the run")
    ap.add_argument("--device", default="auto")
    return ap.parse_args()


def asymmetry(model):
    """Forced: max over heads of ||A_h - A_h^T|| / ||A_h||, A_h = W_q^h W_k^h^T; None for ET."""
    blk = model.core.self_attn
    if model.config.mode != "forced":
        return None
    d_h, out = blk.d_head, 0.0
    for h in range(blk.n_heads):
        sl = slice(h * d_h, (h + 1) * d_h)
        A = blk.W_q.weight[sl].T.double() @ blk.W_k.weight[sl].double()
        out = max(out, ((A - A.T).norm() / A.norm()).item())
    return out


def plot(res, metric, path):
    """curves.png, 4 panels per step, short titles (figure for the paper; details go to inference.log):
    metric (dataset PLOT_METRIC; dotted: value at max_gradient_steps, dash-dot: at the fixed point), local distance
    (log), global distance, energy (median over the split in red, N_THIN samples in blue); a dashed vertical line at
    max_gradient_steps in every panel."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator
    keys, title, label, ylim = metric
    T = res["max_gradient_steps"]
    S = res["local"].shape[-1]
    t = torch.arange(1, S + 1)
    fig, axes = plt.subplots(1, 4, figsize=(18, 3.6))
    ax = axes[0]
    values = {}
    styles = [{"color": "C0", "lw": 2.4, "zorder": 3}, {"color": "C1", "lw": 1.3, "alpha": 0.8, "zorder": 2}]
    for i, (key, name) in enumerate(keys.items()):      # first key = main line: bold, on top
        st = styles[min(i, 1)]
        y = res["metric_curve"][key][0]
        at_T, at_fp = float(y[min(T, S) - 1]), float(y[-1])
        values[key] = (at_T, at_fp)
        ax.plot(t, y, label=name, **st)
        ax.axhline(at_T, color=st["color"], ls=":", lw=1, zorder=1)
        ax.axhline(at_fp, color=st["color"], ls="-.", lw=1, zorder=1)
        ax.plot([min(T, S), S], [at_T, at_fp], "o", color=st["color"], ms=5 if i == 0 else 4, zorder=st["zorder"])
    if len(keys) > 1:
        ax.legend(fontsize=8, loc="best")
    ax.set_title(title)
    ax.set_ylabel(label)
    if ylim is not None:
        ax.set_ylim(*ylim)
    for ax, (k, name, ylab, logy) in zip(axes[1:], (("local", "Local distance", r"$\|Z_t - Z_{t-1}\|_F$", True),
                                                     ("glob", "Global distance", r"$\|Z_t - Z_1\|_F$", False),
                                                     ("energy", "Energy", r"$E(Z_t)$", False))):
        v = res[k][0].double()
        if k == "local":                        # after a sample stops its local dist is 0 (frozen): not drawn
            stc = res["steps_to_converge"][0]
            v = torch.where(t.view(1, -1) > torch.where(stc > 0, stc, S).view(-1, 1), float("nan"), v)
        ax.plot(t, v[:N_THIN].T, color="C0", alpha=0.3, lw=0.8)
        ax.plot(t, v.nanmedian(0).values, color="C3", lw=2)
        if logy:
            ax.set_yscale("log")
        ax.set_title(name)
        ax.set_ylabel(ylab)
    for ax in axes:
        ax.axvline(T, color="k", ls="--", lw=1)
        ax.set_xlabel("Step")
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return values


def main():
    args = parse_args()
    run_dir = Path(args.run)
    cfg = yaml.safe_load((run_dir / "config.yaml").read_text(encoding="utf-8"))
    ckpt_path = run_dir / CHECKPOINT_DIR / f"{args.checkpoint}_checkpoint.pt"
    device = torch.device(("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device)
    dtype = getattr(torch, cfg["dtype"])
    ckpt = load_checkpoint(ckpt_path, device)

    model = models.build_model(cfg["task"], **cfg["model"]).to(device, dtype)
    model.load_state_dict(ckpt["model"])
    T, c = model.config.max_gradient_steps, model.config
    dataset = get_dataset(cfg["task"], args.split)
    size = cfg["data"].get("test_size")
    loader = DataLoader(dataset if size is None else Subset(dataset, range(size)),
                        batch_size=args.batch_size or cfg["data"]["test_batch_size"], shuffle=False)

    name = f"{args.split}-{args.checkpoint}"
    out_dir = run_dir / "inference" / name
    out_dir.mkdir(parents=True, exist_ok=True)
    options = {"run": str(run_dir), "checkpoint": str(ckpt_path), ckpt["x_name"]: ckpt["x"], "split": args.split,
               "max_gradient_steps": T, "fwd_tol": c.fwd_tol, "fwd_max_steps": c.fwd_max_steps, "n_z0": args.n_z0, "seed": args.seed, "save_z": args.save_z,
               "device": str(device)}
    log = Logger("inference", out_dir / "inference.log", config=options)
    log.kv("run", f"{cfg['task']}, {c.mode}, {c.backward}, {c.n_heads} head{'s' if c.n_heads > 1 else ''}")
    log.kv("size", f"d_model {c.d_model}, n_tokens {c.n_tokens}, n_context {c.n_context}, "
                   f"params {sum(p.numel() for p in model.parameters()):,}")
    log.kv("checkpoint", f"{ckpt_path.name} ({ckpt['x_name']} {ckpt['x']})")
    log.kv("split", f"{args.split} ({len(loader.dataset):,} samples)")
    log.kv("loop", f"to the fixed point (local dist < {c.fwd_tol:g}, at most {c.fwd_max_steps} steps), "
                   f"max_gradient_steps {T}")
    asym = asymmetry(model)
    if asym is not None:
        log.kv("forced", f"max relative asymmetry of A_h = {asym:.2e}")

    results = []
    for k in range(args.n_z0):
        torch.manual_seed(args.seed + k)
        r = run_inference(model, loader, dataset, record=True, save_z=args.save_z)
        stc = r["steps_to_converge"]
        conv = stc[stc >= 0]
        fmt = lambda m: ", ".join(f"{k} {v:.4f}" for k, v in m.items())
        log.kv("Z0", f"seed {args.seed + k}")
        log.kv("result", f"step {T}: {fmt(r['metric_at_T'])} | fixed point: {fmt(r['metric'])}, loss {r['loss']:.4f}")
        log.kv("converged", f"{100 * (stc >= 0).float().mean():.1f}%"
                            + (f", median steps {conv.median().item()}" if len(conv) else "")
                            + f", median E(Z*) {r['energy_star'].median():.4g}, {r['local'].shape[1]} steps recorded")
        results.append(r)

    S = max(r["local"].shape[1] for r in results)
    pad = {"local": False, "glob": True, "energy": True}
    save = {"cfg": cfg, "checkpoint": {"path": str(ckpt_path), ckpt["x_name"]: ckpt["x"]}, "split": args.split,
            "max_gradient_steps": T, "fwd_tol": c.fwd_tol, "seeds": [args.seed + k for k in range(args.n_z0)],
            "metric": [r["metric"] for r in results], "metric_at_T": [r["metric_at_T"] for r in results],
            "loss": [r["loss"] for r in results], "labels": results[0]["labels"],
            "metric_curve": {key: torch.stack([torch.cat([r["metric_curve"][key], r["metric_curve"][key][-1:].expand(
                S - len(r["metric_curve"][key]))]) for r in results]) for key in results[0]["metric_curve"]}}
    for key in ("pred", "steps_to_converge", "final_local", "energy_star"):
        save[key] = torch.stack([r[key] for r in results])
    for key, fill_last in pad.items():
        save[key] = torch.stack([pad_steps(r[key], S, fill_last) for r in results])
    if args.save_z:
        save["z_star"] = torch.stack([r["z_star"] for r in results])
    torch.save(save, out_dir / "inference.pt")
    metric = type(dataset).PLOT_METRIC
    values = plot(save, metric, out_dir / "curves.png")
    log.kv("figure", f"panel 1 {metric[1]} (first Z0): " + "; ".join(
        f"{metric[0][k]} {a:.4g} at step {T} (dotted), {b:.4g} at fixed point (dash-dot)" for k, (a, b) in values.items()))
    log.kv("figure", f"panels 2-4 local distance (log), global distance, energy; red median, blue {N_THIN} samples; "
                     f"dashed vertical line at max_gradient_steps {T}")
    log.kv("saved", f"{out_dir / 'inference.pt'}, curves.png")
    log.close()


if __name__ == "__main__":
    sys.exit(main())
