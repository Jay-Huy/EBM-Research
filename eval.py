"""Print the setup and the results of one inference folder (Phase C.5, Task 8). Reads only inference.pt, no model.

Run:
    python eval.py --inference results/<run>/inference/<split>-<checkpoint> [--steps N ...]
    - setup: task, mode (et | forced), heads, training (unroll with max_gradient_steps steps | ift), split, checkpoint
    - results: metric at max_gradient_steps and at the fixed point (recomputed from pred, labels), % of samples
      converged and median steps; --steps N: the metric at step N too (read on the saved curve)
    - several Z0 (inference --n_z0 K): mean +- std over the Z0
Out: printed, and written to results/<run>/metrics/<split>-<checkpoint>.txt (overwritten). Not logged to W&B.
"""
import argparse
import sys
from pathlib import Path

import torch

import data  # noqa: F401  (registers the datasets)
from data.base import get_dataset


def parse_args():
    ap = argparse.ArgumentParser(description="Setup and results of one inference folder (Phase C.5 Task 8)")
    ap.add_argument("--inference", required=True, help="results/<run>/inference/<split>-<checkpoint>")
    ap.add_argument("--steps", type=int, nargs="*", default=[], help="also the metric at these steps")
    return ap.parse_args()


def mean_std(values):
    v = torch.tensor(values, dtype=torch.float64)
    return f"{v.mean():.4f}" if len(v) == 1 else f"{v.mean():.4f} +- {v.std():.4f}"


def main():
    args = parse_args()
    folder = Path(args.inference)
    r = torch.load(folder / "inference.pt", map_location="cpu", weights_only=False)
    cfg, m, T = r["cfg"], r["cfg"]["model"], r["max_gradient_steps"]
    dataset = get_dataset(cfg["task"], r["split"])
    curve, S = r["metric_curve"], next(iter(r["metric_curve"].values())).shape[1]
    keys = list(curve)
    ck = {k: v for k, v in r["checkpoint"].items() if k != "path"}

    lines = ["setup",
             f"  task        {cfg['task']}",
             f"  mode        {m['mode']}",
             f"  heads       {m['n_heads']}",
             f"  training    " + (f"unroll, {T} steps (max_gradient_steps)" if m["backward"] == "unroll" else
                                  f"ift (max_gradient_steps = {T}, not used in training)"),
             f"  split       {r['split']} ({len(r['labels'])} samples)",
             f"  checkpoint  {Path(r['checkpoint']['path']).name}, " + ", ".join(f"{k} {v}" for k, v in ck.items()),
             f"  Z0          {len(r['seeds'])} (seeds {r['seeds']})",
             "results"]
    rows = [(f"step {T} (max_gradient_steps)", [{k: float(curve[k][i, min(T, S) - 1]) for k in keys}
                                                 for i in range(len(r["seeds"]))])]
    for n in args.steps:
        rows.append((f"step {n}", [{k: float(curve[k][i, min(n, S) - 1]) for k in keys} for i in range(len(r["seeds"]))]))
    fixed = [dataset.metric(r["pred"][i], r["labels"]) for i in range(len(r["seeds"]))]    # recomputed (8.2)
    rows.append(("fixed point", fixed))
    width = max(len(name) for name, _ in rows)
    for name, values in rows:
        lines.append(f"  {name:<{width}}  " + "   ".join(f"{k} {mean_std([v[k] for v in values])}" for k in keys))
    stc = r["steps_to_converge"].float()
    conv = [100 * float((s >= 0).float().mean()) for s in stc]
    med = [float(s[s >= 0].median()) if (s >= 0).any() else float("nan") for s in stc]
    lines.append(f"  converged (local dist < {r['fwd_tol']:g}): {mean_std(conv)} %, median steps {mean_std(med)}"
                 + (f"   (curve beyond step {S} = value at step {S})" if any(n > S for n in args.steps) else ""))

    text = "\n".join(lines)
    print(text)
    out = folder.parent.parent / "metrics" / f"{folder.name}.txt"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text + "\n", encoding="utf-8")
    print(f"written: {out}")


if __name__ == "__main__":
    sys.exit(main())
