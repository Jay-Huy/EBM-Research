"""Config of a run (Phase C.5, Task 2): read config/<task>.yaml and apply command-line overrides.

- one yaml per task holds everything that stays the same between runs (model sizes, step_size, T, optimizer, ...)
- what changes per run is set on the command line with --set, dotted keys:
    python train.py --config config/inverse.yaml --set model.mode=forced model.backward=ift
    test run without W&B:  --set wandb.mode=disabled
- --set only overrides keys that exist in the yaml (a typo is an error, not a silent new key)
- values of --set are parsed as yaml scalars: 6 -> int, 0.75 -> float, 1e-4 -> float, true -> bool, null -> None
- no seed in the config: the training seed is free; pick_seed() draws one per run and it is logged (W&B, config.yaml)
- run name and folder (Task 3.2): W&B run "{mode}-{backward}-{n_heads}h" in the task's project; local folder
  results/{task}-{mode}-{backward}-{n_heads}h, with -2, -3, ... when the folder already exists

Example:
    cfg = load_config("config/inverse.yaml", ["model.mode=forced"])
    run_dir = make_run_dir(cfg)
    save_config(cfg, run_dir / "config.yaml")
"""
import argparse
import copy
import random
from pathlib import Path

import yaml

from utils.logger import REPO_ROOT

RESULTS_DIR = REPO_ROOT / "results"


def _parse_value(text):
    """A yaml scalar; strings that look like numbers (1e-4) become floats."""
    value = yaml.safe_load(text)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return value
    return value


def apply_overrides(cfg, sets):
    """Apply ["a.b=value", ...] to a nested dict (in place). Only existing keys can be overridden."""
    for item in sets or []:
        if "=" not in item:
            raise ValueError(f"--set {item!r}: expected key=value")
        key, text = item.split("=", 1)
        parts = key.strip().split(".")
        node = cfg
        for p in parts[:-1]:
            if not isinstance(node, dict) or p not in node:
                raise KeyError(f"--set {key}: no key '{p}' in the config")
            node = node[p]
        if not isinstance(node, dict) or parts[-1] not in node:
            raise KeyError(f"--set {key}: no key '{parts[-1]}' in the config")
        node[parts[-1]] = _parse_value(text)
    return cfg


def load_config(path, sets=None):
    """Read a yaml config and apply --set overrides; returns a nested dict (with 'config_file' added)."""
    path = Path(path)
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cfg = apply_overrides(cfg, sets)
    cfg["config_file"] = str(path)
    return cfg


def parse_args(description=""):
    """Common command line: --config path, --set key=value ... ; returns argparse namespace."""
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--config", required=True, help="config/<task>.yaml")
    parser.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE",
                        help="override config keys, e.g. model.mode=forced model.backward=ift wandb.mode=disabled")
    return parser.parse_args()


def save_config(cfg, path):
    """Write the full config of a run (after overrides) as yaml."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)


def pick_seed():
    """A fresh training seed for this run (training seeds are free; the seed is logged, not fixed)."""
    return random.SystemRandom().randrange(2**31)


def run_name(cfg):
    """W&B run name inside the task's project: '{mode}-{backward}-{n_heads}h', e.g. 'et-unroll-1h';
    '-hard' appended with model.tie_mode 'hard' (no W_v, W_o), e.g. 'et-unroll-1h-hard';
    '-mu{train.mu_penalty}' appended when the mu penalty is on, e.g. 'et-unroll-1h-mu10';
    '-xraw' appended with model.context_norm false (v2: X + P_X not normalised), e.g. 'et-unroll-1h-xraw'."""
    m, mu = cfg["model"], cfg.get("train", {}).get("mu_penalty", 0)
    return (f"{m['mode']}-{m['backward']}-{m['n_heads']}h" + ("-hard" if m.get("tie_mode") == "hard" else "")
            + (f"-mu{mu:g}" if mu else "") + ("-xraw" if m.get("context_norm", True) is False else ""))


def make_run_dir(cfg, root=RESULTS_DIR):
    """Create results/{task}-{mode}-{backward}-{n_heads}h (or -2, -3, ... if taken) and return its path."""
    base = f"{cfg['task']}-{run_name(cfg)}"
    root = Path(root)
    path, k = root / base, 1
    while path.exists():
        k += 1
        path = root / f"{base}-{k}"
    path.mkdir(parents=True)
    return path


def flat(cfg, prefix=""):
    """{'a': {'b': 1}} -> {'a.b': 1}, for logging."""
    out = {}
    for k, v in cfg.items():
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            out.update(flat(v, key + "."))
        else:
            out[key] = copy.deepcopy(v)
    return out
