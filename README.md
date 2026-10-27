# EBM Research — Loop Transformer as Energy Descent

```
EBM Research/
├── docs/                    # research notes (theory, proofs, decisions)
├── plans/                   # Phase C plan and progress
├── samples/                 # reference code (cloned repos, files)
├── data/                    # datasets, builders and task metrics
├── config/                  # one <task>.yaml per run setup: model, dataset, training
├── models/                  # energy model: ET and Forced
│   ├── blocks/              # branches: mlp, cross-attention, self-attention; heads.py (multi-head helpers)
│   ├── tasks/               # task models: encoder / decoder / loss per task (matrix, sudoku)
│   ├── positional_encoding.py
│   └── modeling_base.py     # core latent, base model, config, registry
├── utils/                   # shared utilities
│   ├── logger.py            # Logger for the whole system; TestLogger for tests
│   ├── config.py            # read config/<task>.yaml, command-line overrides
│   ├── checkpoint.py        # save / load checkpoints
│   └── wandb_utils.py       # Weights & Biases: init, offline, resume
├── tests/                   # tests for each Phase C task
│   └── logs/                # test logs
├── results/                 # one folder per run (not in git)
│   └── <run>/
│       ├── config.yaml      # full config of the run
│       ├── train.log
│       ├── checkpoints/     # latest_checkpoint.pt, best_checkpoint.pt, convergence_checkpoint.pt
│       ├── inference/       # outputs and plots of inference.py
│       └── metrics/         # results of eval.py
├── engine.py                # shared run logic (run_inference), used by train.py and inference.py
├── train.py                 # train a model; periodic test; checkpoints
├── inference.py             # run a trained model and save its outputs; plots to check convergence
├── eval.py                  # setup and results of one inference folder
└── README.md
```

## Setup

```
pip install pyyaml wandb
wandb login          # once per machine; the API key is never stored in the repo
```

## Configs

One file per task in `config/`; it holds everything that stays the same between runs
(model size, `step_size`, `max_gradient_steps`, optimizer, epochs / iterations, tie coefficients, W&B project):

| file | task | W&B project |
|---|---|---|
| `config/addition.yaml` | matrix addition | `ebm-addition` |
| `config/lowrank.yaml` | low-rank completion | `ebm-lowrank` |
| `config/inverse.yaml` | matrix inverse | `ebm-inverse` |
| `config/sudoku_rrn.yaml` | Sudoku (RRN) | `ebm-sudoku` |

Per run, set these keys on the command line with `--set` (dotted keys; only keys that exist in the yaml):

| key | values | default |
|---|---|---|
| `model.mode` | `et` \| `forced` | `et` |
| `model.backward` | `unroll` \| `ift` | `unroll` |
| `model.n_heads` | `1` \| `8` | `1` |
| `model.tie_mode` | `orbit` \| `hard` (ablation: no W_v, W_o, exact gradient of E) | `orbit` |
| `model.context_norm` | `true` \| `false` (v2: normalise X + P_X with RMSNorm before injection and cross-attention) | `true` |
| `train.clip_grad` | `false` \| `true` (clip the gradient norm at `train.clip_grad_norm`, 1.0) | `false` |
| `train.mu_penalty` | `0` (off) \| λ_μ > 0: add λ_μ·max(0, μ − `train.mu_max`)² to the loss (μ = Σ_h ‖W_q^h‖₂‖W_k^h‖₂, Theorem 1: μ < 1/2; `mu_max` 0.45) | `0` |

Any other key can be overridden the same way for a quick try, e.g. `--set train.lr=3e-4`.

## Train

```
python train.py --config config/<task>.yaml [--set key=value ...]
```

| example | command |
|---|---|
| new run | `python train.py --config config/sudoku_rrn.yaml --set model.mode=forced model.n_heads=8` |
| run B: IFT from the convergence checkpoint of an unroll run | `python train.py --config config/sudoku_rrn.yaml --set model.backward=ift --init_from results/sudoku_rrn-et-unroll-1h/checkpoints/convergence_checkpoint.pt` |
| continue an interrupted run (same W&B run) | `python train.py --resume results/sudoku_rrn-et-unroll-1h` |
| test run, nothing sent to W&B | add `wandb.mode=disabled` to `--set` (`offline`: keep locally, upload later with `wandb sync`) |

- W&B: run `{mode}-{backward}-{n_heads}h` in the task's project, with `-hard` appended for `model.tie_mode=hard` `-mu{λ_μ}` for `train.mu_penalty` > 0 (e.g. `et-unroll-1h-mu10`) and `-xraw` for `model.context_norm=false`.
- Run folder `results/{task}-{run name}`, e.g. `results/sudoku_rrn-et-unroll-1h` (`-2`, `-3`, … if taken):
  - `config.yaml`: full config with the seed
  - `train.log`: train / test lines, `% converged` of test_id, saved checkpoints
  - `checkpoints/latest_checkpoint.pt`: every epoch, for `--resume`
  - `checkpoints/best_checkpoint.pt`: best test_id metric (Sudoku board_acc, matrix tasks mse)
  - `checkpoints/convergence_checkpoint.pt`: unroll runs, once ≥ 99% of test_id has local dist < 1e-5 within 100 steps

## Inference

```
python inference.py --run results/<run> --split <split> [options]
```

Always runs to the fixed point (local dist < 1e-5); the result at `max_gradient_steps` is read on the same curve.

| option | default | meaning |
|---|---|---|
| `--checkpoint` | `best` | `best` \| `latest` \| `convergence` |
| `--n_z0 K` | 1 | number of random Z0 (seeds `--seed`, `--seed`+1, …) |
| `--save_z` | off | also save Z* |

Example: `python inference.py --run results/sudoku_rrn-et-unroll-1h --split test_id`

Output in `results/<run>/inference/<split>-<checkpoint>/`:
- `inference.pt`: predictions, labels, metric at `max_gradient_steps` and at the fixed point, metric per step; per sample steps to converge, E(Z*); per-step local dist, global dist, E
- `curves.png`: local dist (log), global dist, E and the metric per step; dashed line at `max_gradient_steps`
- `inference.log`

## Eval

```
python eval.py --inference results/<run>/inference/<split>-<checkpoint> [--steps N ...]
```

Prints the setup (mode, heads, unroll with its steps or ift, split, checkpoint) and the results (metric at
`max_gradient_steps`, at the fixed point and at each `--steps N`; % converged, median steps), and writes them to
`results/<run>/metrics/<split>-<checkpoint>.txt`.
