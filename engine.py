"""Shared run logic for train.py and inference.py (Phase C.5, Task 1.2). Library only, not run directly.

- run_inference(model, batches, dataset, loop="fixed_steps", converge=None, record=False): the only inference loop
    - train.py: periodic test (ID, OOD), record=False (Task 5.5); on test_id of an unroll run also converge=(steps, tol)
      (Task 6): one pass of `steps` steps, loss / metric read at max_gradient_steps, convergence checked up to `steps`
    - inference.py: record=True (Task 7, 8): always to the fixed point; per-step local dist, global dist, E for every
      sample and the split metric at every step (max_gradient_steps and the fixed point are two points on that
      curve), plus predictions, labels, steps to converge, E(Z*), optionally Z*
- test_loop(model): the loop used to test during training (Task 5.5): backward 'unroll' -> 'fixed_steps'
  (max_gradient_steps steps, as in training); backward 'ift' -> 'fixed_point' (to fwd_tol, at most fwd_max_steps);
  the cross cases (unroll checkpoint at fixed_point, ift checkpoint at fixed steps) are for eval.py
- task metrics come from each dataset's metric() (data/), not from here
"""
import torch


def test_loop(model):
    """'fixed_steps' for a model trained with unroll, 'fixed_point' for one trained with ift."""
    return {"unroll": "fixed_steps", "ift": "fixed_point"}[model.config.backward]


def to_device(batch, device, dtype):
    """Move a batch {"inputs", "labels"} to device; float tensors are cast to dtype, integer tensors stay (Sudoku)."""
    return {k: v.to(device, dtype) if v.is_floating_point() else v.to(device) for k, v in batch.items()}


@torch.no_grad()
def record_batch(model, inputs, labels, dataset):
    """One batch, step by step to the fixed point, recording what inference.py plots and saves (Task 7, 8).

    - as EnergyLoopModel.solve_fixed_point: a sample stops (frozen) once local dist < fwd_tol, the loop ends when all
      stopped or after fwd_max_steps; fixed steps (max_gradient_steps) is a point on the same curve
    - returns outputs (at the last step) and, per sample and step t = 1..S: local ||Z_t - Z_{t-1}||_F, glob
      ||Z_t - Z_1||_F, energy E(Z_t), scores {key: (B, S)} = dataset.sample_scores of the prediction decoded from Z_t;
      steps_to_converge (B,) first t with local dist < fwd_tol (-1: never); z (B, N, d) the last Z
    """
    c = model.config
    x_tilde = model.core.add_context_pe(model.encoder(inputs))
    z = model.init_state(inputs.shape[0])
    conv = torch.full((z.shape[0],), -1, dtype=torch.long, device=z.device)
    local, glob, energy, scores, z1 = [], [], [], [], None
    for t in range(1, c.fwd_max_steps + 1):
        z_next = model.step(z, x_tilde)
        z_next = torch.where((conv < 0).view(-1, 1, 1), z_next, z)
        d = (z_next - z).flatten(1).norm(dim=-1)
        z = z_next
        z1 = z if z1 is None else z1
        conv = torch.where((conv < 0) & (d < c.fwd_tol), torch.full_like(conv, t), conv)
        local.append(d)
        glob.append((z - z1).flatten(1).norm(dim=-1))
        energy.append(model.core.get_energy(z, x_tilde))
        scores.append(dataset.sample_scores(model.predict(model.decoder(z)), labels))
        if bool((conv >= 0).all()):
            break
    return {"outputs": model.decoder(z), "local": torch.stack(local, 1), "glob": torch.stack(glob, 1),
            "energy": torch.stack(energy, 1), "scores": {k: torch.stack([s[k] for s in scores], 1) for k in scores[0]},
            "steps_to_converge": conv, "z": z}


def pad_steps(x, length, fill_last):
    """(B, S) -> (B, length): pad with the last value (frozen state: global dist, E) or with 0 (local dist)."""
    if x.shape[1] == length:
        return x
    pad = x[:, -1:].expand(-1, length - x.shape[1]) if fill_last else x.new_zeros(x.shape[0], length - x.shape[1])
    return torch.cat([x, pad], 1)


@torch.no_grad()
def run_inference(model, batches, dataset, loop="fixed_steps", converge=None, record=False, save_z=False):
    """Run the model without grad on batches of one split; returns the task loss and the metric of the whole split.

    - batches: iterable of {"inputs", "labels"} (a DataLoader)
    - loop: 'fixed_steps' | 'fixed_point' (EnergyLoopModel.solve)
    - converge: None, or (steps, tol) with loop 'fixed_steps' (Task 6): one pass of `steps` (>= max_gradient_steps)
      steps from the same Z0; outputs, loss, metric are taken at step max_gradient_steps (as without converge);
      a sample converges if its local dist ||Z_{t+1} - Z_t||_F < tol at some step t <= steps
    - returns dict:
        loss       float, task loss (model.compute_loss, no tie term), mean over samples
        metric     dict, dataset.metric on the predictions of the whole split (not a mean of batch metrics)
        converged  float, fraction of samples that converged: with converge, local dist < tol within `steps` steps;
                   without, steps >= 0 of solve (local dist < fwd_tol within the loop)
        n          number of samples
    - record=True (inference.py, Task 7, 8): step by step to the fixed point with record_batch (loop is ignored; the
      result at max_gradient_steps is read on the same curve); also returns pred, labels (N, ...) at the fixed point,
      steps_to_converge, final_local (local dist at the step of convergence, else at the last step), energy_star (N,),
      local, glob, energy (N, S), metric_curve {key: (S,)} the split metric at every step (from sample_scores),
      metric_at_T (metric at step max_gradient_steps), z_star (N, N_tok, d) if save_z; batches are padded to the
      longest (local with 0, the others with the last value: a stopped sample stays where it is)
    """
    if converge is not None and (loop != "fixed_steps" or record):
        raise ValueError("converge is for loop 'fixed_steps' without record")
    model.eval()
    p = next(model.parameters())
    T = model.config.max_gradient_steps
    loss_sum, n, n_conv, preds, labels = 0.0, 0, 0, [], []
    rec = {k: [] for k in ("steps_to_converge", "local", "glob", "energy", "scores", "z")}
    for batch in batches:
        batch = to_device(batch, p.device, p.dtype)
        b = batch["inputs"].shape[0]
        if record:
            r = record_batch(model, batch["inputs"], batch["labels"], dataset)
            outputs, conv = r["outputs"], r["steps_to_converge"] >= 0
            for k in rec:
                if k == "scores":
                    rec[k].append({key: v.cpu() for key, v in r[k].items()})
                elif k != "z" or save_z:
                    rec[k].append(r[k].cpu())
        elif converge is None:
            out = model(batch["inputs"], batch["labels"], loop=loop)
            outputs, conv = out["outputs"], out["converged"]
        else:
            n_max, tol = converge
            if n_max < T:
                raise ValueError(f"converge steps {n_max} < max_gradient_steps {T}")
            x_tilde = model.core.add_context_pe(model.encoder(batch["inputs"]))
            z = model.init_state(b)
            conv = torch.zeros(b, dtype=torch.bool, device=z.device)
            for t in range(1, n_max + 1):
                z_next = model.step(z, x_tilde)
                conv |= (z_next - z).flatten(1).norm(dim=-1) < tol
                z = z_next
                if t == T:
                    outputs = model.decoder(z)                     # what the test reports: step max_gradient_steps
        loss_sum += model.compute_loss(outputs, batch["labels"]).item() * b
        n += b
        n_conv += int(conv.sum())
        preds.append(model.predict(outputs).cpu())
        labels.append(batch["labels"].cpu())
    pred, lab = torch.cat(preds), torch.cat(labels)
    out = {"loss": loss_sum / n, "metric": dataset.metric(pred, lab), "converged": n_conv / n, "n": n}
    if record:
        S = max(x.shape[1] for x in rec["local"])
        local = torch.cat([pad_steps(x, S, False) for x in rec["local"]])
        glob = torch.cat([pad_steps(x, S, True) for x in rec["glob"]])
        energy = torch.cat([pad_steps(x, S, True) for x in rec["energy"]])
        scores = {k: torch.cat([pad_steps(sc[k], S, True) for sc in rec["scores"]]) for k in rec["scores"][0]}
        curve = [dataset.metric_from_scores({k: v[:, t] for k, v in scores.items()}) for t in range(S)]
        stc = torch.cat(rec["steps_to_converge"])
        last = torch.where(stc > 0, stc - 1, torch.full_like(stc, S - 1))     # local dist at convergence, else last step
        out.update({"pred": pred, "labels": lab, "steps_to_converge": stc,
                    "final_local": local.gather(1, last.view(-1, 1)).squeeze(1), "energy_star": energy[:, -1],
                    "local": local, "glob": glob, "energy": energy,
                    "metric_curve": {k: torch.tensor([c[k] for c in curve], dtype=torch.float64) for k in curve[0]},
                    "metric_at_T": curve[min(T, S) - 1]})
        if save_z:
            out["z_star"] = torch.cat(rec["z"])
    return out
