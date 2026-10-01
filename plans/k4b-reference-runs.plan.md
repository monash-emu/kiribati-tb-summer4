# K4b (revised) — Reference calibrations on MASSIVE, four samplers

Supersedes the "reference posterior" and "fast pipeline" steps of
`plans/k4b-fast-calibration.plan.md` (kept as the record of the first approach). Same branch.

## Why the change (the user, 2026-10-01)

A converged NUTS reference is many hours on the shared, heavily loaded development machine
(measured: ~40 s per NUTS iteration per chain under load, ~64–136 leapfrog steps each). Instead
of a local reference, deliver a kit the user launches on MASSIVE: four independent long runs of
the base case, one per sampler, that can be compared at any point. NeuTra reparameterisation and
gradient performance are separate workstreams (other agents), not this branch.

## The four arms (`src/kiribati_tb/reference.py`)

| arm | kernel | chains | notes |
| --- | --- | --- | --- |
| `nuts_td8` | NUTS, dense mass, `max_tree_depth=8`, ridge-sheared coordinates | 8 parallel | Laplace metric start, staged warmup |
| `nuts_td5` | as `nuts_td8`, `max_tree_depth=5` | 8 parallel | cheaper iterations |
| `sa` | numpyro `SA` | 8 parallel | gradient-free, thinned ×10 |
| `ess` | numpyro `ESS`, differential move | 64 walkers, vectorised | gradient-free |

All share the front end (design, L-BFGS, Laplace metric) and the stop rule (split R-hat ≤ 1.01,
bulk and tail ESS ≥ 400), checkpoint every warmup round and sampling chunk, and stop before a
wall-clock deadline so a SLURM timeout never loses work; resubmitting resumes.

## Deliverables

- `scripts/reference_run.py` (one arm, `--smoke` for tiny settings),
  `scripts/cluster/reference_arm.sh` + `scripts/cluster/submit_reference.sh` (all or named arms),
  `scripts/cluster/smoke_reference.sh` (every arm, then a resume check), run locally.
- `scripts/compare_reference.py`: diagnostics, quantiles with MCSE, pairwise agreement, ESS per
  CPU hour, against the published posterior (draws 10,000–19,999).
- `notebooks/06-fast-calibration.ipynb` reads it; README "Running the reference calibrations on
  MASSIVE"; `docs/reference-runs.md` holds the per-arm cost measurements and resources.
- Tests: arm settings, every kernel's checkpoint and resume, the deadline, the launcher.
