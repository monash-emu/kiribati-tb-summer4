# Reference runs: arms, measured costs, cluster resources

Four long calibrations of the base case, one per sampler, defined in
`src/kiribati_tb/reference.py` and launched with `scripts/cluster/submit_reference.sh` (see the
README, *Running the reference calibrations on MASSIVE*). All share the front end (256-point
design, L-BFGS from the best 24, Laplace metric at the best optimum) and the stop rule (split
R-hat ≤ 1.01 and bulk and tail ESS ≥ 400 for every parameter); all checkpoint every warmup
round and sampling chunk and stop between them before `--max-hours` (46 h in a 48 h job), so a
resubmission continues where the job stopped.

## Arms

| arm | kernel | chains | cpus | warmup | chunk | notes |
| --- | --- | --- | --- | --- | --- | --- |
| `nuts_td8` | NUTS, dense mass, `max_tree_depth=8`, ridge-sheared coordinates | 8, `parallel` | 8 | rounds 100, 100, 200, 200, 400 (stop early once `wf.WarmupRule` passes) | 100 | Laplace metric start; each round starts from the last round's pooled metric |
| `nuts_td5` | as `nuts_td8`, `max_tree_depth=5` | 8, `parallel` | 8 | as above | 100 | at most 31 gradients per iteration |
| `sa` | numpyro `SA` | 8, `parallel` | 8 | 20,000 | 10,000, thinned ×10 | gradient-free; SA cannot run seeded `vectorized` chains (S9) |
| `ess` | numpyro `ESS`, differential move | 64 walkers, `vectorized` | 2 | 300 | 100 | gradient-free; walkers drawn from the Laplace approximation; R-hat treats walkers as chains (optimistic) |

The NUTS arms use the ridge shear. Evidence that it helps is weak: in 20–40-iteration
single-chain warmups the sheared and plain kernels needed similar trajectory lengths (mean
~70–90 leapfrog steps), and in the first 50-iteration round of the 4-chain pilots the sheared
run iterated faster than the plain one. It is exact (unit Jacobian), so it cannot bias the
posterior; the comparison between `nuts_td8` and the gradient-free arms will show if it matters.

## Measured costs (this development machine)

The machine is shared and was heavily loaded (load average 15–60 on 10 cores) during every
measurement below, so per-iteration times are upper bounds; "unloaded" rows scale the warm
single-call timings of `scripts/bench_gradients.py` (load ≈ 10).

| quantity | value | source |
| --- | --- | --- |
| log density, one call | 0.047 s | `outputs/bench/gradients.json` |
| log density, vmapped batch of 16 | 0.020 s per point | `vmap` timing |
| reverse-mode gradient (checkpointed adjoint) | 0.175 s (3.7 evaluations) | `gradients.json` |
| forward-mode gradient (`jacfwd`, `ForwardMode`) | 0.257 s | `gradients.json` |
| `DirectAdjoint` gradient | 0.448 s | `gradients.json` |
| NUTS depth 8, pilot round 1 (4 parallel chains, load 15–25) | 64 leapfrog steps per iteration (second half of round), ~43 s per iteration per chain, 21% divergences while adapting | `outputs/logs/pilot_shear.log` |
| NUTS depth 8, unloaded estimate | ~11 s per iteration per chain at 64 steps; 45 s at the 255-step cap | 64 × 0.175 s |
| NUTS depth 5, unloaded estimate | ≤ 5.4 s per iteration per chain (≤ 31 steps) | 31 × 0.175 s |
| SA, one chain (load 64) | 0.072 s per iteration | `outputs/bench/kernels.json` |
| ESS, 64 walkers (load ~17–30) | ~63 s per iteration (~1 s per walker-iteration), ~1.4 cores used | `kernels.json`, `ps` |

## Wall-time expectations per arm on MASSIVE

The posterior's ESS per iteration is not yet known for any arm (no run has reached it locally);
these are the costs of the stages that can be predicted, assuming a MASSIVE core about as fast
as an unloaded core here. Multiply by your node's slowdown.

| arm | setup (design, L-BFGS, Laplace) | warmup | per 100-iteration chunk | 46 h buys (per chain) |
| --- | --- | --- | --- | --- |
| `nuts_td8` | ~10 min | 1–3 h (up to ~12 h if trees hit the cap) | ~20 min | ~8,000–13,000 iterations |
| `nuts_td5` | ~10 min | 0.5–1.5 h | ≤ 10 min | ~25,000 iterations |
| `sa` | ~10 min | ~20 min | 10,000 iterations ≈ 10 min | ~2–3 million iterations |
| `ess` | ~10 min | ~5 h | ~1.7 h | ~2,500 iterations × 64 walkers |

For comparison, the published DEMetropolisZ run took 72,130 s on 8 cores (160 CPU hours) and
reached a smallest bulk ESS of 35 (0.22 effective draws per CPU hour, `compare_reference.py`).
