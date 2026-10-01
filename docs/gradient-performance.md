# Gradient performance of the calibration log density

What one gradient of the Kiribati calibration log density costs, what dominates it, and which
levers move it. Every number here comes from a script under `scripts/bench/` (run with
`pixi run bench-*`; results land in `outputs/bench/*.json`, which is not committed).

**Measurement conditions.** Apple M4 (4 performance + 6 efficiency cores), shared with other
jobs: the load average was 20–30 throughout, so absolute wall times are 2–5x those of an idle
machine (an idle-machine baseline from `scripts/bench_gradients.py`: eval 0.047 s, gradient
0.175 s). Comparisons are therefore made **within one process, interleaved** (one call of
each variant per round, rotating order, 12–30 rounds), and reported as median wall time with
the interquartile range, plus the median process CPU time (less sensitive to other
processes). Treat ratios, step counts and errors as the results, and absolute times as
indicative. Accuracy is measured against a reference gradient from Dopri5 at
`rtol = atol = 1e-10` with the mixing jumps given to the controller, itself checked against
central finite differences of a 1e-10 log density at the MAP (relative error 6.5e-9).

## Summary

- **The solve is stability-limited, not tolerance-limited.** The fastest Jacobian eigenvalue
  is about -10 per year at the MAP (-8 to -10 over posterior draws, -20 at the prior's upper
  corner). Dopri5 cannot step past about 0.31 years at any tolerance: 567 accepted steps at
  `rtol = 1.4e-4` against 592 predicted by the stability limit, and the same ~700 steps at
  `1e-3`. Diffrax's default I-controller oscillates at that edge and rejects 22% of steps.
- **The gradient's error is dominated by the yearly mixing jumps**, which the original's solver
  steps across: median 0.7%, worst 3.6% relative error over seven posterior points (0.3%
  in the code comment was optimistic). Giving the controller the jumps (`jump_ts`) cuts it
  10–25x at no cost in steps.
- **Recommended and applied:** Bosh3 + PI controller (`pcoeff=0.4, icoeff=0.3`) + `jump_ts` at
  the integer years + `RecursiveCheckpointAdjoint(checkpoints=1024)`, at the same tolerance
  (`kiribati_tb.calibration.CALIBRATION_SOLVER`). Gradient **1.4x faster** (wall, 30
  interleaved rounds; 1.3x CPU), gradient error over 25 points **0.05% median / 0.5% worst**
  against 0.33% / 8.6%, and reverse-mode gradients **finite at all 256 prior design points and
  25 posterior draws**, where the original's are NaN at 43 and 1 (S5).
- **Parallelism is the next big lever, not solver tuning.** One chain cannot use more than one
  core (one thread is as fast as many). Independent chains on separate XLA host devices scale
  almost linearly (4 devices: 3.7x gradients per second); `vmap` across chains in one device
  saturates at about 2x.
- **What did not help:** implicit solvers (Kvaerno3/5: 15–30x slower, less accurate),
  lower-order 2-stage methods (Heun, Ralston: twice the CPU), forward mode (2x CPU), other
  XLA CPU runtime flags, saving fewer time points, and a loop-free interpolation knot search
  (prototyped in summer4; 1–5%). float32 gives 1.2x but costs as much gradient accuracy as the
  solver change gains; not applied.
- **NUTS cost is geometry.** With the Laplace metric, pilot NUTS adapts step sizes of
  0.05–0.17 and needs about 64 leapfrog steps per iteration. Along leapfrog trajectories at
  step size 0.1 the energy error is about 1.2 whichever solver computes the gradient (a 1e-8
  solve included); the original solver adds at most 0.1 to it and the new one 0.01. At 0.2
  every solver's trajectories blow up. Cheaper gradients scale the cost of every
  iteration; they do not change how many gradients an iteration needs (reparameterisation is
  the NeuTra work).

## 1. Profile of one gradient (original configuration)

`pixi run bench-profile --solver original` (`outputs/bench/profile_original_run0.json`; MAP,
7 rounds, loaded machine).

| piece | time | note |
|---|---|---|
| log density (eval) | 0.131 s | idle machine: 0.047 s |
| `value_and_grad` | 0.585 s | 4.5x eval (3.7x on an idle machine) |
| solve, 186 yearly saves | 0.124 s | 95% of eval |
| solve, one save | 0.115 s | saving costs ~7% of eval |
| outputs + likelihood + priors | ~0.007 s | eval less solve (5%) |
| run-start stage (`prepare`: mixing stack, hoisted rates) | 0.0035 s | its gradient 0.0053 s |
| one vector-field call | 25–31 µs | `lax.scan` of 1,000 calls; VJP 24 µs |
| steps | 567 accepted + 156 rejected | 6 new vector-field calls per step (Dopri5, FSAL) |

- **Vector-field calls are the forward solve.** 723 steps x 6 calls x ~26 µs accounts for all
  of the solve; per-step solver overhead (error norm, controller, save checks) is not visible
  above that.
- **The vector field is many tiny kernels.** Its jaxpr is 1,785 equations (VJP 3,042); the
  compiled XLA program is ~114 kernels on 160-element arrays (68 fusions, 9 gathers, scatters,
  copies) plus three `while` loops (the `searchsorted` binary searches of the death-rate,
  births and treatment-success interpolations). Inside the solve, diffrax runs the six
  Runge-Kutta stages as a loop whose body is 110 kernels. The work per kernel is negligible;
  the count of kernels, steps and stages sets the cost.
- **The reverse pass is 3.5x the forward.** Per step it re-linearises the step (one forward
  step) and runs six VJPs of the vector field; with diffrax's default checkpoint count
  (`floor(sqrt(2 * max_steps))` = 90 for 723 steps) it also recomputes stretches of the
  forward solve. 1,024 checkpoints remove the recomputation (section 2).
- **Compile:** gradient trace + lower 7.0 s, XLA compile 8.4 s (loaded); the full gradient
  jaxpr is 23,728 equations.

## 2. Levers: measured

`pixi run bench-levers` (`outputs/bench/levers_all.json`, 12 rounds) and a 30-round
confirmation of the leading variants (`levers_confirm.json`). Gradient time is the median wall
time at the MAP with its interquartile range; speedups are against `base` in the same run
(wall / CPU). Steps are at the MAP (median over seven points in brackets). Errors are over the
MAP and six published-posterior draws.

| variant | grad (ms) | speedup wall / CPU | steps acc + rej | grad error median / max | log-density error median / max |
|---|---|---|---|---|---|
| **base**: Dopri5, I-controller, 1.4e-4, default adjoint | 784 [536–1090] | 1.00 / 1.00 | 567 + 156 (486 + 155) | 0.73% / 3.6% | 2.8e-3 / 5.5e-3 |
| `checkpoints=1024` | 592 | 1.32 / 0.98 | same | same | same |
| ForwardMode (`jacfwd`, 19 tangents) | 722 | 1.09 / 0.49 | same | same | same |
| `jump_ts` (integer years) | 802 | 0.98 / 0.94 | 637 + 88 | 0.12% / 0.35% | 2.4e-4 / 3.2e-4 |
| PI controller 0.4/0.3 | 707 | 1.11 / 1.13 | 582 + 50 | 0.34% / 2.2% | 1.6e-3 / 3.1e-3 |
| PI + `jump_ts` | 885 | 0.89 / 0.92 | 705 + 35 | 0.027% / 0.16% | 6.2e-5 / 2.8e-4 |
| PI + `jump_ts` + `checkpoints=1024` | 692 | 1.13 / 0.99 | 705 + 35 | 0.027% / 0.16% | 6.2e-5 / 2.8e-4 |
| `dtmax=1` + `jump_ts` | 724 | 1.08 / 0.93 | 637 + 88 | (as `jump_ts`) | |
| tolerance 1e-3, PI + `jump_ts` | 698 | 1.12 / 0.97 | 698 + 15 | 0.083% / 0.42% | 2.8e-4 / 8.7e-4 |
| tolerance 1e-5, PI + `jump_ts` | 930 | 0.84 / 0.74 | 887 + 61 | 0.007% / 0.14% | 2.2e-5 / 6.8e-5 |
| tolerance 1e-6, PI + `jump_ts` | 1193 | 0.66 / 0.59 | 1150 + 74 | 0.0007% / 0.005% | 3e-7 / 5e-6 |
| Tsit5, PI + `jump_ts` | 746 | 1.05 / 0.95 | 725 + 37 | 0.046% / 0.71% | 3.2e-4 / 6.0e-4 |
| **Bosh3**, PI + `jump_ts` | 499 | 1.57 / 1.19 | 905 + 94 | 0.12% / 0.29% | 1.4e-4 / 1.9e-3 |
| **Bosh3**, PI + `jump_ts`, `checkpoints=1024` | 431 | **1.82 / 1.32** | 905 + 94 | 0.12% / 0.29% | 1.4e-4 / 1.9e-3 |
| Heun, PI + `jump_ts` | 678 | 1.16 / 0.46 | 1815 + 96 | 0.047% / 0.63% | 4.1e-4 / 5.8e-4 |
| Ralston, PI + `jump_ts` | 700 | 1.12 / 0.46 | 1802 + 85 | 0.015% / 0.023% | 3.7e-5 / 3.2e-4 |
| Kvaerno5 (implicit), `jump_ts` | 11987 | 0.07 / 0.03 | 281 + 6 | 2.2% / 15% | 7.1e-3 / 2.3e-2 |
| Kvaerno3 (implicit), `jump_ts` | 20702 | 0.04 / 0.02 | 431 + 188 | 1.1% / 9.7% | 2.0e-3 / 7.6e-3 |
| constant Dopri5, dt = 0.25 | 782 | 1.00 / 0.81 | 740 + 0 | 2.0% / 5.1% | 1.9e-3 / 9.2e-3 |
| constant Bosh3, dt = 0.2 | 529 | 1.48 / 1.33 | 925 + 0 | 0.058% / 0.37% | 9.5e-5 / 2.2e-4 |
| save target years only (15 + 1 per year, not 186) | 852 | 0.92 / 0.90 | same as base | identical | identical |

30-round confirmation (`levers_confirm.json`; wall / CPU speedup against base in that run):
`checkpoints=1024` 1.19 / 1.07; PI + `jump_ts` + checkpoints 1.14 / 1.03; Bosh3 + PI + `jump_ts`
1.30 / 1.21; the same with checkpoints 1.44 / 1.30; plus target-year saves (`recommended`)
1.59 / 1.30; target-year saves alone 1.02 / 0.98.

Reading the table:

- **Tolerance barely changes cost** between 1e-3 and 1.4e-4 (stability-limited); below
  1.4e-4 it starts to (1e-5: +30% steps, 1e-6: +70%).
- **`jump_ts` is the accuracy lever.** It forces a step boundary at each year, which costs
  about one extra step per year where the stability-limited step does not divide the year,
  and buys 6–25x lower gradient error. With a PI controller the rejections that remain are
  few (35 of 740).
- **Bosh3 is the speed lever.** It needs 3 new vector-field calls per step against Dopri5's
  6; its smaller stability region (boundary 2.5 against 3.3) costs fewer extra steps (5 per
  year, 905 + 94) than it saves calls. Its error estimate is third-order, so at the same
  tolerance its gradient error is higher than PI + `jump_ts` Dopri5 (0.12% against 0.03%) but
  still 6x lower than the original's.
- **Checkpoints:** 1,024 checkpoints (more than any plausible solve's steps) remove the reverse
  pass's recomputation: 1.07–1.3x, identical gradients.
- **Constant steps** that divide a year need no `jump_ts` or error estimate, but fail when the
  eigenvalue exceeds the stability limit: constant Dopri5 at 0.25 failed at 90 of 256 prior
  design points. Not safe for a sampler.
- **Implicit solvers** take a third of the steps but each costs a 160x160 Jacobian and linear
  solves, and their gradients through the Newton iterations are less accurate at this
  tolerance.

## 3. Model-level levers

| lever | finding | source |
|---|---|---|
| hoisting `t`/`y`-independent work | Already done: the mixing stack (eigvalsh normalisation, 186 x 8 x 8) is built once in `prepare_fn` (3.5 ms, gradient 5 ms of a 585 ms gradient), and summer4 hoists 8 parameter-only rate subtrees. What remains in the vector field depends on `t` (death-rate, births and treatment-success interpolations, the tanh detection scale-up) or `y`. | `profile_original_run0.json` |
| piecewise-constant mixing lookup | Already one `Lookup` gather of an 8 x 8 slice per call; no loop, negligible. | HLO of the vector field |
| `searchsorted` loops in interpolation | Three XLA `while` loops per vector-field call. Replacing them with a loop-free search (`compare_all`) gives identical answers and 1.01–1.09x (wall) / 1.02–1.06x (CPU): small. Prototyped in summer4 (`perf/gradient-performance`). | `searchsorted_ab.json` |
| float32 | Gradient 1.26x wall / 1.20x CPU faster, eval 1.05x (in-process, 20 rounds, original solver). But float32 moves the gradient by 0.15% median / 0.46% worst and the log density by 1e-3 / 8e-3 nats against float64 at the same solver settings: as large as the original solver's error and 3–10x the new solver's. Not applied: the package enables x64 for parity with summer2gen, and the precision loss would undo the accuracy gain above. | `pixi run bench-float32` (`float32_ab.json`) |
| unneeded outputs | The calibration already evaluates only the 10 targets (`restrict_outputs`): outputs + likelihood are ~5% of an eval. Saving 31 target years instead of 186 is ~7% of an eval and not measurable on the gradient (0.92–1.12x over three runs). Not applied (it needs a private `BayesianModel` override; summer4 note `bayesian-model-save-grid.md`). | `levers_*.json` |
| unneeded compartments | None: every compartment feeds the targets' dynamics. | |
| XLA flags | One intra-op thread: same speed (gradient 0.376 s against 0.388 s). Legacy (non-thunk) CPU runtime: eval equal, gradient slower (0.54 s). Disabling the new fusion emitters: 2x slower. | `pixi run bench-ab --only default,single_thread,legacy_runtime,no_fusion_emitters` (`xla_flags.json`, 2 rounds x 10 reps) |

## 4. Sampler-facing: cost per gradient against gradients per iteration

- **Stiffness numbers** (`stiffness.json`): largest `|Re λ|` 10.6 at the MAP, 8.2–10.2 over
  six posterior draws, 19.8 at the prior's upper corner. The step-size limit therefore varies
  2x across the prior, which is why adaptive (not constant) steps are needed.
- **NaN gradients are solver-specific.** Over the 256-point prior design (`nan_rate_*.json`):
  Dopri5 (I, PI, PI + jumps) 43 non-finite reverse-mode gradients at finite log densities;
  Tsit5 27; Bosh3 0 (with or without checkpoints). Over 25 published-posterior draws, the
  original solver's gradient was non-finite at 1 (the original's "none of 96 draws" in S5 does not hold for every draw), Bosh3's at none (`before_after_accuracy.json`). A NaN
  gradient inside a NUTS trajectory is a divergence, so Bosh3 also removes a source of
  divergences.
- **Energy error** (`pixi run bench-energy --points 4 --steps 32 --eps 0.1,0.2`,
  `energy_error.json`). Four published-posterior starts, momenta for the Laplace metric, 32
  leapfrog steps, the same starts and momenta for each solver:

  | step size | solver | max \|ΔH\| | RMS ΔH | max \|ΔH − ΔH(1e-8)\| |
  |---|---|---|---|---|
  | 0.1 | original Dopri5 | 1.15 | 0.39 | 0.10 |
  | 0.1 | `CALIBRATION_SOLVER` | 1.18 | 0.40 | 0.010 |
  | 0.1 | Dopri5 1e-8 + jumps | 1.18 | 0.40 | – |
  | 0.2 | all three | 3e4 to 4e9 (trajectories diverge, some NaN) | | |

  The energy error that limits NUTS's step size comes from the posterior's curvature: a 1e-8
  solve has the same error, and the stable step lies between 0.1 and 0.2 for every solver
  (consistent with the 0.05–0.17 the pilot adapted). Solver error is a small addition (up to
  9% of it for the original, 1% for the new solver), so neither a tighter nor a looser solver
  will change the tree depth much.
- **Tree depth.** The pilot warmup (`outputs/calibrate/pilot_shear/nuts/warmup.csv`, Laplace
  metric) adapted step sizes 0.05–0.17, used 64 leapfrog steps per iteration on average, hit
  the maximum tree depth 6% of the time, and diverged 21% of the time. The Laplace covariance
  has condition number 7,900 (square root 89): the thin, curved ridge (notebook 06) sets the
  step size. Gradient cost multiplies every iteration; it does not set the number of
  gradients.

## 5. Parallelism

`pixi run bench-parallel` (`parallel.json`; load average 21–31 during the run, so these are
lower bounds).

| layout | time per call | gradients per call | per gradient | throughput vs one |
|---|---|---|---|---|
| one gradient | 0.87 s | 1 | 0.87 s | 1.0x |
| `vmap` 2 | 1.20 s | 2 | 0.60 s | 1.4x |
| `vmap` 4 | 1.88 s | 4 | 0.47 s | 1.8x |
| `vmap` 8 | 3.57 s | 8 | 0.45 s | 1.9x |
| `vmap` 16 | 6.54 s | 16 | 0.41 s | 2.1x |
| `pmap` over 1 device | 1.30 s | 1 | 1.30 s | |
| `pmap` over 2 devices | 0.89 s | 2 | 0.44 s | |
| `pmap` over 4 devices | 0.94 s | 4 | 0.24 s | ~3.7x |
| `pmap` x `vmap` 4, 4 devices | 2.04 s | 16 | 0.13 s | |

- A single gradient uses one core: with one intra-op thread it is as fast as with the default
  pool (`xla_flags.json`, from `bench-ab`).
- `vmap` across chains saturates at about 2x: every lane runs until the slowest lane's solve
  finishes, and the batched kernels (scatters in particular) grow with the batch.
- Separate host devices (`--xla_force_host_platform_device_count`, numpyro
  `chain_method="parallel"`) scale nearly linearly while there are free cores.
- So `--cpus-per-task=8` buys **8 chains at the speed of one**, not a faster chain: use
  `chain_method="parallel"` with one device per chain (as `scripts/calibrate.py` does), not
  more threads per chain.

## 6. Before and after

`pixi run bench-before-after` (`before_after_timing.json`: MAP, 30 interleaved rounds;
`before_after_accuracy.json`: MAP and 24 published-posterior draws).

| | before (original Dopri5) | after (`CALIBRATION_SOLVER`) |
|---|---|---|
| eval, median wall | 0.102 s | 0.079 s (1.29x; CPU 1.24x) |
| gradient, median wall | 1.03 s | 0.73 s (1.42x; p10 1.48x; CPU 1.28x) |
| idle-machine gradient | 0.175 s (measured) | ~0.12 s (expected at 1.4–1.5x) |
| steps, median / max | 666 / 1049 | 924 / 1281 |
| gradient error, median / p90 / max | 0.33% / 3.0% / 8.6% | 0.05% / 0.21% / 0.50% |
| log-density error, median / max | 1.6e-3 / 1.1e-2 | 2.0e-4 / 1.8e-3 |
| non-finite gradients (25 points) | 1 | 0 |

## Recommendations, ranked

1. **Run chains on separate devices** (`chain_method="parallel"`, one XLA host device per
   chain; already the pipeline's default): near-linear in cores. Expected: 4–8x wall per
   iteration for 4–8 chains against running them in series or vectorised.
2. **Calibrate with Bosh3 + PI + `jump_ts` + 1,024 checkpoints** (applied on this branch):
   1.4x per gradient (1.3–1.8x across runs), 6–17x more accurate gradients, no NaN gradients
   over the prior design or the posterior draws checked.
3. **Reparameterise the ridge** (NeuTra / shear, other branch): the only lever on gradients
   per iteration, which is where 64–255x of the per-iteration cost sits.
4. **summer4 follow-ups** (`perf/gradient-performance`, `futureplans/`): expose discontinuity
   times for `jump_ts`, document solver choice for stability-limited models, a better default
   checkpoint count, the loop-free knot search, a public calibration save grid. Each is
   1.0–1.3x or an accuracy/usability gain; none changes the picture above.
5. **Not worth pursuing here:** implicit solvers, 2-stage methods, forward mode, XLA runtime
   flags, constant steps. float32 (1.2x) only if gradient error of ~0.2% is acceptable.

## Not verified

- Absolute idle-machine timings of the new configuration (the machine was never idle); the
  expected idle gradient time is extrapolated from the measured ratio.
- Linux / cluster CPUs (MASSIVE): thread and device scaling were measured on an M4 with
  heterogeneous cores under load.
- That NUTS mixes better with the new solver (fewer divergences, different adapted step
  size): only the energy-error and NaN-rate proxies were measured, not a NUTS run.
- The posterior itself under the new solver: the target density moves by solver error only
  (log density within 1.8e-3 nats of the 1e-10 solve at every point checked, against 1.1e-2
  for the original), but no sampled posterior was compared.
- PI coefficients other than 0.4/0.3, Bosh3 at tolerances other than 1.4e-4, and a
  stabilised explicit (Runge-Kutta-Chebyshev) method, which would suit this stiffness profile
  but is not in diffrax.
- Where exactly Dopri5's reverse-mode NaN comes from (S5): only that it is solver-specific.
