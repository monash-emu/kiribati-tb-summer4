# K4b — Fast, checkpointed calibration that recovers the posterior

Follow-up to K4 (`plans/kiribati-tb-summer4-port.plan.md`). Branch `feat/k4b-fast-calibration`,
stacked on `feat/k6-analysis`.

## Goal (the user's words, 2026-10-01)

"Rather than matching the calibration algorithm closely (we really don't care about that!),
focus instead on matching results; we want to recover a similar posterior (but quickly and
reliably). Feel free to use gradient based methods instead."

The posterior is fixed: same priors, targets and likelihood as the original (parity at the
golden parameters to 1e-6). The sampler is free. Success is measured, not argued:

1. A **reference posterior** with split R-hat ≤ 1.01 and bulk and tail ESS well above 400 for
   every parameter.
2. A **fast pipeline** whose posterior agrees with the reference within Monte Carlo error, per
   parameter quantile and in the projected averted burden, with its own diagnostics passing.
3. The **published posterior** (`reference/published/idata.nc`, unconverged DEMetropolisZ) as a
   consistency check whose error is set by its ESS.
4. The pipeline runs unattended on MASSIVE (16-task grid, 3 sensitivity analyses), resumes after
   a kill, and records its diagnostics in `details.yaml`.

## Pipeline (`src/kiribati_tb/pipeline.py`)

Built from summer4's workflow pieces where they fit, with port-side pieces where they do not
(each recorded in `docs/summer4-workarounds.md`):

1. `wf.lhs` + `wf.evaluate`: a 256-point design over the priors.
2. `lbfgs`: L-BFGS-B from the best 24, one host loop per start, 8 threads (W7). Where the
   reverse-mode gradient is NaN, a forward-mode gradient (S5, W6).
3. `distinct_optima`: report every separate optimum (multimodality check).
4. `laplace_covariance`: the Laplace covariance at the best optimum (tight solver), used as the
   initial NUTS metric (S7).
5. `RidgeShear`: a unit-Jacobian change of coordinates that straightens the
   `raw_transmission_rate` / `infection_pop_scale` ridge (W9).
6. `wf.warmup_until` + `wf.sample_until` with dense-mass NUTS, chains one per core
   (`chain_method="parallel"`), the stop callable a `Checkpoint` that saves draws and sampler
   state every chunk and stops on R-hat ≤ 1.01 and bulk and tail ESS ≥ 400 (W8).
7. `calibrate` is the one-call convenience; `run_full_analysis` and the cluster drivers use it.

Alternatives measured against the same diagnostics: Laplace + PSIS, and AIES with the
differential-evolution move.

## Checks

- `tests/test_pipeline.py`: the pieces on toy problems (checkpoint and resume, L-BFGS, Laplace,
  the shear's unit Jacobian, mode grouping, diagnostics).
- `notebooks/06-fast-calibration.ipynb`: the user gate — gradient costs, posterior geometry,
  diagnostics, marginals and quantiles (fast vs reference vs published vs the earlier AIES demo),
  projections against the paper's headline numbers, alternatives, cost.

## Decisions for the user

- Whether the paper's text (likelihood **sum** over notification years, §9.2) or the code
  (**mean**, estival 0.6, which produced the published posterior) is right. The port keeps the
  mean; `--aggregate sum` runs the other.
- The reference run's length (ESS target) and whether to run it on the cluster.
