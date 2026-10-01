# kiribati-tb-summer4

A port of the Kiribati tuberculosis screening model
([`monash-emu/kiribati_tb_modelling`](https://github.com/monash-emu/kiribati_tb_modelling),
built on summer2gen with estival and pymc) to
[summer4](https://github.com/monash-emu/summer4), with calibration on numpyro.

## The study

The original model supports *From rollout to refinement: using early screening data to model
the next phase of population-wide tuberculosis screening in Kiribati*, which evaluates
population-wide TB screening in South Tarawa (the PEARL study) for 2026–2035: how much TB
incidence and mortality different screening algorithms (PEARL's CXR-Xpert, dropping Xpert,
dropping TST-guided preventive therapy) avert at 65%, 75% and 84.5% coverage.

The model has 10 TB states (from *M. tuberculosis*-naïve through incipient, contained and
cleared infection, four forms of disease by clinical status and infectiousness, treatment and
recovery) × 8 age bands × 2 reachability strata = 160 compartments, runs 1850–2035 with UN
demography and a time-varying age mixing matrix, and is calibrated to PEARL screening data and
historical notifications.

## The port

| Module | Ports | summer4 pieces |
| --- | --- | --- |
| `demography.py` | `tbh/demographic_tools.py` | `Data.table(...).interp("sigmoidal")` |
| `model.py` | `tbh/model.py` | `PropertyMap`, `TransitionFlow`/`EntryFlow`, `TraitChain.from_breakpoints`, `InitialPopulation`, `ForceOfInfection(kind=GENERALISED)`, rate trees |
| `mixing.py` | `tbh/age_mixing.py` | yearly stack in `compile(prepare_fn=...)`, `Lookup` in `MixingMatrix` |
| `outputs.py` | `tbh/outputs.py` | one `OutputSet` with every original name |
| `interventions.py`, `scenarios.py` | `tbh/interventions.py`, `data/scenarios.py` | screening flows as rate trees |
| `calibration.py` | calibration in `tbh/runner_tools.py` | `BayesianModel`, `priors_from_frame`, `NormalLikelihood` |
| `analysis.py` | full runs in `tbh/runner_tools.py` | `BayesianModel.posterior_runs` |

**Correct means numerical parity** with golden outputs produced by the original summer2gen
code (`tests/golden/`, regenerated only by `pixi run -e reference golden`): every compartment
and every one of the ~680 derived outputs matches to 1e-5 relative, for five fixtures including
two screening scenarios. Where summer4 had to be worked around, the list is in
`docs/summer4-workarounds.md`.

The plan is `plans/kiribati-tb-summer4-port.plan.md` (phases K0–K6).

## Reproducing

```bash
pixi run test                   # parity and unit tests against the committed goldens
pixi run test-slow              # slower checks (likelihood vs estival, gradients, notebooks)
pixi run -e reference golden    # regenerate goldens from the original model (~15 min)

pixi run find-map               # MAP fit with optax (replaces the nevergrad notebook)
pixi run synthetic-recovery     # K4 acceptance: recover known parameters
pixi run calibrate              # fast calibration + scenario full runs, original file formats
pixi run notebook               # the phase gate notebooks
```

## Calibration

The posterior is the original's (same priors, targets and likelihood); the sampler is not.
`pixi run calibrate` runs `kiribati_tb.pipeline.calibrate`: a Latin-hypercube design, L-BFGS
from its best points, a check for separate optima, a Laplace metric at the best one, then
dense-mass NUTS (in coordinates that straighten the transmission ridge) in checkpointed
chunks until split R-hat ≤ 1.01 and bulk and tail ESS ≥ 400 for every parameter. Every stage
writes to the output folder, so rerunning the same command after a kill resumes it;
`details.yaml` records the diagnostics. Whether the samplers recover the same posterior is
settled by four long reference runs on MASSIVE (next section), compared with each other and
with the published posterior (`reference/published/`) in `notebooks/06-fast-calibration.ipynb`.

summer4 is pinned to `main` at `c9548d5` (after `v0.2.0a5`): the solver backends of its step 29
let the port choose diffrax's adjoint (forward mode where reverse-mode gradients are NaN), and
step 30's run objects are what `wf.sample_until` and `wf.optimize` return.

## Running the reference calibrations on MASSIVE

Four long runs of the base case, one per sampler (`kiribati_tb.reference`): `nuts_td8` (dense
NUTS, tree depth 8), `nuts_td5` (tree depth 5), `sa` (Sample Adaptive MCMC) and `ess` (ensemble
slice sampling, 64 walkers). Each is one SLURM job; each stops when every parameter has split
R-hat ≤ 1.01 and bulk and tail ESS ≥ 400, or 2 h before its wall time, and resumes from its
checkpoint when resubmitted.

```bash
# 1. On a login node: get the branch and the environment (once).
git clone https://github.com/monash-emu/kiribati-tb-summer4.git
cd kiribati-tb-summer4
git checkout feat/k4b-fast-calibration
pixi install

# 2. Optional, minutes: check every arm end to end with tiny settings (and its resume).
scripts/cluster/smoke_reference.sh            # logs in outputs/logs/smoke_<arm>.log

# 3. Submit all four arms (or name some: scripts/cluster/submit_reference.sh nuts_td5 ess).
#    Either put the project in the template (replace <account> in
#    scripts/cluster/reference_arm.sh; the original analysis used sh30) or pass it:
ACCOUNT=<account> scripts/cluster/submit_reference.sh

# 4. Watch: progress bars, one [warmup] line per warmup round, one [checkpoint] line per chunk.
squeue -u $USER
tail -f outputs/cluster/logs/ref_nuts_td8_<jobid>.out
cat outputs/reference/nuts_td8/mcmc/diagnostics.json     # latest R-hat / ESS per parameter

# 5. A job that stopped for its deadline (or was killed) continues where it left off:
ACCOUNT=<account> scripts/cluster/submit_reference.sh nuts_td8

# 5b. A fresh, separate set of runs (new folders, logs and job names) beside earlier ones:
RUN=r2 ACCOUNT=<account> scripts/cluster/submit_reference.sh   # outputs/reference/r2/<arm>/
# The comparison scripts and notebook 06 read a named set the same way:
RUN=r2 pixi run python scripts/compare_reference.py      # -> outputs/compare_reference/r2/
RUN=r2 pixi run python scripts/compare_posteriors.py     # -> outputs/compare/r2/
RUN=r2 pixi run notebook                                 # notebook 06 reads the r2 set

# 6. Compare whatever has finished (partial runs included) with each other and the paper's
#    posterior, and project them through the scenarios; then open notebook 06.
pixi run python scripts/compare_reference.py
pixi run python scripts/compare_posteriors.py
pixi run notebook                            # notebooks/06-fast-calibration.ipynb
```

Per-arm resources (set in `scripts/cluster/submit_reference.sh`) and the local measurements
behind them are in the table in `docs/reference-runs.md`.

## Cluster runs

`scripts/cluster/` holds the array-job drivers for the published analyses (the 4×4 grid of
regression rate and unreachable susceptibility, `massiverun.py`, and the three sensitivity
analyses, `massiverun_sas.py`) and the base-case reference posterior (`reference_job.sh`), with
sbatch templates that run in this repository's pixi environment. Replace `<account>` and
`<repo>` in the templates, run `pixi install` once on a login node, then submit from the
repository root with `scripts/cluster/submit.sh <template>`, which creates
`outputs/cluster/logs/` first (SLURM does not). Each task writes to
`outputs/cluster/<analysis>/task_<i>/` and resumes from its checkpoints when resubmitted.

The `tpt_60` sensitivity analysis is not recalibrated: TPT completion affects no calibration
target, so its posterior is the base case's (`kiribati_tb.analysis.REUSES_BASE_POSTERIOR`). Its
task reuses a base-case posterior, checks at 32 draws that it scores identically, and runs only
the scenarios. Pass the posterior when submitting, e.g.
`BASE_POSTERIOR=outputs/reference/r2/nuts_td8/mcmc/idata.nc scripts/cluster/submit.sh scripts/cluster/array_job_sas.sh`
(add `BASE_BURN_IN=<draws per chain>` for a file that still holds warmup draws, such as the
published one at 10000). Without it, that task exits with a message and the other two run.

## Licence

BSD-2-Clause. The data under `data/` and the original model under `reference/tbh/` are copied
from `monash-emu/kiribati_tb_modelling` (BSD-2-Clause, Copyright (c) 2025, monash-emu).
