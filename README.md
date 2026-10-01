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
`details.yaml` records the diagnostics. `notebooks/06-fast-calibration.ipynb` checks that it
recovers a long reference run and compares both with the published posterior
(`reference/published/`).

summer4 is pinned to `main` at `c9548d5` (after `v0.2.0a5`): the solver backends of its step 29
let the port choose diffrax's adjoint (forward mode where reverse-mode gradients are NaN), and
step 30's run objects are what `wf.sample_until` and `wf.optimize` return.

## Cluster runs

`scripts/cluster/` holds the array-job drivers for the published analyses (the 4×4 grid of
regression rate and unreachable susceptibility, `massiverun.py`, and the three sensitivity
analyses, `massiverun_sas.py`) and the base-case reference posterior (`reference_job.sh`), with
sbatch templates that run in this repository's pixi environment. Replace `<account>` and
`<repo>` in the templates, run `pixi install` once on a login node, then submit from the
repository root with `scripts/cluster/submit.sh <template>`, which creates
`outputs/cluster/logs/` first (SLURM does not). Each task writes to
`outputs/cluster/<analysis>/task_<i>/` and resumes from its checkpoints when resubmitted.

## Licence

BSD-2-Clause. The data under `data/` and the original model under `reference/tbh/` are copied
from `monash-emu/kiribati_tb_modelling` (BSD-2-Clause, Copyright (c) 2025, monash-emu).
