---
name: kiribati-tb-summer4-port
description: Port the Kiribati TB screening model (summer2gen, estival/pymc) to summer4 in a new pixi repo, monash-emu/kiribati-tb-summer4, at numerical parity and with the calibration and scenario analyses re-run on numpyro.
---

# Kiribati TB model → summer4 (`monash-emu/kiribati-tb-summer4`)

## Context

`monash-emu/kiribati_tb_modelling` holds the transmission model behind *From
rollout to refinement: using early screening data to model the next phase of
population-wide tuberculosis screening in Kiribati*. It runs on **summer2gen**
(`summerepi2` from `monash-emu/summer2gen`, a summer2 fork adding
`add_infection_generalised_flow`), calibrates with **estival + pymc
`DEMetropolisZ`**, finds a MAP with nevergrad, and runs posterior samples through
12+ screening scenarios on a SLURM cluster.

This plan lands the model in a **new repository**, `monash-emu/kiribati-tb-summer4`,
managed with pixi, depending on summer4 from GitHub. It is also kept in summer4
as `plans/kiribati-tb-summer4-port.plan.md` and copied into the new repo's
`plans/` in phase K0.

**Authoritative gap record:** `docs/evaluation/tb-ports.md` in summer4, rows
`KI1`–`KI23`. **Feature work:** `plans/tb-ports-feature-completeness.plan.md`
(WP12, WP13, WP3, WP14, WP15, WP16, WP10). Each phase below names the packages it
needs; do not start a phase against a summer4 that lacks them, and do not
hand-write a workaround for a gap a package is designed to close. Hand-written
workarounds are exactly what the ports ledger counts as `partial`.

Decisions taken with the user:

- **Calibration is numpyro** through summer4 WP10. No pymc, no estival.
- **Done means numerical parity** against golden outputs produced by the original
  summer2gen model, then the calibration and scenarios re-run.
- The new repo is created fresh (not a fork) under `monash-emu`.

### What the original model is

Read these in the original repo before starting.

| File | What it builds |
|---|---|
| `code/tbh/model.py` | 10 states; `AgeStratification` over `0 3 5 10 15 18 40 65`; reachability split by `reachable_pop_frac`; generalised infection from 4 susceptible states; latency, progression, clinical/infectiousness transitions; detection; screening flows; treatment outcomes; deaths recycled to (`mtb_naive`, age 0); births as importation |
| `code/tbh/age_mixing.py` | Time-varying mixing matrix from `bg_mixing`, `a_spread`, `pc_strength`, UN single-age weights and fertility age gaps, normalised by spectral radius; Canberra distance to a conmat matrix |
| `code/tbh/demographic_tools.py` | UN population and mortality → per-age death-rate sigmoidal interpolants; lookups for mixing |
| `code/tbh/interventions.py`, `data/scenarios.py` | Screening tools (sensitivities by state), programs as rate pulses `-log(1 - cov/frac)/duration`, scenarios |
| `code/tbh/outputs.py` | ~150 derived outputs (see K3) |
| `code/tbh/runner_tools.py` | Params and priors from `data/parameters.xlsx`, targets, Metropolis, full runs, quantiles, averted differences, parquet |
| `code/tbh/plotting.py`, `notebooks/*`, `remote_cluster/*`, `appendix/*` | Figures, manuscript tables, cluster array jobs |

Semantics to reproduce exactly (from reading summer2gen):

- **FOI:** per age category `g`, `M(t) @ (I_w[g] / N[g] ** infection_pop_scale)`,
  times `raw_transmission_rate * pop_2020 ** infection_pop_scale * rel_sus`.
  Categories are age only; reachability mixes homogeneously.
- **Flow outputs** (`raw_results=False`): `out[0] = f[0]`,
  `out[i] = (f[i] + f[i-1]) / 2` over instantaneous rates at output times.
- **Cumulative from `start_time`:** zero before, running sum from that index.
- **Solver:** summer2gen's JAX `odeint`, `rtol = atol = 1.4e-8`, `max_step = 1.0`,
  yearly output times 1850–2035.
- **Ageing:** `AgeStratification` adds ageing at `1/width` for every compartment.

---

## Repository shape

```
kiribati-tb-summer4/
├── pixi.toml
├── pyproject.toml              # package kiribati_tb (hatchling), black line-length 100
├── AGENTS.md                   # conventions below; points at summer4's tb-ports ledger
├── LICENSE                     # BSD-2-Clause, attributing the original repo
├── plans/kiribati-tb-summer4-port.plan.md
├── data/                       # copied verbatim from the original data/ (with attribution note)
├── src/kiribati_tb/
│   ├── paths.py  demography.py  mixing.py  model.py  interventions.py
│   ├── scenarios.py  outputs.py  params.py  calibration.py  analysis.py  plotting.py
├── reference/                  # runs ONLY in the reference env
│   └── make_golden.py
├── tests/
│   ├── golden/*.parquet        # committed; regenerated only by `pixi run -e reference golden`
│   ├── test_parity_structure.py  test_parity_transmission.py  test_parity_outputs.py
│   ├── test_mixing.py  test_demography.py  test_calibration.py  test_scenarios.py
├── scripts/cluster/            # array-job drivers and sbatch templates
└── notebooks/                  # outputs stripped (nbstripout pre-commit)
```

`pixi.toml`:

```toml
[workspace]
name = "kiribati-tb-summer4"
channels = ["conda-forge"]
platforms = ["osx-arm64", "linux-64"]     # JAX has no native win-64 support

[dependencies]
python = "3.13.*"
pandas = ">=2.2"
openpyxl = "*"
pyarrow = "*"
pyyaml = "*"
matplotlib = "*"
seaborn = "*"
jupyterlab = "*"
pytest = "*"
black = "*"

[pypi-dependencies]
# Before WP12 tags a release, pin the stack commit instead:
#   rev = "377e3f84a803d6898d42d06d7cf10d47d9875d5a"
summer4 = { git = "https://github.com/monash-emu/summer4.git", tag = "v0.2.0a1", extras = ["jax", "calibration", "pandas", "frames"] }
arviz = "*"
kiribati_tb = { path = ".", editable = true }

[tasks]
test = "pytest -q"
format = "black src tests reference scripts"
map = "python -m kiribati_tb.analysis map"
calibrate = "python -m kiribati_tb.analysis calibrate"
full-runs = "python -m kiribati_tb.analysis full-runs"
notebook = "jupyter lab notebooks"

[feature.reference]
platforms = ["osx-arm64", "linux-64"]
dependencies = { python = "3.10.*", pandas = "*", openpyxl = "*", pyarrow = "*" }
pypi-dependencies = { summerepi2 = { git = "https://github.com/monash-emu/summer2gen.git", rev = "<pin the commit used>" } }
tasks = { golden = "python reference/make_golden.py" }

[environments]
default = { solve-group = "default" }
reference = { features = ["reference"], no-default-feature = true }
```

When summer4 releases a new tag, bump it in one PR whose only other change is
whatever the parity suite forces. Never float on a branch.

### Conventions (write into the new repo's `AGENTS.md`)

- Google Python style, line length 100, black; type-annotate every function.
- `data/` and `tests/golden/` are inputs. Change them only with a commit that
  says why.
- **Parity is the definition of correct.** A PR that changes model code must keep
  `pixi run test` green against the committed goldens. If an intended change
  breaks parity (e.g. fixing an original bug), regenerate nothing: add the change
  behind a config flag defaulting to the original behaviour, and say so in the PR.
- Model code is summer4 idiom. No `derived_fn` for something a summer4 API
  expresses; when a phase would need one, the phase is blocked on its package.
- Notebooks are a user gate: every phase's PR lists the notebooks to run and the
  claim to check, and is not merged until the user signs off.
- Plans are immutable records in `plans/`; follow-up work gets a new plan.

---

## Phases

Each phase is one branch and one PR in the new repo. Gate column = summer4
packages that must be in the pinned tag.

| Phase | Gate | Ports rows exercised | Acceptance |
|---|---|---|---|
| K0 scaffold and goldens | none (pin stack commit) | `KI23` | Goldens regenerate byte-stable; summer4 imports |
| K1 structure and demography | WP12, WP13, WP3 | `KI1` `KI2` `KI3` `KI7`–`KI11` | Compartments match the `no_transmission` golden |
| K2 transmission and mixing | WP14 | `KI4` `KI5` `KI6` `KI12` | Compartments match goldens, heterogeneous mixing |
| K3 outputs | WP15 | `KI13`–`KI17` | Every golden derived output matches, same names |
| K4 calibration | WP16, WP10 | `KI18`–`KI20` `KI22` | Synthetic recovery; posterior agrees with the published analysis |
| K5 scenarios and full runs | WP10 | `KI21` | Scenario parity; averted-burden quantiles agree |
| K6 analysis, figures, cluster | — | — | Manuscript figures regenerate; user sign-off |

### K0 — Scaffold and goldens

1. Create `monash-emu/kiribati-tb-summer4` (ask the user before creating the
   repository — it is outward-facing), add the layout, `pixi.toml`, `AGENTS.md`,
   `LICENSE`, CI (`pixi run test` on `linux-64`; a manual-dispatch job for
   `pixi run -e reference golden` that fails if goldens differ).
2. Copy `data/` verbatim and this plan into `plans/`.
3. `reference/make_golden.py` imports the **original** `tbh` code (vendored
   read-only under `reference/tbh/` with its `data/` path pointed at the copied
   `data/`, since `tbh.paths` finds data via the git root) and, for each fixture,
   builds `get_tb_model(config, tv_params, screening_programs)`, runs it with
   `solver_kwargs` pinned (`rtol=atol=1e-8`) at fixed parameters, and writes:
   - `tests/golden/<fixture>/compartments.parquet` — all 160 compartments,
     yearly 1850–2035, columns named by `str(Compartment)`;
   - `tests/golden/<fixture>/derived.parquet` — every derived output, including
     `save_results=False` ones (request them explicitly);
   - `tests/golden/<fixture>/params.yaml` — the exact parameter dict used.
4. Fixtures (the parameter set is the constants sheet of `parameters.xlsx` with
   every prior at its midpoint, overridden by any `params_ow`):

   | Fixture | Mixing | Screening |
   |---|---|---|
   | `no_transmission` | `heterogeneous_mixing=False`, `raw_transmission_rate = 0` | none |
   | `homog_baseline` | `heterogeneous_mixing=False` | none |
   | `hetero_baseline` | heterogeneous | none |
   | `hetero_scenario_3` | heterogeneous | PEARL 85% (`scenario_3`) |
   | `hetero_scenario_8` | heterogeneous | CXR-TST 85% (TST → `cleared` path) |

5. A test in the default env that only imports `summer4` and reads the goldens.

**Acceptance:** two consecutive `golden` runs give identical parquet bytes; CI
green.

### K1 — Structure and demography

Port, in summer4 idiom:

- `demography.py` — `get_population_over_time`, `get_death_rates_by_age`,
  `build_agegap_lookup`, `build_age_weight_lookup` stay host-side pandas and
  NumPy (they are data preparation). The per-age death rates become one
  `Data.table(years, death_rates, over=age).interp("sigmoidal")` (WP13, `KI10`).
- `model.py`:
  - Properties: `state` (10 traits), `age` (8), `reach` (2);
    `PropertyMap.from_property(state).stratify(age).stratify(reach)` (`KI1`).
  - Ageing: `TransitionFlow("ageing", age.present(), age.present(), 1.0,
    pairing=TraitChain.from_breakpoints(age))` (`KI2`).
  - Latency and progression flows with age adjustments
    `Multiply(Param(f"progression_rate_age{band}"), where=age[...])`; BCG
    `rel_sus_children`; reachability adjustments (`KI7`).
  - Detection: `recent_detection_rate * tanh scale-up` as a WP13 rate tree;
    `rel_detection_subclin`, `rel_detection_unreachable`.
  - Treatment outcomes: the `get_neg_tx_outcome_funcs` algebra as rate trees over
    the death-rate table and the linear `tx_success_pct` interpolant, with
    `maximum(..., 0)` (WP13, `KI11`).
  - Deaths: one flow per cause, `TransitionFlow(name, state.isin(...),
    state["mtb_naive"] & age["0"], rate)` with the age-specific rate from the
    table; reachability is free and therefore preserved (`KI8`). Keep the
    original's exclusion of (`mtb_naive`, age 0) from natural death.
  - Births: two `EntryFlow`s into `state["mtb_naive"] & age["0"] & reach[r]` with
    `sigmoidal` of population increments × `reachable_pop_frac` (`KI9`).
  - Initial population: `InitialPopulation(base={state["mtb_naive"]: pop_1850 -
    seed, state["clin_inf"]: seed}, splits={reach: {"reachable": Param(...),
    "unreachable": 1 - Param(...)}})` with the age distribution of the original
    (summer2 `AgeStratification` splits evenly across age when unspecified —
    confirm against the golden `t = 1850` row) (WP3, `KI3`).
  - No infection flows yet. K1 is compared against the `no_transmission` fixture,
    so structure and demography are verified before transmission is added.
- `params.py` — read `parameters.xlsx` (constant sheet → dict, time-variant sheet →
  series) exactly as `get_parameters_and_priors` does, minus estival.
- Solver: `run(params, t0=1850.0, t1=2035.0, dt=1.0, solver="dopri5",
  rtol=1e-8, atol=1e-8, save=SavePlan(ts=np.arange(1850.0, 2036.0)))`.

**Tests — `tests/test_parity_structure.py`:** compartments equal the
`no_transmission` golden with `rtol=1e-5` and an absolute floor of `1e-6` persons,
compared by compartment label mapping `state × age × reach` to summer2's
`{state}Xage_{a}Xreachability_{r}` names; the `t = 1850` row exactly.
`tests/test_demography.py`: table interpolants equal summer2's sigmoidal
functions at 200 times.

### K2 — Transmission and mixing

- `mixing.py`:
  - Vectorise `gen_mixing_matrix_func`: build per-band single-age weight blocks as
    padded arrays and compute `S` with one `einsum` instead of 36 Python-looped
    `vmap`s.
  - Spectral radius: `C = S · diag(pop)` with `S` symmetric, `pop > 0`, so
    `ρ(C) = ρ(diag(√pop) S diag(√pop))`; use `jnp.linalg.eigvalsh`, which is
    reverse-differentiable. A unit test asserts equality with
    `jnp.linalg.eigvals` on the original construction for 20 random parameter
    draws.
  - `yearly_matrices(params) -> (n_years, 8, 8)`, computed once per evaluation in
    a params pre-step (WP10's `preprocess`), read in the vector field by
    `Lookup(Param("mixing_stack"), floor(Time() - 1850.0))` inside
    `MixingMatrix(age, ..., normalize="none")` (WP13, `KI6`). Match the original's
    `time.astype(int)` year truncation.
  - `canberra_distance` against `read_conmat_matrix`, evaluated once at 2025 and
    exposed as a `ComputedValue` or a pre-step output (`KI12`).
- Infection flows: four `EpiModel.add_infection_generalised_flow` (or declarative
  `TransitionFlow(..., ForceOfInfection(kind="generalised", exponent=
  Param("infection_pop_scale"), ...))`) from `mtb_naive`, `contained`, `cleared`,
  `recovered` to `incipient`, `contact_rate = raw_transmission_rate *
  pop_2020 ** infection_pop_scale * rel_sus`, with `adjust=` for BCG and
  `rel_sus_unreachable` (WP14, `KI4`).
- Infectiousness: compartment × age weights — `rel_infectiousness_lowinf`,
  `rel_infectiousness_subclin`, zero below 15 (WP14, `KI5`).
- Screening flows are deferred to K5; K2 fixtures have none.

**Tests:** `test_parity_transmission.py` — `hetero_baseline` and `homog_baseline`
compartments at `rtol=1e-5`; `test_mixing.py` — matrix stack equals the
original builder at 10 years × 5 parameter draws, eigvalsh equality, gradient of
the stack w.r.t. `a_spread` is finite.

### K3 — Outputs

`outputs.py` builds one `OutputSet` (WP15) reproducing **every** name in the
original `request_model_outputs`, including `save_results=False` intermediates
(they are needed for parity and cost nothing to keep in the set):

- population totals by age, reach, and age aggregates `3_9`, `15+`, `18+`
  (explicit age lists — there is no ordinal selector);
- births by reach; `tb_incidence` (both progression flows, `.midpoint()`) by reach,
  per 100k, `prop_tb_incidenceXreach_unreachable`, `cum_tb_incidence`
  (`cumulative(start=2026.0)`);
- prevalence per compartment × age × reach, TB and TBI prevalence per capita,
  viable TBI;
- measured TST / PEARL / CXR positivity: prevalence traces × `Param(prev_se_*)`,
  aggregated and divided by the matching reachable population;
- `% subclinical`, `% infectious`, notifications (four detection flows,
  midpoint), `% notifications clinical`, screening total, natural and TB mortality
  (midpoint), per 100k, `cum_tb_mortality`;
- computed values `passive_detection_rate_clin`, `…_subclin`,
  `mixing_matrix_distance`.

`Result.to_frame(shape="wide")` gives the `derived_outputs` DataFrame the
original analysis code expects (year index, one column per output).

**Tests — `test_parity_outputs.py`:** for every fixture, every golden column
matches at `rtol=1e-5` with an absolute floor of `1e-8`; the set of names is
equal (no missing, no extra); the output set evaluates under `jax.jit`.

### K4 — Calibration

- `calibration.py`:
  - Priors: `priors_from_frame` over the `parameters.xlsx` constant sheet;
    replicate `run_full_analysis`'s `infectiousness_gain_rate` narrowing rule.
  - Targets: `Target`s for each entry of `runner_tools.targets` with
    `Normal.from_tolerance` (20%, notifications 40%), and
    `mixing_matrix_distance ~ Normal(0, sd=Uniform(5, 20))` (`KI18`). Drop the
    distance target when mixing is homogeneous.
  - `BayesianModel(compiled, fixed_params, priors, targets, outputs=OUTPUTS,
    init=INITIAL_POPULATION, preprocess=add_mixing_stack, run_kwargs=...)`.
  - `find_map` replaces the nevergrad notebook (`KI20`); `sample(kind="aies")`
    replaces `DEMetropolisZ` (`KI19`), chains and draws configurable; NUTS is an
    option once K4 shows gradients are finite everywhere in the prior support.
  - Sensitivity variants as config: `tpt_60`, `subclinical_50`,
    `homogeneous_mixing`.
- WP16's benchmark numbers set expectations; record wall time per 1000 likelihood
  evaluations here (`KI22`).

**Acceptance:**
1. Synthetic recovery: simulate targets from the `hetero_baseline` parameters,
   calibrate a 5-parameter subset, true values inside 95% intervals.
2. Agreement with the published analysis: obtain the published run's `idata.nc`
   from the original authors (it is not in the git repo) and check that posterior
   medians of calibrated parameters fall inside the new 95% intervals and vice
   versa. If the file cannot be obtained, compare against the appendix parameter
   table (`appendix/tab-params.tex`) and say so in the PR.

### K5 — Scenarios and full runs

- `interventions.py`: `ScreeningTools` sensitivities as `Param` maps;
  `ScreeningProgram` rate `-log(1 - cov / reachable_pop_frac / 100) / duration`
  as a WP13 rate tree (`KI11`), shaped by `linear(Time(), [s - .01, s, e - .01, e],
  [0, r, r, 0])`; multiplied by sensitivity and success proportion; age
  multipliers and zero for unreachable via `adjust=`.
- `scenarios.py`: the scenario list from `data/scenarios.py`, unchanged in ids and
  names.
- `analysis.full_runs`: `bm.posterior_runs(idata, n, scenarios=...)` then
  `quantiles` and `differences(ref="baseline", outputs={"TB_averted":
  "cum_tb_incidence", "deaths_averted": "cum_tb_mortality"}, at=2035.0)`; parquet
  written with the original filenames (`uncertainty_df_<sc>.parquet`,
  `diff_quantiles_df_ref_baseline_<sc>.parquet`) and schema, plus `details.yaml`
  (timings, config, commit) (`KI21`).

**Acceptance:** `hetero_scenario_3` and `hetero_scenario_8` goldens match at
`rtol=1e-5` (screening flows and TST path); averted-burden quantiles for
`scenario_3` overlap the published manuscript intervals.

### K6 — Analysis, figures, cluster

- `plotting.py` ported with estival removed (targets come from summer4 `Target`s;
  priors from the WP10 prior objects).
- Notebooks: `full_analysis`, `find_map`, `manuscript_results`,
  `manuscript_figs`, `appendix_generation`, `sa_outputs`, `check_demographics`,
  `mixing_model` — each re-pointed at the new modules; the ones that only read
  parquet outputs need only path changes.
- `scripts/cluster/`: `massiverun.py` and `massiverun_sas.py` equivalents over the
  same sensitivity grid (`REGRESSION_RATE_VALUES × REL_SUS_UNREACHABLE_VALUES`),
  with sbatch templates taking a pixi environment instead of a conda one.
- README describing the study, the port, and how to reproduce figures.

**Acceptance (user gate):** the user runs `manuscript_figs` and `appendix_generation`
against a fresh full run and signs off that figures 1–4 and the appendix tables
tell the same story as the published versions.

---

## Verification

- `pixi run test` green at every phase against the committed goldens.
- `pixi run -e reference golden` reproduces goldens byte-for-byte (CI dispatch job).
- After K5, summer4's `docs/evaluation/tb-ports.md` shows every `KI*` row `full`.
  If a row is still `partial` because the port needed a workaround, open a summer4
  issue naming the row rather than closing it.
- Each phase PR body: rows exercised, gate tag, parity numbers (max relative error
  per fixture), notebooks for the user to run.
