# summer4 workarounds

Every place this port works around a summer4 API, with the reason. Code carries a matching
`# WORKAROUND(summer4): ...` comment. summer4's roadmap step 21 handoff reads this list so the
next summer4 package can absorb them. Row IDs refer to summer4's `docs/evaluation/tb-ports.md`
(`KI*`) and `docs/evaluation/composability.md` (`CX*`).

## Workarounds

| # | Phase | File | What | Why | Relates to |
| --- | --- | --- | --- | --- | --- |
| W1 | K2 | `src/kiribati_tb/model.py` (`POOL`, `pmap`, `force_of_infection`) | With homogeneous mixing the map carries a one-trait property `pool` and the force of infection groups by it. | `ForceOfInfection` requires `group_by: Property`. summer2's homogeneous model has one mixing category (`I_total / N_total ** exp`); grouping by age or reachability would give per-group denominators instead, and no constant mixing matrix reproduces a single pool under a generalised exponent. | `KI4`; suggest `ForceOfInfection(group_by=None)` meaning one pool |
| W2 | K3 | `src/kiribati_tb/outputs.py` (`add_computed_values`) | The original's three computed values (`passive_detection_rate_clin`, `passive_detection_rate_subclin`, `mixing_matrix_distance`) are `SaveFn` leaves that recompute the value from the save time and the prepared parameters, duplicating the detection rate tree in plain JAX (both checked against the goldens). | summer4 cannot save the value of an ungrouped rate expression: `Capture`/`GroupedOutput` need a `GroupedRate`, and `ComputedValue` needs a `derived_fn` that runs in every vector-field call to produce values only the outputs read. (A clean timing showed no measurable `derived_fn` cost here, so this is about idiom, not speed.) | `KI12`; suggest a save quantity for a rate expression, e.g. `SaveRequest(Rate(expr))` evaluated at save times |

## Sharp edges met (no workaround needed)

Places where summer4 did what was asked but a port could easily get wrong. Recorded so summer4
can decide whether to change a default or its documentation.

| # | Phase | Where | What happened | Suggested summer4 change |
| --- | --- | --- | --- | --- |
| S1 | K1 | `model.py` `SUMMER2_SIGMOID_CURVATURE` | summer2's `get_sigmoidal_interpolation_function` defaults to curvature 16; summer4's `sharpness` (documented as "summer2's curvature parameter") defaults to 1, which is near-linear. Porting the call without the argument moved the population by up to 3.6 persons per compartment (7e-4 relative) after 1950. Passing `sharpness=16.0` gives 1e-9 agreement (`tests/test_demography.py`). | Say in `sigmoidal` / `Data.interp` / `TableData.interp` docstrings, and in the summer2 migration page, that summer2's default is 16. |
| S2 | K1 | `kiribati_tb/__init__.py` | summer2gen runs in float64 (computegraph enables `jax_enable_x64`); summer4 does not. The port enables it on import, as `tb-macro-summer4` does. | Document float64 as the parity setting for summer2 ports. |
| S3 | K1–K2 | `model.py` `SOLVER_KWARGS`, `reference/make_golden.py` | The yearly mixing matrix (a `Lookup` on `floor(Time() - 1850)`) jumps at every integer year. At `rtol = atol = 1e-8` both summer2gen's `odeint` and diffrax's `Dopri5` carry ~1.5e-5 relative error from the jumps, above the 1e-5 parity bar, so goldens and parity runs use `1e-10` (port within 1.5e-7 of the original). summer2gen also caps the step at one year; `CompiledModel.run` in v0.2.0a5 builds its own `PIDController` with no `dtmax` or `jump_ts`, and neither is needed at 1e-10. | Telling the controller about the yearly jumps (`jump_ts`) would let a port run at looser tolerance; summer4 `main`'s CP1 lets `run()` take the caller's controller, so this is available from the next tag. |
