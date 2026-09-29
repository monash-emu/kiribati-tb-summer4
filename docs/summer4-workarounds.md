# summer4 workarounds

Every place this port works around a summer4 API, with the reason. Code carries a matching
`# WORKAROUND(summer4): ...` comment. summer4's roadmap step 21 handoff reads this list so the
next summer4 package can absorb them. Row IDs refer to summer4's `docs/evaluation/tb-ports.md`
(`KI*`) and `docs/evaluation/composability.md` (`CX*`).

## Workarounds

| # | Phase | File | What | Why | Relates to |
| --- | --- | --- | --- | --- | --- |

## Sharp edges met (no workaround needed)

Places where summer4 did what was asked but a port could easily get wrong. Recorded so summer4
can decide whether to change a default or its documentation.

| # | Phase | Where | What happened | Suggested summer4 change |
| --- | --- | --- | --- | --- |
| S1 | K1 | `model.py` `SUMMER2_SIGMOID_CURVATURE` | summer2's `get_sigmoidal_interpolation_function` defaults to curvature 16; summer4's `sharpness` (documented as "summer2's curvature parameter") defaults to 1, which is near-linear. Porting the call without the argument moved the population by up to 3.6 persons per compartment (7e-4 relative) after 1950. Passing `sharpness=16.0` gives 1e-9 agreement (`tests/test_demography.py`). | Say in `sigmoidal` / `Data.interp` / `TableData.interp` docstrings, and in the summer2 migration page, that summer2's default is 16. |
| S2 | K1 | `kiribati_tb/__init__.py` | summer2gen runs in float64 (computegraph enables `jax_enable_x64`); summer4 does not. The port enables it on import, as `tb-macro-summer4` does. | Document float64 as the parity setting for summer2 ports. |
| S3 | K1 | `model.py` `SOLVER_KWARGS` | summer2gen's `odeint` caps the step at one year (`max_step=1`); `CompiledModel.run` in v0.2.0a5 builds its own `PIDController` with no `dtmax`. Parity holds without the cap at `rtol=atol=1e-8`, so nothing is worked around. summer4 `main` already lets `run()` take the caller's diffrax controller (CP1), which would let a port set `dtmax` when it matters. | Released in the next tag (CP1). |
