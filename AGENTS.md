# Agent instructions

This repository ports the Kiribati TB screening model from summer2gen to summer4. The plan is
`plans/kiribati-tb-summer4-port.plan.md` (phases K0–K6). Where a summer4 capability is missing,
the authoritative gap record is summer4's
[`docs/evaluation/tb-ports.md`](https://github.com/monash-emu/summer4/blob/main/docs/evaluation/tb-ports.md),
rows `KI1`–`KI23`. Quote those IDs in PRs.

## Branches and PRs

Each phase is one branch and one PR (`feat/k1-structure`, `feat/k2-transmission`, ...). Phases
stack: cut phase N from phase N-1's branch and open the PR against it. Do not merge without
review.

## Style

- Google Python style, line length 100, black (`pixi run format`). Type-annotate every function,
  including tests and scripts.
- JAX first, as in summer4's `AGENTS.md`: vectorised gathers and one batched op rather than
  Python loops that emit one JAX op per iteration. Host-side pandas/NumPy is fine for data
  preparation that produces constant arrays.
- JAX float64 is enabled in `kiribati_tb/__init__.py`. summer2gen enables it; summer4 does not.

## Inputs

- `data/` and `tests/golden/` are inputs. Change them only with a commit that says why.
- Goldens are regenerated only by `pixi run -e reference golden`, from the original code
  vendored under `reference/tbh/`. Two consecutive runs must give identical bytes.
- Do not format `reference/tbh/`.

## Parity is the definition of correct

A PR that changes model code keeps `pixi run test` green against the committed goldens. If an
intended change breaks parity (for example, fixing a bug in the original), regenerate nothing:
put the change behind a config flag defaulting to the original behaviour, and say so in the PR.
Parity failures are reported as max absolute and relative error per compartment or output, not
pass/fail.

## summer4 idiom, and recording workarounds

Model code is summer4 idiom. No `derived_fn` for something a summer4 API expresses; when a phase
would need one, the phase is blocked on the summer4 package that closes it. Do not hand-write a
workaround for a gap a summer4 work package is designed to close.

Where the port must work around a summer4 API anyway, mark the code
`# WORKAROUND(summer4): <one line>` and add a row to `docs/summer4-workarounds.md` (file, what,
why, which `KI` row or summer4 finding it relates to). summer4's roadmap handoff for step 21
reads that list.

Two limitations belong to this port, not to summer4 (summer4 `tb-ports.md`, *Limitations that
stay in the ports*):

- Spectral normalisation of the mixing matrix uses the symmetric eigenvalue routine
  (`jnp.linalg.eigvalsh` on `diag(√pop) S diag(√pop)`) so it is reverse-differentiable.
- The original mixing builder loops in Python over 36 age-band pairs. The port vectorises it
  and builds one matrix per year before the solve.

## Notebooks and plans

- Notebooks are a user gate: every phase PR lists the notebooks to run and the claim to check,
  and is not merged until the user signs off. Commit notebooks without outputs.
- Plans are immutable records in `plans/`; follow-up work gets a new plan.

## Checks

```bash
pixi run format-check
pixi run test
pixi run -e reference golden && git diff --exit-code -- tests/golden   # only when regenerating
```
