# kiribati-tb-summer4

A port of the Kiribati tuberculosis screening model
([`monash-emu/kiribati_tb_modelling`](https://github.com/monash-emu/kiribati_tb_modelling),
built on summer2gen with estival and pymc) to
[summer4](https://github.com/monash-emu/summer4), with calibration on numpyro.

The original model supports *From rollout to refinement: using early screening data to model
the next phase of population-wide tuberculosis screening in Kiribati*: 10 TB states × 8 age
bands × 2 reachability strata (160 compartments), 1850–2035, about 150 derived outputs, and
screening scenarios for 2026–2035.

**Correct means numerical parity** with golden outputs produced by the original summer2gen
code (`tests/golden/`, regenerated only by `pixi run -e reference golden`).

The work is planned in `plans/kiribati-tb-summer4-port.plan.md` as phases K0–K6. Gaps in
summer4 are tracked in summer4's `docs/evaluation/tb-ports.md` (rows `KI1`–`KI23`).

```bash
pixi run test                   # parity and unit tests against the committed goldens
pixi run -e reference golden    # regenerate goldens from the original model (slow, ~10 min)
```
