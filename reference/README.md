# Reference (original model)

`reference/` runs only in the `reference` pixi environment
(`pixi run -e reference golden`). It never imports summer4.

`tbh/` is vendored read-only from
[`monash-emu/kiribati_tb_modelling`](https://github.com/monash-emu/kiribati_tb_modelling)
at commit `46a63f0` (`code/tbh/`), BSD-2-Clause, Copyright (c) 2025, monash-emu.

| File | Status |
| --- | --- |
| `model.py`, `age_mixing.py`, `demographic_tools.py`, `interventions.py`, `outputs.py` | Verbatim |
| `paths.py` | Modified: `DATA_FOLDER` points at this repository's `data/` instead of searching for the git root of the working directory |
| `__init__.py` | Modified: no eager imports (the original imports `plotting` and `runner_tools`, which need pymc and estival) |
| `plotting.py`, `runner_tools.py` | Not vendored (pymc / estival). `make_golden.py` copies the two things it needs from `runner_tools.py`: `DEFAULT_MODEL_CONFIG` and the `parameters.xlsx` reader |

The reference environment reproduces the original repository's `pixi.lock`: python 3.10,
summer2gen (`summerepi2`) at `aad6d4b414db2dbf81ea64e8aece5e0c6cdaa425`, jax/jaxlib 0.4.38,
computegraph 0.4.5, numpy 1.25.2, pandas 2.3.0.

Do not format `tbh/` (it is excluded from black in `pyproject.toml`).
