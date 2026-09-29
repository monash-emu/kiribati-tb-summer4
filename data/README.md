# Data

Every file in this directory except this README is copied verbatim from
[`monash-emu/kiribati_tb_modelling`](https://github.com/monash-emu/kiribati_tb_modelling)
at commit `46a63f0` (`data/`), under that repository's BSD-2-Clause licence
(Copyright (c) 2025, monash-emu).

| File | What it is | Original source |
| --- | --- | --- |
| `parameters.xlsx` | Constant parameters and priors (`constant` sheet); treatment success over time (`time_variant` sheet) | Model authors |
| `un_population.csv`, `un_mortality.csv`, `un_fertility_rates_KIR.csv` | UN World Population Prospects extracts for Kiribati | UN WPP |
| `notifications.xlsx` | Historical Kiribati TB notifications (calibration target) | Model authors |
| `Rscript/conmat_all_KIR.csv`, `Rscript/KIR_pop_2025.csv`, `Rscript/conmat_matrix.R` | Synthetic contact matrix from the `conmat` R package and the script that made it | Model authors |
| `scenarios.py`, `__init__.py` | Screening scenario definitions (imports `tbh.interventions`; used by `reference/make_golden.py`) | Model authors |

These files are inputs. Change them only in a commit that says why (see `AGENTS.md`).
