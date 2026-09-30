# Published posterior (original analysis)

`idata.nc` is the posterior the original authors sampled for the **base case** (no parameter
overrides, no sensitivity analysis; confirmed by the user, 2026-10-01). It was produced by the
original code (`monash-emu/kiribati_tb_modelling`, estival + pymc), not by this port.

| Attribute | Value |
| --- | --- |
| Sampler | pymc 5.2.0 `DEMetropolisZ` |
| Chains × draws | 8 × 20,000, after 10,000 tuning steps per chain |
| Sampling time | 72,130 s (about 20 h) |
| Written by | arviz 0.21.0, 2026-08-19 |
| Groups | `posterior` (the port's 19 calibrated parameters, same names), `sample_stats` (`accept`, `accepted`, `lambda`, `scaling`); no `log_likelihood` |

It loads with `az.from_netcdf` in this repository's default environment.

**It is not converged.** Over all 160,000 draws the largest rank-normalised split R-hat is
1.22 and several parameters have bulk ESS below 30 (dropping the first 10,000 draws: R-hat 1.16,
ESS 35). Use it as a consistency check whose Monte Carlo error is set by its ESS, not by its
draw count; the port's converged reference posterior is `notebooks/06-fast-calibration.ipynb`'s
subject. Committed as data: change it only with a commit that says why.
