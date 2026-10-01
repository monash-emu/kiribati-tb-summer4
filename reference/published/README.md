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

**Provenance.** The post-burn-in traces in the paper's results appendix (Figure S2) match this
file draw for draw (for example the dip of `infection_pop_scale` to about 0.1 and the spike of
`raw_transmission_rate` to about 0.0015 just after draw 10,000), so it is the paper's base-case
run. The paper's priors (methods appendix, Table 2) are `data/parameters.xlsx`'s; its base case
is regression rates 1.0 per year, `rel_sus_unreachable` 1.5 and TPT completion 70%, which is
`calibration_setup()` with no arguments.

**The paper's protocol** (methods appendix §9.3): 8 `DEMetropolisZ` chains of 10,000 tuning plus
20,000 draws (240,000 model evaluations), 8 cores per cluster job, "10 to 15 hours" per
configuration; convergence judged by trace plots, smoothness and Gelman-Rubin R-hat below
1.05. Projections **discard the first 10,000 draws of each chain** and use 2,000 posterior
samples. The port's "published" arm therefore uses draws 10,000-19,999 only, and projects 2,000
of them drawn with a fixed seed (recorded where the projections are written). The recorded
`sampling_time` (72,130 s, about 20 h) is above the paper's 10-15 h; unresolved.

**It is not converged, even by its own criterion.** Over all 160,000 draws the largest
rank-normalised split R-hat is 1.22 and several parameters have bulk ESS below 30. On the
post-burn-in draws the largest R-hat is 1.16 (rank-normalised), 1.18 (split) or 1.17 (classic
Gelman-Rubin), with 12 of 19 parameters above 1.05, and bulk ESS is 35-419. The chains overlap
in every pairwise panel (one mode); the excess R-hat is slow, sticky random-walk mixing along
correlated directions, with chain means about one posterior sd apart for the worst parameters. Use it as a consistency check whose Monte Carlo error is set by its ESS, not by its
draw count; the port's converged reference posterior is `notebooks/06-fast-calibration.ipynb`'s
subject. Committed as data: change it only with a commit that says why.
