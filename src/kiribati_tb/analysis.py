"""Full runs of posterior draws under the screening scenarios (``run_full_runs`` and friends).

Ports the second half of ``run_full_analysis`` in ``tbh/runner_tools.py``: posterior draws run
under the baseline and every scenario through summer4's ``BayesianModel.posterior_runs``,
summarised as quantile frames and averted-burden quantiles, and written with the original's
file names and schemas:

- ``uncertainty_df_<scenario>.parquet``: time index, ``(output, quantile)`` columns, one per
  output the original saved (``save_results=True``);
- ``diff_quantiles_df_ref_baseline_<scenario>.parquet``: quantile index, columns
  ``TB_averted``, ``deaths_averted`` and their ``_relative`` versions, at 2035;
- ``details.yaml``: timings, config and commit.
"""

from __future__ import annotations

import subprocess
import time
import warnings
from collections.abc import Mapping, Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from summer4.epi.calibration import Scenario as RunScenario

from kiribati_tb.calibration import (
    CALIBRATION_END,
    TIGHT_SOLVER,
    CalibrationSetup,
    bayesian_model,
    forward_mode_solver,
)
from kiribati_tb.demography import AGE_GROUPS
from kiribati_tb.model import ACTIVE_COMPS, COMPARTMENTS, ModelConfig, compile_model
from kiribati_tb.pipeline import PipelineConfig, calibrate
from kiribati_tb.outputs import AGE_AGGREGATES, REACHABLE, build_outputs, restrict_outputs
from kiribati_tb.scenarios import SCENARIOS_BY_ID

QUANTILES: tuple[float, ...] = (0.025, 0.25, 0.5, 0.75, 0.975)
DIFF_NAMES: dict[str, str] = {
    "TB_averted": "cum_tb_incidence",
    "deaths_averted": "cum_tb_mortality",
}
END_YEAR = 2035.0


def intermediate_names(config: ModelConfig) -> set[str]:
    """Outputs the original requested with ``save_results=False``."""
    names: set[str] = set()
    for age in (*AGE_GROUPS, *AGE_AGGREGATES):
        names.add(f"populationXage_{age}Xreach_{REACHABLE}")
    for inf in ("lowinf", "inf"):
        for reach in ("reachable", "unreachable"):
            names.add(f"tb_incidence_{inf}Xreach_{reach}")
    for comp in COMPARTMENTS:
        names.add(f"prev_{comp}")
        for reach in ("reachable", "unreachable"):
            names.add(f"prev_{comp}Xreach_{reach}")
        for age in AGE_GROUPS:
            names.add(f"prev_{comp}Xage_{age}")
            names.add(f"tst_pos_{comp}Xage_{age}Xreach_{REACHABLE}")
            for reach in ("reachable", "unreachable"):
                names.add(f"prev_{comp}Xage_{age}Xreach_{reach}")
            if comp != "mtb_naive" or age != AGE_GROUPS[0]:
                names.add(f"all_cause_mortality_from_{comp}_age_{age}")
    for comp in ACTIVE_COMPS:
        names.add(f"notifications_{comp}")
        for age in AGE_GROUPS:
            for test in ("pearl", "cxr"):
                names.add(f"{test}_prev_{comp}Xage_{age}Xreach_{REACHABLE}")
    for age in AGE_GROUPS:
        names.add(f"tx_death_age_{age}")
        for inf in ("inf", "lowinf"):
            names.add(f"tb_mortality_{inf}_age_{age}")
    names.add(f"subclin_tb_prevalenceXreach_{REACHABLE}")
    names.add(f"infectious_tb_prevalenceXreach_{REACHABLE}")
    names.update(config.screening_flows())
    return names


def saved_names(config: ModelConfig) -> list[str]:
    """The outputs the original saved, in declaration order."""
    hidden = intermediate_names(config)
    return [name for name in build_outputs(config).keys() if name not in hidden]


def run_scenarios(
    setup: CalibrationSetup,
    draws: Any,
    scenario_ids: Sequence[str],
    *,
    n: int | None,
    burn_in: int = 0,
    seed: int = 0,
    batch_size: int = 16,
    solver: Mapping[str, Any] | None = None,
) -> Any:
    """``PosteriorRuns`` for the baseline and each scenario, recording the saved outputs."""
    names = saved_names(setup.config)
    bm = bayesian_model(setup, solver=solver)
    runs: dict[str, RunScenario | None] = {
        "baseline": RunScenario(outputs=restrict_outputs(build_outputs(setup.config), names))
    }
    for sc_id in scenario_ids:
        scenario = SCENARIOS_BY_ID[sc_id]
        config = ModelConfig(setup.config.heterogeneous_mixing, scenario.programs)
        runs[sc_id] = RunScenario(
            params=dict(scenario.params_ow) or None,
            model=compile_model(config),
            outputs=restrict_outputs(build_outputs(config), names),
        )
    return bm.posterior_runs(
        draws, n=n, burn_in=burn_in, seed=seed, scenarios=runs, batch_size=batch_size
    )


def averted_quantiles(
    runs: Any, ref: str = "baseline", at: float = END_YEAR, q: Sequence[float] = QUANTILES
) -> dict[str, pd.DataFrame]:
    """``calculate_diff_output_quantiles``: ``ref − scenario`` at ``at``, absolute and relative.

    WORKAROUND(summer4): ``PosteriorRuns.differences`` computes ``scenario − ref`` (and divides
    that by ``ref``), the opposite sign to the original's averted burden, and quantiles do not
    commute with negation. The per-draw samples are summarised here instead. See
    docs/summer4-workarounds.md, W4.
    """
    t_idx = int(np.argmin(np.abs(np.asarray(runs.times) - at)))
    q_arr = np.asarray(q, dtype=float)
    out: dict[str, pd.DataFrame] = {}
    for scenario in runs.scenario_names:
        if scenario == ref:
            continue
        cols: dict[str, np.ndarray] = {}
        rel: dict[str, np.ndarray] = {}
        for colname, output in DIFF_NAMES.items():
            ref_v = runs.samples(ref, output)[:, t_idx]
            sc_v = runs.samples(scenario, output)[:, t_idx]
            cols[colname] = np.quantile(ref_v - sc_v, q_arr)
            rel[f"{colname}_relative"] = np.quantile((ref_v - sc_v) / ref_v, q_arr)
        out[scenario] = pd.DataFrame({**cols, **rel}, index=q_arr)
    return out


def _git_commit() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def write_full_runs(
    runs: Any,
    folder: Path,
    *,
    times: Mapping[str, str] | None = None,
    model_config: Mapping[str, Any] | None = None,
    analysis_config: Mapping[str, Any] | None = None,
) -> None:
    """Write the original's parquet files and ``details.yaml``."""
    folder.mkdir(parents=True, exist_ok=True)
    for scenario, frame in runs.quantiles(QUANTILES).items():
        frame.to_parquet(folder / f"uncertainty_df_{scenario}.parquet")
    for scenario, frame in averted_quantiles(runs).items():
        frame.to_parquet(folder / f"diff_quantiles_df_ref_baseline_{scenario}.parquet")
    with open(folder / "details.yaml", "w") as handle:
        yaml.dump_all(
            [
                dict(times or {}),
                dict(model_config or {}),
                dict(analysis_config or {}) | {"scenarios": list(runs.scenario_names[1:])},
                {"commit": _git_commit()},
            ],
            handle,
            default_flow_style=False,
        )


def full_runs(
    setup: CalibrationSetup,
    draws: Any,
    folder: Path,
    scenario_ids: Sequence[str],
    *,
    n: int,
    burn_in: int = 0,
    seed: int = 0,
    batch_size: int = 16,
) -> Any:
    """Run the scenarios for ``n`` posterior draws and write the result files."""
    start = time.time()
    runs = run_scenarios(
        setup, draws, scenario_ids, n=n, burn_in=burn_in, seed=seed, batch_size=batch_size
    )
    elapsed = time.time() - start
    write_full_runs(
        runs,
        folder,
        times={"full_runs_time": f"{round(elapsed)} sec ({n} draws x {len(scenario_ids) + 1})"},
        model_config={"heterogeneous_mixing": setup.config.heterogeneous_mixing},
        analysis_config={"full_runs_samples": n, "burn_in": burn_in},
    )
    return runs


def run_full_analysis(
    folder: Path,
    *,
    sensitivity_analysis: str | None = None,
    param_overrides: Mapping[str, float] | None = None,
    config: PipelineConfig | None = None,
    full_runs_samples: int = 1000,
    seed: int = 0,
    scenario_ids: Sequence[str] | None = None,
    aggregate: str = "mean",
    deadline: float | None = None,
) -> Any:
    """Calibrate with :func:`kiribati_tb.pipeline.calibrate`, then run and write the scenarios.

    Every calibration stage checkpoints into ``folder`` (``optima.npz``, ``nuts/``), so
    rerunning the same call after a kill resumes it. ``idata.nc`` holds every post-warmup draw;
    ``details.yaml`` adds the convergence diagnostics (per-parameter R-hat, bulk and tail
    ESS, the optima found, stage timings). Full runs draw ``full_runs_samples`` posterior draws
    uniformly from all chains. ``deadline`` (``time.time()``) stops the calibration between
    warmup rounds or sampling chunks (``pipeline.DeadlineReached`` from warmup); the full runs
    happen only once it has converged or spent its draw budget.
    """
    from kiribati_tb.calibration import calibration_setup
    from kiribati_tb.scenarios import SCENARIOS

    folder.mkdir(parents=True, exist_ok=True)
    setup = calibration_setup(sensitivity_analysis, param_overrides, aggregate=aggregate)
    bm = bayesian_model(setup, t1=CALIBRATION_END)
    config = PipelineConfig() if config is None else config
    start = time.time()
    fallback = bayesian_model(setup, solver=forward_mode_solver(), t1=CALIBRATION_END)
    tight = bayesian_model(setup, solver=TIGHT_SOLVER, t1=CALIBRATION_END)
    fit = calibrate(
        bm, folder, config, fallback=fallback, metric_bm=tight, seed=seed, deadline=deadline
    )
    mcmc_time = time.time() - start
    idata = fit.idata
    idata.to_netcdf(folder / "idata.nc")
    if fit.checkpoint.rows and fit.checkpoint.rows[-1].get("decision") == "deadline":
        print(f"[analysis] {folder} stopped for its deadline; resubmit to resume.", flush=True)
        return fit, None
    summary = fit.summary()
    model_config = {
        "heterogeneous_mixing": setup.config.heterogeneous_mixing,
        "sensitivity_analysis": sensitivity_analysis,
        "param_overrides": dict(param_overrides or {}),
        "likelihood_aggregate": aggregate,
    }
    analysis_config = {
        "pipeline": _plain(asdict(config)),
        "full_runs_samples": full_runs_samples,
        "diagnostics": summary,
    }
    calibration_time = f"{round(mcmc_time)} sec (this call; stages: {summary['seconds']})"
    runs = None
    if full_runs_samples > 0:
        ids = [s.sc_id for s in SCENARIOS] if scenario_ids is None else list(scenario_ids)
        start = time.time()
        runs = run_scenarios(setup, idata, ids, n=full_runs_samples, seed=seed)
        runs_time = time.time() - start
        write_full_runs(
            runs,
            folder,
            times={
                "calibration_time": calibration_time,
                "full_runs_time": f"{round(runs_time)} sec ({full_runs_samples} draws)",
            },
            model_config=model_config,
            analysis_config=analysis_config,
        )
    else:
        with open(folder / "details.yaml", "w") as handle:
            yaml.dump_all(
                [{"calibration_time": calibration_time}, model_config, analysis_config],
                handle,
                default_flow_style=False,
            )
    if not fit.converged:
        warnings.warn(
            f"Calibration in {folder} stopped without converging ({summary['decision']}); "
            "rerun to extend it.",
            RuntimeWarning,
            stacklevel=2,
        )
    return fit, runs


def _plain(value: Any) -> Any:
    """``value`` with tuples and numpy scalars turned into YAML-safe Python types."""
    if isinstance(value, Mapping):
        return {k: _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


#: Sensitivity analyses that change only parameters no calibration target depends on, so their
#: posterior is the base case's: they reuse the base-case draws and run only the scenarios
#: (:func:`run_reused_analysis`). ``tpt_60`` lowers TPT completion, which acts only inside the
#: screening programmes; it scored identically to the base case at all 128 draws checked
#: (``docs/neutra.md``). :func:`same_posterior` re-checks this at run time.
REUSES_BASE_POSTERIOR: frozenset[str] = frozenset({"tpt_60"})


def posterior_draws(idata: Any, *, burn_in: int = 0) -> dict[str, np.ndarray]:
    """``idata``'s posterior as flat arrays per site, after dropping ``burn_in`` draws per chain."""
    post = idata.posterior
    return {name: np.asarray(post[name].values)[:, burn_in:].reshape(-1) for name in post.data_vars}


def same_posterior(
    a: CalibrationSetup,
    b: CalibrationSetup,
    draws: Mapping[str, np.ndarray],
    *,
    n: int = 32,
    seed: int = 0,
    tol: float = 1e-9,
) -> float:
    """Check that ``a`` and ``b`` give the same log density at ``n`` of ``draws``.

    Returns the largest absolute difference; raises ``ValueError`` if it exceeds ``tol`` (in
    nats, relative to the larger magnitude) or if the two setups calibrate different sites.
    """
    import jax

    bm_a = bayesian_model(a, t1=CALIBRATION_END)
    bm_b = bayesian_model(b, t1=CALIBRATION_END)
    sites = sorted(bm_a.prior_names())
    if sites != sorted(bm_b.prior_names()):
        raise ValueError("The two setups calibrate different parameters.")
    total = len(next(iter(draws.values())))
    rows = np.random.default_rng(seed).choice(total, size=min(n, total), replace=False)
    density_a, density_b = jax.jit(bm_a.log_density), jax.jit(bm_b.log_density)
    worst = 0.0
    for i in rows:
        z = bm_a.unconstrain({s: float(draws[s][i]) for s in sites})
        la, lb = float(density_a(z)), float(density_b(z))
        diff = abs(la - lb)
        if not np.isfinite(diff) or diff > tol * max(1.0, abs(la), abs(lb)):
            raise ValueError(
                f"Posteriors differ at draw {i}: log density {la:.6f} vs {lb:.6f}; "
                "this analysis must be calibrated, not reused."
            )
        worst = max(worst, diff)
    return worst


def run_reused_analysis(
    folder: Path,
    posterior: Path,
    *,
    sensitivity_analysis: str,
    burn_in: int = 0,
    full_runs_samples: int = 1000,
    seed: int = 0,
    scenario_ids: Sequence[str] | None = None,
    check_draws: int = 32,
) -> Any:
    """Run a sensitivity analysis's scenarios on the base-case posterior, without recalibrating.

    Only for analyses in :data:`REUSES_BASE_POSTERIOR`. ``posterior`` is a base-case
    ``idata.nc`` (for example a converged reference run's ``mcmc/idata.nc``); ``burn_in``
    drops that many draws per chain first. :func:`same_posterior` confirms at ``check_draws``
    draws that the analysis's posterior equals the base case's before anything is run. Writes
    the same files as :func:`run_full_analysis`, with the source posterior recorded in
    ``details.yaml``.
    """
    import numpyro  # noqa: F401 - import before arviz; see docs/summer4-workarounds.md, S4
    import arviz as az

    from kiribati_tb.calibration import calibration_setup
    from kiribati_tb.scenarios import SCENARIOS

    if sensitivity_analysis not in REUSES_BASE_POSTERIOR:
        raise ValueError(
            f"{sensitivity_analysis!r} changes the posterior; calibrate it with run_full_analysis."
        )
    folder.mkdir(parents=True, exist_ok=True)
    idata = az.from_netcdf(posterior)
    base = calibration_setup()
    setup = calibration_setup(sensitivity_analysis)
    worst = same_posterior(
        base, setup, posterior_draws(idata, burn_in=burn_in), n=check_draws, seed=seed
    )
    print(
        f"[analysis] {sensitivity_analysis}: posterior equals the base case's at {check_draws} "
        f"draws (largest difference {worst:.2e} nats); reusing {posterior}",
        flush=True,
    )
    idata.to_netcdf(folder / "idata.nc")
    ids = [s.sc_id for s in SCENARIOS] if scenario_ids is None else list(scenario_ids)
    start = time.time()
    runs = run_scenarios(setup, idata, ids, n=full_runs_samples, burn_in=burn_in, seed=seed)
    write_full_runs(
        runs,
        folder,
        times={
            "calibration_time": "0 sec (base-case posterior reused)",
            "full_runs_time": f"{round(time.time() - start)} sec ({full_runs_samples} draws)",
        },
        model_config={
            "heterogeneous_mixing": setup.config.heterogeneous_mixing,
            "sensitivity_analysis": sensitivity_analysis,
            "param_overrides": {},
            "posterior_reused_from": str(posterior),
            "posterior_burn_in": burn_in,
        },
        analysis_config={
            "full_runs_samples": full_runs_samples,
            "same_posterior_check": {"draws": check_draws, "max_abs_diff_nats": worst},
        },
    )
    return runs
