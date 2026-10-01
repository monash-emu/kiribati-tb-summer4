"""Shared helpers for the gradient-performance benchmarks in ``scripts/bench/``.

Every timing here is taken with ``jax.block_until_ready`` and, where several variants are
compared, interleaved (one call of each variant per round) so that load on a shared machine
hits every variant alike. Reported numbers are medians with the 25th and 75th percentiles.
"""

from __future__ import annotations

import dataclasses
import json
import statistics
import time
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import diffrax
import jax
import numpyro.distributions  # noqa: F401 - before arviz (docs/summer4-workarounds.md, S4)
import jax.numpy as jnp
import numpy as np

from summer4.solvers import Diffrax

from kiribati_tb.calibration import (
    CALIBRATION_SOLVER,
    ORIGINAL_CALIBRATION_SOLVER,
    bayesian_model,
    calibration_setup,
    map_point,
)
from kiribati_tb.paths import REPO_ROOT

OUT_DIR = REPO_ROOT / "outputs" / "bench"
PUBLISHED = REPO_ROOT / "reference" / "published" / "idata.nc"
# Integer years 1851..2034 inside the solve window: where the yearly mixing matrix jumps.
MIXING_JUMPS: np.ndarray = np.arange(1851.0, 2035.0)


def block(tree: Any) -> Any:
    """Wait for every array in ``tree``."""
    return jax.tree.map(
        lambda x: x.block_until_ready() if hasattr(x, "block_until_ready") else x, tree
    )


def quantiles(times: Sequence[float], cpu: Sequence[float] | None = None) -> dict[str, float]:
    """Median, interquartile range and 10th percentile of ``times`` (seconds).

    ``cpu`` (process CPU seconds per call, all threads) is less sensitive than wall time to
    other processes on a loaded machine; its median is reported as ``cpu_median_s``.
    """
    arr = np.asarray(times, dtype=float)
    out = {
        "median_s": float(np.median(arr)),
        "p25_s": float(np.percentile(arr, 25)),
        "p75_s": float(np.percentile(arr, 75)),
        "p10_s": float(np.percentile(arr, 10)),
        "n": int(arr.size),
    }
    if cpu is not None and len(cpu):
        c = np.asarray(cpu, dtype=float)
        out["cpu_median_s"] = float(np.median(c))
        out["cpu_p25_s"] = float(np.percentile(c, 25))
        out["cpu_p75_s"] = float(np.percentile(c, 75))
    return out


_LAST_CPU: list[float] = [0.0]


def time_once(fn: Callable[..., Any], *args: Any) -> tuple[float, Any]:
    """Wall time of one blocked call (its process CPU time is left in ``last_cpu()``)."""
    cpu0 = time.process_time()
    start = time.perf_counter()
    out = block(fn(*args))
    wall = time.perf_counter() - start
    _LAST_CPU[0] = time.process_time() - cpu0
    return wall, out


def last_cpu() -> float:
    """Process CPU seconds of the last :func:`time_once` call."""
    return _LAST_CPU[0]


def interleaved(
    calls: Mapping[str, tuple[Callable[..., Any], tuple[Any, ...]]], rounds: int
) -> dict[str, dict[str, float]]:
    """Time every call once per round, ``rounds`` times, rotating the order each round.

    Every call must already be compiled (call it once first).
    """
    names = list(calls)
    samples: dict[str, list[float]] = {n: [] for n in names}
    cpu: dict[str, list[float]] = {n: [] for n in names}
    for r in range(rounds):
        order = names[r % len(names) :] + names[: r % len(names)]
        for name in order:
            fn, args = calls[name]
            seconds, _ = time_once(fn, *args)
            samples[name].append(seconds)
            cpu[name].append(last_cpu())
    return {name: quantiles(samples[name], cpu[name]) for name in names}


def count_eqns(jaxpr: Any) -> int:
    """Equations in ``jaxpr``, counting every nested sub-jaxpr (loop bodies, branches, pjit)."""
    core = jaxpr.jaxpr if hasattr(jaxpr, "jaxpr") else jaxpr
    total = 0
    for eqn in core.eqns:
        total += 1
        for value in eqn.params.values():
            for sub in value if isinstance(value, (list, tuple)) else (value,):
                if hasattr(sub, "eqns") or hasattr(sub, "jaxpr"):
                    total += count_eqns(sub)
    return total


def compile_seconds(fn: Callable[..., Any], *args: Any) -> dict[str, float]:
    """Trace+lower and XLA-compile times of ``jax.jit(fn)`` at ``args``."""
    start = time.perf_counter()
    lowered = jax.jit(fn).lower(*args)
    lower_s = time.perf_counter() - start
    start = time.perf_counter()
    lowered.compile()
    return {"lower_s": lower_s, "compile_s": time.perf_counter() - start}


def pid(
    rtol: float,
    atol: float | None = None,
    *,
    jumps: bool = False,
    dtmax: float | None = None,
    pcoeff: float = 0.0,
    icoeff: float = 1.0,
    dcoeff: float = 0.0,
    factormax: float = 10.0,
    safety: float = 0.9,
) -> Any:
    """A diffrax ``PIDController``; ``jumps`` tells it about the yearly mixing jumps."""
    return diffrax.PIDController(
        rtol=rtol,
        atol=rtol if atol is None else atol,
        jump_ts=MIXING_JUMPS if jumps else None,
        dtmax=dtmax,
        pcoeff=pcoeff,
        icoeff=icoeff,
        dcoeff=dcoeff,
        factormax=factormax,
        safety=safety,
    )


def solver_kwargs(
    solver: Any = None,
    controller: Any = None,
    adjoint: Any = None,
    max_steps: int = 4096,
) -> dict[str, Any]:
    """``run`` keyword arguments for a diffrax backend (default Dopri5 at 1.4e-4)."""
    return {
        "solver": Diffrax(
            diffrax.Dopri5() if solver is None else solver,
            stepsize_controller=pid(1.4e-4) if controller is None else controller,
            adjoint=adjoint,
        ),
        "max_steps": max_steps,
    }


def setup_and_model(
    solver: Mapping[str, Any] | str = "original", *, target_years: bool = False
) -> tuple[Any, Any]:
    """The base-case ``CalibrationSetup`` and a ``BayesianModel`` with ``solver``.

    ``solver`` is ``"original"`` (the original's Dopri5 at 1.4e-4: the configuration before this
    work), ``"calibration"`` (``CALIBRATION_SOLVER``: after), or a ``run`` kwargs dict. With
    ``target_years`` the solve saves only the target years and the year before each (a private
    override of the save plan, for measuring that lever; not used by the port).
    """
    setup = calibration_setup()
    if solver == "original":
        kwargs = dict(ORIGINAL_CALIBRATION_SOLVER)
    elif solver == "calibration":
        kwargs = dict(CALIBRATION_SOLVER)
    else:
        kwargs = dict(solver)
    bm = bayesian_model(setup, solver=kwargs)
    if target_years:
        bm._save_plan = dataclasses.replace(bm._save_plan, ts=target_save_ts(setup))
    return setup, bm


def target_save_ts(setup: Any) -> np.ndarray:
    """Every target year and the year before it (flow outputs use summer2's midpoint)."""
    years = {float(t) for target in setup.targets for t in np.asarray(target.times)}
    return np.asarray(sorted(years | {y - 1.0 for y in years}))


def published_points(bm: Any, n: int, seed: int = 0) -> list[dict[str, Any]]:
    """``n`` unconstrained points drawn from the published posterior (post burn-in)."""
    import arviz as az

    idata = az.from_netcdf(PUBLISHED)
    post = idata.posterior.isel(draw=slice(10_000, None))
    flat = {k: np.asarray(post[k]).reshape(-1) for k in bm.prior_names() if k in post}
    rng = np.random.default_rng(seed)
    idx = rng.choice(next(iter(flat.values())).size, size=n, replace=False)
    mid = map_point(calibration_setup())
    points = []
    for i in idx:
        params = {k: float(flat[k][i]) if k in flat else float(mid[k]) for k in bm.prior_names()}
        points.append({k: jnp.asarray(v) for k, v in bm.unconstrain(params).items()})
    return points


def map_z(bm: Any) -> dict[str, Any]:
    """The MAP (``outputs/find_map/base/map.json``, else prior midpoints), unconstrained."""
    return {k: jnp.asarray(v) for k, v in bm.unconstrain(map_point(calibration_setup())).items()}


def flat_grad(grad: Mapping[str, Any], names: Sequence[str]) -> np.ndarray:
    """A gradient dict as a vector in ``names`` order."""
    return np.asarray([float(grad[k]) for k in names])


def write_json(name: str, payload: Any) -> Path:
    """Write ``payload`` to ``outputs/bench/<name>.json`` and return the path."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / f"{name}.json"
    path.write_text(json.dumps(payload, indent=2, default=float))
    return path


def median(values: Sequence[float]) -> float:
    """Median of ``values``."""
    return float(statistics.median(values))
