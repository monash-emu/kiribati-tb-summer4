"""Cheaper alternatives to NUTS, timed against the same diagnostics.

    pixi run python scripts/alternatives.py laplace [--from outputs/calibrate/base]
    pixi run python scripts/alternatives.py aies-de [--from ...] [--walkers 40] [--max-hours 2]

``laplace``: a Student-t (5 degrees of freedom) at the best optimum with the Laplace covariance,
corrected by Pareto-smoothed importance sampling; reports ``k_hat`` (reliable below 0.7) and the
importance-sampling ESS, and writes 4,000 resampled draws.

``aies-de``: numpyro's affine-invariant ensemble sampler with the differential-evolution move
(``AIES.DEMove``), walkers drawn from the Laplace approximation, run in checkpointed chunks
with the same stop rule as the NUTS pipeline (R-hat ≤ 1.01, bulk and tail ESS ≥ 400, walkers
treated as chains) or until ``--max-hours``.

Both read ``optima.npz`` and ``metric.npy`` from ``--from`` (a pipeline output folder) and
write to ``outputs/alternatives/<method>/``.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpyro  # noqa: F401 - import before arviz; see docs/summer4-workarounds.md, S4
import arviz as az
import jax.numpy as jnp
import numpy as np

from summer4.epi.calibration import workflow as wf

from kiribati_tb.calibration import bayesian_model, calibration_setup
from kiribati_tb.paths import REPO_ROOT
from kiribati_tb.pipeline import Checkpoint, StopCriteria, laplace_psis

OUT = REPO_ROOT / "outputs" / "alternatives"


def mode_and_metric(folder: Path) -> tuple[dict[str, float], np.ndarray]:
    best = wf.Candidates.load(folder / "optima.npz").best(1)
    return {k: float(np.asarray(v)[0]) for k, v in best.z.items()}, np.load(folder / "metric.npy")


def run_laplace(bm: object, folder: Path, n: int, seed: int) -> None:
    mode, _ = mode_and_metric(folder)
    out = OUT / "laplace_psis"
    out.mkdir(parents=True, exist_ok=True)
    tight = bayesian_model(
        calibration_setup(), solver={"solver": "dopri5", "rtol": 1e-8, "atol": 1e-8}
    )
    fit = laplace_psis(bm, mode, n=n, seed=seed, hessian_bm=tight)
    draws = fit.resample(n, seed=seed)
    params = bm.constrain({k: jnp.asarray(v) for k, v in draws.items()})  # type: ignore[attr-defined]
    az.from_dict({"posterior": {k: np.asarray(v)[None, :] for k, v in params.items()}}).to_netcdf(
        out / "idata.nc"
    )
    record = {
        "k_hat": fit.k_hat,
        "ess": fit.ess,
        "n": n,
        "failed_solves": fit.failed,
        "seconds": fit.seconds,
    }
    (out / "summary.json").write_text(json.dumps(record, indent=2))
    np.save(out / "log_ratio.npy", fit.log_ratio)
    print(json.dumps(record), flush=True)


def run_aies_de(
    bm: object, folder: Path, walkers: int, chunk: int, hours: float, seed: int
) -> None:
    from numpyro.infer import AIES, MCMC

    mode, metric = mode_and_metric(folder)
    sites = sorted(mode)
    rng = np.random.default_rng(seed)
    start = rng.multivariate_normal([mode[s] for s in sites], metric, size=walkers)
    init = {s: start[:, i] for i, s in enumerate(sites)}
    kernel = AIES(bm.numpyro_model(), moves={AIES.DEMove(): 1.0})  # type: ignore[attr-defined]
    mcmc = MCMC(
        kernel,
        num_warmup=chunk,
        num_samples=chunk,
        num_chains=walkers,
        chain_method="vectorized",
        progress_bar=False,
    )
    out = OUT / "aies_de"
    criteria = StopCriteria(max_samples=10**6, max_seconds=hours * 3600.0, max_divergence_frac=None)
    stop = Checkpoint.resume(out, mcmc, criteria) or Checkpoint(mcmc, out, criteria)
    t0 = time.perf_counter()
    if stop.rows:
        wf.sample_until(mcmc, stop, seed=seed + len(stop.rows), warn=False)
    else:
        wf.sample_until(mcmc, stop, init_params=init, seed=seed, warn=False)
    print(f"aies-de finished after {time.perf_counter() - t0:.0f} s: {stop.rows[-1]}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("method", choices=["laplace", "aies-de"])
    parser.add_argument("--from", dest="source", default="outputs/calibrate/base")
    parser.add_argument("--n", type=int, default=4000)
    parser.add_argument("--walkers", type=int, default=40)
    parser.add_argument("--chunk", type=int, default=500)
    parser.add_argument("--max-hours", type=float, default=2.0)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    bm = bayesian_model(calibration_setup())
    folder = REPO_ROOT / args.source
    if args.method == "laplace":
        run_laplace(bm, folder, args.n, args.seed)
    else:
        run_aies_de(bm, folder, args.walkers, args.chunk, args.max_hours, args.seed)


if __name__ == "__main__":
    main()
