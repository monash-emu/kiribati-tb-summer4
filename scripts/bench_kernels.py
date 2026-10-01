"""Seconds per MCMC iteration for each reference arm's kernel, compilation excluded.

    pixi run python scripts/bench_kernels.py --arm sa [--short 10 --long 40]

Starts from the optima and Laplace metric of ``--from`` (a pipeline output folder), runs the arm's
kernel (one chain, or the arm's full ensemble of walkers for ``ess``) twice with the same warmup
and ``--short`` then ``--long`` draws, and reports ``(t_long - t_short) / (long - short)``: each
``MCMC.run`` compiles afresh, so the difference cancels compilation. Appends to
``outputs/bench/kernels.json``.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import replace

import jax
import numpy as np
from jax import random

from summer4.epi.calibration import workflow as wf

from kiribati_tb.calibration import CALIBRATION_END, bayesian_model, calibration_setup
from kiribati_tb.paths import REPO_ROOT
from kiribati_tb.pipeline import (
    KERNEL_FIELDS,
    initial_positions,
    make_mcmc,
    ridge_shear,
    sheared_covariance,
)
from kiribati_tb.reference import REFERENCE_ARMS


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", required=True, choices=sorted(REFERENCE_ARMS))
    parser.add_argument("--from", dest="source", default="outputs/explore")
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--short", type=int, default=5)
    parser.add_argument("--long", type=int, default=20)
    args = parser.parse_args()
    folder = REPO_ROOT / args.source
    bm = bayesian_model(calibration_setup(), t1=CALIBRATION_END)
    optima = wf.Candidates.load(folder / "optima.npz")
    metric = np.load(
        folder / ("metric.npy" if (folder / "metric.npy").exists() else "laplace_cov.npy")
    )
    arm = REFERENCE_ARMS[args.arm]
    chains = arm.num_chains if arm.kernel == "ess" else 1
    config = replace(
        arm,
        num_chains=chains,
        chain_method="vectorized" if chains > 1 else "sequential",
        progress_bar=False,
    )
    init = initial_positions(optima, config, metric)
    shear = None
    run_metric = metric
    if config.kernel == "nuts":
        best = optima.best(1)
        mode = {k: float(np.asarray(v)[0]) for k, v in best.z.items()}
        shear = ridge_shear(bm, mode, metric)
        run_metric = sheared_covariance(shear, mode, metric)
        init = {k: np.asarray(v) for k, v in shear.forward(init).items()}
    if chains == 1:
        init = {k: np.asarray(v)[0] for k, v in init.items()}
    times = {}
    steps = {}
    for n in (args.short, args.long):
        mcmc = make_mcmc(
            bm, replace(config, chunk=n), args.warmup, inverse_mass_matrix=run_metric, shear=shear
        )
        start = time.perf_counter()
        mcmc.run(random.PRNGKey(0), init_params=init, extra_fields=KERNEL_FIELDS[config.kernel])
        jax.block_until_ready(mcmc.last_state)
        times[n] = time.perf_counter() - start
        extra = mcmc.get_extra_fields()
        if "num_steps" in extra:
            steps[n] = float(np.mean(np.asarray(extra["num_steps"])))
    per_iteration = (times[args.long] - times[args.short]) / (args.long - args.short)
    row = {
        "arm": args.arm,
        "chains_or_walkers": chains,
        "warmup": args.warmup,
        "draws": [args.short, args.long],
        "seconds": times,
        "seconds_per_iteration": per_iteration,
        "leapfrog_mean": steps.get(args.long),
        "load_average": float(__import__("os").getloadavg()[0]),
    }
    print(json.dumps(row), flush=True)
    path = REPO_ROOT / "outputs" / "bench" / "kernels.json"
    rows = json.loads(path.read_text()) if path.exists() else []
    path.write_text(json.dumps(rows + [row], indent=2, default=str))


if __name__ == "__main__":
    main()
