"""Time NUTS iterations with vectorised against parallel chains.

    pixi run python scripts/bench_chains.py --chains 4 --method parallel [--iters 40]

Runs ``--iters`` warmup and ``--iters`` sampling iterations of dense-mass NUTS, seeded from the
optimised points in ``outputs/explore/optima.npz`` (or a design), and reports seconds per
iteration and per gradient evaluation. Appends a row to ``outputs/bench/chains.json``.
"""

from __future__ import annotations

import argparse
import os
import sys


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--chains", type=int, default=4)
    parser.add_argument("--method", default="parallel", choices=["parallel", "vectorized"])
    parser.add_argument("--iters", type=int, default=40)
    args = parser.parse_args()
    if args.method == "parallel":
        # One XLA CPU device per chain; must be set before jax is imported.
        assert "jax" not in sys.modules
        os.environ["XLA_FLAGS"] = f"--xla_force_host_platform_device_count={args.chains}"

    import json
    import time

    import jax
    import numpy as np
    from jax import random

    from summer4.epi.calibration import workflow as wf

    from kiribati_tb.calibration import bayesian_model, calibration_setup
    from kiribati_tb.paths import REPO_ROOT
    from kiribati_tb.pipeline import nuts_factory

    setup = calibration_setup()
    bm = bayesian_model(setup)
    path = REPO_ROOT / "outputs" / "explore" / "optima.npz"
    seeds = wf.Candidates.load(path).best(args.chains)
    make = nuts_factory(bm, num_chains=args.chains, chunk=args.iters, chain_method=args.method)
    mcmc = make(args.iters)
    start = time.perf_counter()
    mcmc.warmup(
        random.PRNGKey(0),
        init_params=seeds.init_params(args.chains, jitter=0.01),
        extra_fields=("num_steps",),
    )
    warm_s = time.perf_counter() - start
    warm_steps = int(np.sum(mcmc.get_extra_fields()["num_steps"]))
    start = time.perf_counter()
    mcmc.run(random.PRNGKey(1), extra_fields=("num_steps",))
    run_s = time.perf_counter() - start
    steps = np.asarray(mcmc.get_extra_fields(group_by_chain=True)["num_steps"])
    row = {
        "chains": args.chains,
        "method": args.method,
        "devices": jax.device_count(),
        "iters": args.iters,
        "warmup_s_incl_compile": round(warm_s, 1),
        "warmup_leapfrog_total": warm_steps,
        "sample_s_incl_compile": round(run_s, 1),
        "sample_leapfrog_per_chain": steps.sum(axis=1).tolist(),
        "s_per_iter": round(run_s / args.iters, 3),
        "s_per_leapfrog_max_chain": round(run_s / float(steps.sum(axis=1).max()), 4),
    }
    print(json.dumps(row), flush=True)
    out = REPO_ROOT / "outputs" / "bench" / "chains.json"
    rows = json.loads(out.read_text()) if out.exists() else []
    out.write_text(json.dumps(rows + [row], indent=2))


if __name__ == "__main__":
    main()
