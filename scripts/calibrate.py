"""Full calibration and scenario runs (``run_full_analysis`` without pymc or estival).

Samples the posterior with numpyro AIES (the gradient-free stand-in for DEMetropolisZ), then
runs posterior draws under the baseline and every scenario and writes the original's files to
``outputs/calibrate/<name>/``: ``idata.nc``, ``uncertainty_df_*.parquet``,
``diff_quantiles_df_ref_baseline_*.parquet`` and ``details.yaml``.

    pixi run calibrate [--analysis tpt_60] [--override rel_sus_unreachable=2.0] \\
        [--walkers 40] [--warmup 5000] [--samples 5000] [--full-runs 1000]

The original ran 8 DEMetropolisZ chains of 10,000 tuning plus 20,000 draws (240,000 model
evaluations). One likelihood evaluation costs about 0.09 s on one CPU core here; plan full
runs on the cluster.
"""

from __future__ import annotations

import argparse

from kiribati_tb.analysis import run_full_analysis
from kiribati_tb.paths import REPO_ROOT


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--analysis", default=None)
    parser.add_argument("--override", action="append", default=[], metavar="NAME=VALUE")
    parser.add_argument("--name", default=None, help="output folder name")
    parser.add_argument("--walkers", type=int, default=40)
    parser.add_argument("--warmup", type=int, default=5000)
    parser.add_argument("--samples", type=int, default=5000)
    parser.add_argument("--burn-in", type=int, default=0)
    parser.add_argument("--full-runs", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    overrides = {k: float(v) for k, v in (item.split("=", 1) for item in args.override)}
    name = args.name or args.analysis or "base"
    run_full_analysis(
        REPO_ROOT / "outputs" / "calibrate" / name,
        sensitivity_analysis=args.analysis,
        param_overrides=overrides,
        walkers=args.walkers,
        warmup=args.warmup,
        samples=args.samples,
        full_runs_samples=args.full_runs,
        burn_in=args.burn_in,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
