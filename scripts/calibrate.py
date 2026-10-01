"""Full calibration and scenario runs (``run_full_analysis`` without pymc or estival).

Calibrates with the fast pipeline (``kiribati_tb.pipeline.calibrate``: design, multi-start
L-BFGS, mode check, seeded dense-mass NUTS in checkpointed chunks until split R-hat ≤ 1.01 and
bulk and tail ESS ≥ 400), then runs posterior draws under the baseline and every scenario and
writes the original's files to ``outputs/calibrate/<name>/``: ``idata.nc``,
``uncertainty_df_*.parquet``, ``diff_quantiles_df_ref_baseline_*.parquet`` and
``details.yaml`` (with the diagnostics). Rerunning the same command resumes a killed run.

    pixi run calibrate [--analysis tpt_60] [--override rel_sus_unreachable=2.0] \\
        [--chains 8] [--chain-method parallel] [--max-hours 40] [--full-runs 1000]

``--chain-method parallel`` gives each chain its own XLA CPU device (one core each), which is
set up here before JAX is imported.

The long reference runs (four samplers) are ``scripts/reference_run.py`` and
``scripts/cluster/submit_reference.sh``.
"""

from __future__ import annotations

import argparse
import os
import sys


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--analysis", default=None)
    parser.add_argument("--override", action="append", default=[], metavar="NAME=VALUE")
    parser.add_argument("--name", default=None, help="output folder name")
    parser.add_argument("--chains", type=int, default=8)
    parser.add_argument("--chain-method", default="parallel", choices=["parallel", "vectorized"])
    parser.add_argument(
        "--warmup-rounds",
        default="100,100,200,200,400",
        help="warmup round lengths; rounds stop early once wf.WarmupRule passes",
    )
    parser.add_argument("--chunk", type=int, default=200, help="draws per chain per checkpoint")
    parser.add_argument("--rhat", type=float, default=1.01)
    parser.add_argument("--ess", type=float, default=400.0, help="bulk and tail ESS target")
    parser.add_argument("--max-samples", type=int, default=20000, help="draws per chain")
    parser.add_argument("--max-hours", type=float, default=None, help="sampling budget")
    parser.add_argument("--full-runs", type=int, default=1000, help="0: calibrate only")
    parser.add_argument("--jitter", type=float, default=0.05, help="start spread (logit units)")
    parser.add_argument(
        "--progress",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="numpyro progress bars in the log (default on)",
    )
    parser.add_argument("--no-shear", action="store_true", help="NUTS in plain logit coordinates")
    parser.add_argument(
        "--aggregate",
        default="mean",
        choices=["mean", "sum"],
        help="combine a target's years by mean (the code, default) or sum (the paper's text)",
    )
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args(argv)


def use_host_devices(n: int) -> None:
    """One XLA CPU device per chain, for ``chain_method="parallel"``; before JAX is imported."""
    if "jax" in sys.modules:
        raise RuntimeError("Set XLA host devices before importing jax.")
    flag = f"--xla_force_host_platform_device_count={int(n)}"
    os.environ["XLA_FLAGS"] = (os.environ.get("XLA_FLAGS", "") + " " + flag).strip()


def main() -> None:
    args = parse_args()
    if args.chain_method == "parallel":
        use_host_devices(args.chains)

    from kiribati_tb.analysis import run_full_analysis
    from kiribati_tb.paths import REPO_ROOT
    from kiribati_tb.pipeline import PipelineConfig, StopCriteria

    overrides = {k: float(v) for k, v in (item.split("=", 1) for item in args.override)}
    name = args.name or args.analysis or "base"
    criteria = StopCriteria(
        rhat=args.rhat,
        ess=args.ess,
        max_samples=args.max_samples,
        max_seconds=None if args.max_hours is None else args.max_hours * 3600.0,
    )
    config = PipelineConfig(
        num_chains=args.chains,
        chain_method=args.chain_method,
        chunk=args.chunk,
        warmup_rounds=tuple(int(n) for n in args.warmup_rounds.split(",")),
        criteria=criteria,
        shear=not args.no_shear,
        jitter=args.jitter,
        progress_bar=args.progress,
    )
    run_full_analysis(
        REPO_ROOT / "outputs" / "calibrate" / name,
        sensitivity_analysis=args.analysis,
        param_overrides=overrides,
        config=config,
        full_runs_samples=args.full_runs,
        seed=args.seed,
        aggregate=args.aggregate,
    )


if __name__ == "__main__":
    main()
