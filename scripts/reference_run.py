"""Run one reference calibration arm of the base case (``kiribati_tb.reference``).

    pixi run python scripts/reference_run.py --arm nuts_td8 [--out DIR] [--max-hours 44]
    pixi run python scripts/reference_run.py --arm sa --smoke          # minutes, tiny settings

Arms: ``nuts_td8``, ``nuts_td5``, ``sa``, ``ess``. Output (default
``outputs/reference/<arm>``, ``outputs/reference_smoke/<arm>`` with ``--smoke``):
``optima.npz``, ``metric.npy``, ``mcmc/warmup.{pkl,csv}`` after every warmup round, and after
every sampling chunk ``mcmc/idata.nc``, ``mcmc/diagnostics.json`` (R-hat, bulk and tail ESS per
parameter, divergences, tree depth, acceptance, wall time, CPU hours), ``mcmc/progress.csv``
and ``mcmc/mcmc_state.pkl``. Rerunning with the same ``--out`` resumes from the last saved
round or chunk. The run stops when every parameter has R-hat ≤ 1.01 and bulk and tail
ESS ≥ 400 (decision ``diagnostics``), or before ``--max-hours`` of wall time have passed: it
does not start a warmup round or sampling chunk that its previous pace says would overrun
(decision ``deadline``), so a job given a little less than its SLURM wall time never loses
work to the timeout.

Parallel-chain arms get one XLA CPU device per chain; that is set here before JAX is imported.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

# Jax-free copy of kiribati_tb.reference.DEVICES (tests keep the two equal).
DEVICES: dict[str, int] = {"nuts_td8": 8, "nuts_td5": 8, "sa": 8, "ess": 1}
SMOKE_DEVICES = 2


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", required=True, choices=sorted(DEVICES))
    parser.add_argument("--out", default=None, help="output folder (resumes if it exists)")
    parser.add_argument("--smoke", action="store_true", help="tiny settings, minutes")
    parser.add_argument(
        "--max-hours",
        type=float,
        default=None,
        help="wall-clock budget for this job; stops between rounds/chunks, never mid-way",
    )
    parser.add_argument("--max-samples", type=int, default=None, help="draws per chain cap")
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    if "jax" in sys.modules:
        raise RuntimeError("XLA devices must be set before jax is imported.")
    devices = SMOKE_DEVICES if args.smoke and DEVICES[args.arm] > 1 else DEVICES[args.arm]
    flag = f"--xla_force_host_platform_device_count={devices}"
    os.environ["XLA_FLAGS"] = (os.environ.get("XLA_FLAGS", "") + " " + flag).strip()

    from dataclasses import asdict, replace
    from pathlib import Path

    import jax

    from kiribati_tb.calibration import (
        CALIBRATION_END,
        TIGHT_SOLVER,
        bayesian_model,
        calibration_setup,
        forward_mode_solver,
    )
    from kiribati_tb.paths import REPO_ROOT
    from kiribati_tb.pipeline import DeadlineReached, calibrate
    from kiribati_tb.reference import reference_config

    started = time.time()
    deadline = None if args.max_hours is None else started + args.max_hours * 3600.0
    config = reference_config(args.arm, smoke=args.smoke)
    if args.max_samples is not None:
        config = replace(config, criteria=replace(config.criteria, max_samples=args.max_samples))
    if config.chain_method == "parallel" and jax.device_count() < config.num_chains:
        raise RuntimeError(f"{config.num_chains} parallel chains need as many XLA devices.")
    parent = "reference_smoke" if args.smoke else "reference"
    out = Path(args.out) if args.out else REPO_ROOT / "outputs" / parent / args.arm
    out.mkdir(parents=True, exist_ok=True)
    print(f"[reference] arm={args.arm} out={out} devices={jax.device_count()}", flush=True)
    print(f"[reference] config={json.dumps(asdict(config), default=str)}", flush=True)

    setup = calibration_setup()
    bm = bayesian_model(setup, t1=CALIBRATION_END)
    fallback = bayesian_model(setup, solver=forward_mode_solver(), t1=CALIBRATION_END)
    tight = bayesian_model(setup, solver=TIGHT_SOLVER, t1=CALIBRATION_END)
    try:
        fit = calibrate(
            bm,
            out,
            config,
            fallback=fallback,
            metric_bm=tight if config.tight_metric else bm,
            seed=args.seed,
            deadline=deadline,
        )
    except DeadlineReached as exc:
        print(f"[reference] {args.arm} stopped for the deadline ({exc}); resubmit to resume.")
        return
    summary = fit.summary() | {"arm": args.arm, "smoke": args.smoke, "cpus": config.cpus}
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    print(
        f"[reference] {args.arm} finished in {time.time() - started:.0f} s: "
        f"decision={summary['decision']} rhat_max={summary['rhat_max']:.3f} "
        f"ess_min={min(summary['ess_bulk_min'], summary['ess_tail_min']):.0f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
