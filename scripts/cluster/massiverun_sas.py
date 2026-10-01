"""Array-job driver for the three sensitivity analyses (``massiverun_sas.py``).

    pixi run python scripts/cluster/massiverun_sas.py <array_job_id> <task_id> \\
        [--base-posterior outputs/reference/r2/<arm>/mcmc/idata.nc] [--base-burn-in 0]

Same settings, checkpointing and resume behaviour as ``massiverun.py`` for the analyses that
are calibrated. ``tpt_60`` changes nothing the calibration targets depend on, so its posterior
is the base case's (``kiribati_tb.analysis.REUSES_BASE_POSTERIOR``): it reuses the base-case
draws given by ``--base-posterior`` (checked to score identically before use) and runs only the
scenarios. It refuses to start without one rather than recalibrating.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from massiverun import CALIBRATION_HOURS, RUN_CONFIG, dump_map, log_job, task_folder

from kiribati_tb.analysis import REUSES_BASE_POSTERIOR, run_full_analysis, run_reused_analysis
from kiribati_tb.pipeline import DeadlineReached

ANALYSIS_NAME = "sas"
SA_BY_TASK_ID: dict[int, str] = {1: "tpt_60", 2: "subclinical_50", 3: "homogeneous_mixing"}


def main() -> None:
    start = time.time()
    parser = argparse.ArgumentParser()
    parser.add_argument("array_job_id", type=int)
    parser.add_argument("task_id", type=int)
    parser.add_argument("--base-posterior", default=None, help="base-case idata.nc to reuse")
    parser.add_argument("--base-burn-in", type=int, default=0, help="draws per chain to drop")
    args = parser.parse_args()
    analysis = SA_BY_TASK_ID[args.task_id]
    if analysis in REUSES_BASE_POSTERIOR and not args.base_posterior:
        raise SystemExit(
            f"{analysis} reuses the base-case posterior: pass --base-posterior "
            "(or BASE_POSTERIOR=... to the sbatch script), e.g. a converged reference run's "
            "outputs/reference/<run>/<arm>/mcmc/idata.nc."
        )
    if args.task_id == 1:
        dump_map(ANALYSIS_NAME, "sa", SA_BY_TASK_ID)
    folder = task_folder(args.task_id, ANALYSIS_NAME)
    log_job(folder, args.array_job_id)
    if analysis in REUSES_BASE_POSTERIOR:
        run_reused_analysis(
            folder,
            Path(args.base_posterior),
            sensitivity_analysis=analysis,
            burn_in=args.base_burn_in,
            full_runs_samples=RUN_CONFIG.get("full_runs_samples", 1000),
            seed=args.task_id,
        )
    else:
        try:
            run_full_analysis(
                folder,
                sensitivity_analysis=analysis,
                seed=args.task_id,
                deadline=start + CALIBRATION_HOURS * 3600.0,
                **RUN_CONFIG,
            )
        except DeadlineReached as exc:
            print(
                f"Task {args.task_id} stopped for its deadline ({exc}); resubmit to resume.",
                flush=True,
            )
    print(f"Finished in {time.time() - start:.0f} seconds", flush=True)


if __name__ == "__main__":
    main()
