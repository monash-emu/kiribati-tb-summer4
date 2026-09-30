"""Array-job driver for the published grid (``remote_cluster/scripts/massiverun.py``).

Task ``i`` of the array calibrates and runs the scenarios for the ``i``-th combination of
``REGRESSION_RATE_VALUES`` (clinical regression and infectiousness loss rate) and
``REL_SUS_UNREACHABLE_VALUES``. Task 1 also writes ``config_map.yaml``.

    pixi run python scripts/cluster/massiverun.py <array_job_id> <task_id>

Each task's folder is ``outputs/cluster/<analysis>/task_<i>`` with no job id in it, so a task
that is killed (wall time, preemption) and resubmitted resumes from its checkpoints: the
optima, the Laplace metric, the warmed-up sampler, and every chunk of draws. The array job ids
that touched a task are appended to its ``jobs.log``.

Chains run one per core (``chain_method="parallel"``): ``CHAINS`` XLA CPU devices are
requested before JAX is imported, so the sbatch ``--cpus-per-task`` must equal ``CHAINS``.
"""

from __future__ import annotations

import os
import sys
import time
from itertools import product
from pathlib import Path
from typing import Any

CHAINS = 8
if "jax" not in sys.modules:  # one XLA CPU device per chain; must precede the JAX import
    os.environ["XLA_FLAGS"] = (
        os.environ.get("XLA_FLAGS", "") + f" --xla_force_host_platform_device_count={CHAINS}"
    ).strip()

import yaml  # noqa: E402

from kiribati_tb.analysis import run_full_analysis  # noqa: E402
from kiribati_tb.paths import REPO_ROOT  # noqa: E402
from kiribati_tb.pipeline import PipelineConfig, StopCriteria  # noqa: E402

ANALYSIS_NAME = "longer_runs"
OUTPUT_PARENT = REPO_ROOT / "outputs" / "cluster"
REGRESSION_RATE_VALUES: tuple[float, ...] = (0.5, 1.0, 2.0, 3.0)
REL_SUS_UNREACHABLE_VALUES: tuple[float, ...] = (1.0, 1.5, 2.0, 3.0)
# The original ran DEMetropolisZ for 10,000 tuning + 20,000 draws per chain and projected 2,000
# draws. Here NUTS stops when split R-hat <= 1.01 and bulk and tail ESS >= 400 for every
# parameter; the wall-time budget leaves room inside the sbatch --time for the full runs.
RUN_CONFIG: dict[str, Any] = {
    "config": PipelineConfig(
        num_chains=CHAINS,
        chain_method="parallel",
        chunk=200,
        criteria=StopCriteria(rhat=1.01, ess=400.0, max_seconds=36 * 3600.0),
    ),
    "full_runs_samples": 2000,
}


def build_param_grid() -> list[dict[str, float]]:
    return [
        {
            "clinical_regression_rate": rate,
            "infectiousness_loss_rate": rate,
            "rel_sus_unreachable": rel_sus,
        }
        for rate, rel_sus in product(REGRESSION_RATE_VALUES, REL_SUS_UNREACHABLE_VALUES)
    ]


def task_folder(task_id: int, analysis_name: str) -> Path:
    return OUTPUT_PARENT / analysis_name / f"task_{task_id}"


def log_job(folder: Path, array_job_id: int) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    with open(folder / "jobs.log", "a") as handle:
        handle.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} array job {array_job_id}\n")


def dump_map(analysis_name: str, key: str, mapping: dict) -> None:
    folder = OUTPUT_PARENT / analysis_name
    folder.mkdir(parents=True, exist_ok=True)
    with open(folder / f"{key}_map.yaml", "w") as handle:
        yaml.dump(
            {"analysis_name": analysis_name, "grid_size": len(mapping), f"task_to_{key}": mapping},
            handle,
            sort_keys=False,
        )


def main() -> None:
    start = time.time()
    array_job_id, task_id = int(sys.argv[1]), int(sys.argv[2])
    grid = build_param_grid()
    if not 1 <= task_id <= len(grid):
        raise ValueError(f"Task id {task_id} outside 1-{len(grid)}")
    if task_id == 1:
        dump_map(ANALYSIS_NAME, "config", dict(enumerate(grid, start=1)))
    overrides = grid[task_id - 1]
    folder = task_folder(task_id, ANALYSIS_NAME)
    log_job(folder, array_job_id)
    print(f"Task {task_id} parameter overrides: {overrides}", flush=True)
    run_full_analysis(folder, param_overrides=overrides, seed=task_id, **RUN_CONFIG)
    print(f"Finished in {time.time() - start:.0f} seconds", flush=True)


if __name__ == "__main__":
    main()
