"""Array-job driver for the published grid (``remote_cluster/scripts/massiverun.py``).

Task ``i`` of the array calibrates and runs the scenarios for the ``i``-th combination of
``REGRESSION_RATE_VALUES`` (clinical regression and infectiousness loss rate) and
``REL_SUS_UNREACHABLE_VALUES``. Task 1 also writes ``task_config_map.yaml``.

    pixi run python scripts/cluster/massiverun.py <array_job_id> <task_id>
"""

from __future__ import annotations

import sys
import time
from itertools import product
from pathlib import Path

import yaml

from kiribati_tb.analysis import run_full_analysis
from kiribati_tb.paths import REPO_ROOT

ANALYSIS_NAME = "longer_runs"
OUTPUT_PARENT = REPO_ROOT / "outputs" / "cluster"
REGRESSION_RATE_VALUES: tuple[float, ...] = (0.5, 1.0, 2.0, 3.0)
REL_SUS_UNREACHABLE_VALUES: tuple[float, ...] = (1.0, 1.5, 2.0, 3.0)
# The original doubled its run lengths for this grid: DEMetropolisZ 10,000 tune + 20,000 draws
# per chain, 2,000 full runs after 10,000 burn-in. AIES walkers are set to the same number of
# model evaluations per walker-step budget; adjust on the cluster.
RUN_CONFIG: dict[str, int] = {
    "walkers": 40,
    "warmup": 5000,
    "samples": 5000,
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


def task_folder(array_job_id: int, task_id: int, analysis_name: str) -> Path:
    return OUTPUT_PARENT / f"{array_job_id}_{analysis_name}" / f"task_{task_id}"


def dump_map(array_job_id: int, analysis_name: str, key: str, mapping: dict) -> None:
    folder = OUTPUT_PARENT / f"{array_job_id}_{analysis_name}"
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
        dump_map(array_job_id, ANALYSIS_NAME, "config", dict(enumerate(grid, start=1)))
    overrides = grid[task_id - 1]
    print(f"Task {task_id} parameter overrides: {overrides}", flush=True)
    run_full_analysis(
        task_folder(array_job_id, task_id, ANALYSIS_NAME),
        param_overrides=overrides,
        seed=task_id,
        **RUN_CONFIG,
    )
    print(f"Finished in {time.time() - start:.0f} seconds", flush=True)


if __name__ == "__main__":
    main()
