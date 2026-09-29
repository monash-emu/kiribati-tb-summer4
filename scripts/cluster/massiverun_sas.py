"""Array-job driver for the three sensitivity analyses (``massiverun_sas.py``).

pixi run python scripts/cluster/massiverun_sas.py <array_job_id> <task_id>
"""

from __future__ import annotations

import sys
import time

from kiribati_tb.analysis import run_full_analysis

from massiverun import RUN_CONFIG, dump_map, task_folder

ANALYSIS_NAME = "sas"
SA_BY_TASK_ID: dict[int, str] = {1: "tpt_60", 2: "subclinical_50", 3: "homogeneous_mixing"}


def main() -> None:
    start = time.time()
    array_job_id, task_id = int(sys.argv[1]), int(sys.argv[2])
    if task_id == 1:
        dump_map(array_job_id, ANALYSIS_NAME, "sa", SA_BY_TASK_ID)
    run_full_analysis(
        task_folder(array_job_id, task_id, ANALYSIS_NAME),
        sensitivity_analysis=SA_BY_TASK_ID[task_id],
        seed=task_id,
        **RUN_CONFIG,
    )
    print(f"Finished in {time.time() - start:.0f} seconds", flush=True)


if __name__ == "__main__":
    main()
