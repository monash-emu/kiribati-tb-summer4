#!/bin/bash
# Submit an array job from the repository root, creating the log folder first: SLURM opens the
# --output file before the job starts and does not create missing directories.
#
#   scripts/cluster/submit.sh scripts/cluster/array_job_longer_runs.sh
#   scripts/cluster/submit.sh scripts/cluster/array_job_sas.sh
#   scripts/cluster/submit.sh scripts/cluster/reference_job.sh
#
# Resubmitting the same template resumes every unfinished task from its checkpoints.
set -euo pipefail
cd "$(dirname "$0")/../.."
mkdir -p outputs/cluster/logs
sbatch "$@"
