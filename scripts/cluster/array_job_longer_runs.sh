#!/bin/bash
# sbatch template for scripts/cluster/massiverun.py. Set the account, paths and mail first.
#SBATCH --job-name=longer_runs
#SBATCH --account=<account>
#SBATCH --time=48:00:00
#SBATCH --ntasks=1
#SBATCH --mem-per-cpu=8G
#SBATCH --cpus-per-task=8
#SBATCH --output=<repo>/outputs/cluster/logs/%A_%a.out
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --array=1-16

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $@"
}
log "Starting job"

cd <repo>

# The pixi environment replaces the original's conda environment; install it once with
# `pixi install` on a login node.

sleep $((SLURM_ARRAY_TASK_ID * 30))

pixi run python scripts/cluster/massiverun.py $SLURM_ARRAY_JOB_ID $SLURM_ARRAY_TASK_ID

log "Job completed"
