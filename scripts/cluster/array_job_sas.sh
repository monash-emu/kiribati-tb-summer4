#!/bin/bash
# sbatch template for scripts/cluster/massiverun_sas.py. Submit with
# scripts/cluster/submit.sh, which creates outputs/cluster/logs/ first (SLURM does not
# create the --output directory).
#
# Placeholders to replace before submitting:
#   <account>  the MASSIVE project account (the original analysis used sh30)
#   <repo>     absolute path of this repository on the cluster
#
# One task = one calibration (8 NUTS chains, one per core: --cpus-per-task must equal CHAINS
# in massiverun.py) followed by the scenario full runs. Tasks checkpoint as they go;
# resubmitting this file resumes unfinished tasks. See README.md, "Cluster runs", for the
# wall-time estimate behind --time.
#SBATCH --job-name=sas
#SBATCH --account=<account>
#SBATCH --time=48:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=4G
#SBATCH --output=<repo>/outputs/cluster/logs/%A_%a.out
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --array=1-3

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $@"
}
log "Starting job"

cd <repo>
mkdir -p outputs/cluster/logs

# The pixi environment replaces the original's conda environment; install it once with
# `pixi install` on a login node. Keep XLA to the cores SLURM gave this task.
export OMP_NUM_THREADS=1
# Unbuffered output so progress bars and checkpoint lines reach the log as they happen.
export PYTHONUNBUFFERED=1

# Stagger task starts so sixteen tasks do not resolve the pixi environment at once.
sleep $((SLURM_ARRAY_TASK_ID * 10))

# Task 1 (tpt_60) reuses the base-case posterior instead of recalibrating: submit with
# BASE_POSTERIOR=<path to a base-case idata.nc> (and BASE_BURN_IN=<draws per chain> if it
# still contains warmup draws), e.g.
#   BASE_POSTERIOR=outputs/reference/r2/nuts_td8/mcmc/idata.nc sbatch scripts/cluster/array_job_sas.sh
# Without it, task 1 exits at once with a message; tasks 2 and 3 calibrate as before.
pixi run python scripts/cluster/massiverun_sas.py $SLURM_ARRAY_JOB_ID $SLURM_ARRAY_TASK_ID \
    ${BASE_POSTERIOR:+--base-posterior "$BASE_POSTERIOR"} \
    ${BASE_BURN_IN:+--base-burn-in "$BASE_BURN_IN"}

log "Job completed"
