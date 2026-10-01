#!/bin/bash
# sbatch template for one reference calibration arm (scripts/reference_run.py). Do not submit it
# directly: scripts/cluster/submit_reference.sh sets ARM, the job name, --cpus-per-task, --time
# and the log path per arm, and creates outputs/cluster/logs/ first (SLURM does not).
#
# Placeholder to replace before submitting (or export ACCOUNT=... for submit_reference.sh):
#   <account>  the MASSIVE project account (the original analysis used sh30)
#
# The job runs in the directory it was submitted from (the repository root). Each arm writes
# outputs/reference/<arm>/ and resumes from it when resubmitted: --max-hours (2 h under the
# wall time) stops it cleanly between warmup rounds or sampling chunks, all of them saved.
#SBATCH --account=<account>
#SBATCH --ntasks=1
#SBATCH --mem-per-cpu=4G
#SBATCH --mail-type=END,FAIL

set -euo pipefail
: "${ARM:?set ARM (nuts_td8, nuts_td5, sa or ess); use submit_reference.sh}"
: "${MAX_HOURS:=46}"

cd "${SLURM_SUBMIT_DIR}"
mkdir -p outputs/cluster/logs
echo "[$(date '+%Y-%m-%d %H:%M:%S')] arm=${ARM} job=${SLURM_JOB_ID} cpus=${SLURM_CPUS_PER_TASK:-?} host=$(hostname)"

# Unbuffered output so progress bars and checkpoint lines reach the log as they happen.
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1

pixi run python scripts/reference_run.py --arm "${ARM}" --max-hours "${MAX_HOURS}"

echo "[$(date '+%Y-%m-%d %H:%M:%S')] arm=${ARM} done"
