#!/bin/bash
# Submit the base-case reference calibrations to SLURM, one independent job per arm.
#
#   scripts/cluster/submit_reference.sh                 # all four arms
#   scripts/cluster/submit_reference.sh nuts_td5 ess    # just these
#   ACCOUNT=sh30 scripts/cluster/submit_reference.sh    # account without editing the template
#   SBATCH=echo scripts/cluster/submit_reference.sh     # print the sbatch commands only
#
# Resubmitting an arm resumes it from outputs/reference/<arm>/. Logs:
# outputs/cluster/logs/ref_<arm>_<jobid>.out. Compare with scripts/compare_reference.py.
set -euo pipefail
cd "$(dirname "$0")/../.."
mkdir -p outputs/cluster/logs

# arm -> "cpus walltime budget-hours". The budget is this job's wall-clock limit inside
# reference_run.py: it does not start a round or chunk that would overrun it, so 2 h below the
# SLURM wall time covers start-up and the final save.
resources() {
    case "$1" in
        nuts_td8 | nuts_td5 | sa) echo "8 48:00:00 46" ;;
        ess) echo "4 48:00:00 46" ;;
        *) return 1 ;;
    esac
}

arms=("$@")
if [ ${#arms[@]} -eq 0 ]; then
    arms=(nuts_td8 nuts_td5 sa ess)
fi

account_args=()
if [ -n "${ACCOUNT:-}" ]; then
    account_args=(--account="${ACCOUNT}")
elif [ "${SBATCH:-sbatch}" = sbatch ] && grep -q "<account>" scripts/cluster/reference_arm.sh; then
    echo "Set ACCOUNT=<project> or replace <account> in scripts/cluster/reference_arm.sh" >&2
    exit 1
fi

for arm in "${arms[@]}"; do
    if ! spec=$(resources "${arm}"); then
        echo "Unknown arm ${arm}; expected nuts_td8, nuts_td5, sa or ess" >&2
        exit 1
    fi
    read -r cpus walltime budget <<< "${spec}"
    ${SBATCH:-sbatch} ${account_args[@]+"${account_args[@]}"} \
        --job-name="ref_${arm}" \
        --cpus-per-task="${cpus}" \
        --time="${walltime}" \
        --output="outputs/cluster/logs/ref_${arm}_%j.out" \
        --export=ALL,ARM="${arm}",MAX_HOURS="${budget}" \
        scripts/cluster/reference_arm.sh
done
