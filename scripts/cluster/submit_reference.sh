#!/bin/bash
# Submit the base-case reference calibrations to SLURM, one independent job per arm.
#
#   scripts/cluster/submit_reference.sh                 # all four arms
#   scripts/cluster/submit_reference.sh nuts_td5 ess    # just these
#   ACCOUNT=sh30 scripts/cluster/submit_reference.sh    # account without editing the template
#   SBATCH=echo scripts/cluster/submit_reference.sh     # print the sbatch commands only
#   RUN=r2 ACCOUNT=sh30 scripts/cluster/submit_reference.sh   # a separate set of runs
#
# Without RUN, each arm writes outputs/reference/<arm>/ and logs to
# outputs/cluster/logs/ref_<arm>_<jobid>.out. With RUN=<name>, it writes
# outputs/reference/<name>/<arm>/ and logs to outputs/cluster/logs/ref_<name>_<arm>_<jobid>.out,
# so a new RUN starts fresh beside earlier runs. Resubmitting with the same RUN (or none) resumes
# from that folder. Compare with scripts/compare_reference.py.
set -euo pipefail
cd "$(dirname "$0")/../.."
mkdir -p outputs/cluster/logs

# arm -> "cpus walltime budget-hours". The budget is this job's wall-clock limit inside
# reference_run.py: it does not start a round or chunk that would overrun it, so 2 h below the
# SLURM wall time covers start-up and the final save.
resources() {
    case "$1" in
        nuts_td8 | nuts_td5 | sa) echo "8 48:00:00 46" ;;
        ess) echo "2 48:00:00 46" ;;
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

run="${RUN:-}"
if [ -n "${run}" ] && ! [[ "${run}" =~ ^[A-Za-z0-9._-]+$ ]]; then
    echo "RUN must be letters, digits, '.', '_' or '-'" >&2
    exit 1
fi

for arm in "${arms[@]}"; do
    if ! spec=$(resources "${arm}"); then
        echo "Unknown arm ${arm}; expected nuts_td8, nuts_td5, sa or ess" >&2
        exit 1
    fi
    read -r cpus walltime budget <<< "${spec}"
    if [ -n "${run}" ]; then
        tag="${run}_${arm}"
        out="outputs/reference/${run}/${arm}"
    else
        tag="${arm}"
        out="outputs/reference/${arm}"
    fi
    ${SBATCH:-sbatch} ${account_args[@]+"${account_args[@]}"} \
        --job-name="ref_${tag}" \
        --cpus-per-task="${cpus}" \
        --time="${walltime}" \
        --output="outputs/cluster/logs/ref_${tag}_%j.out" \
        --export=ALL,ARM="${arm}",MAX_HOURS="${budget}",OUT="${out}" \
        scripts/cluster/reference_arm.sh
done
