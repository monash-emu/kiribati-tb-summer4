#!/bin/bash
# Local smoke test of every reference arm (tiny settings, minutes each), then a resume check:
# each arm is run a second time with a larger draw cap and must continue from its checkpoint.
#
#   scripts/cluster/smoke_reference.sh [arm ...]      # logs: outputs/logs/smoke_<arm>.log
set -uo pipefail
cd "$(dirname "$0")/../.."
mkdir -p outputs/logs
arms=("$@")
if [ ${#arms[@]} -eq 0 ]; then
    arms=(nuts_td8 nuts_td5 sa ess)
fi
export PYTHONUNBUFFERED=1
for arm in "${arms[@]}"; do
    log="outputs/logs/smoke_${arm}.log"
    echo "[smoke] ${arm}: first run" | tee "${log}"
    pixi run python scripts/reference_run.py --arm "${arm}" --smoke >> "${log}" 2>&1
    echo "[smoke] ${arm}: first run exit $?" | tee -a "${log}"
    case "${arm}" in
        nuts_*) more=12 ;;
        *) more=60 ;;
    esac
    echo "[smoke] ${arm}: resume with --max-samples ${more}" | tee -a "${log}"
    pixi run python scripts/reference_run.py --arm "${arm}" --smoke --max-samples "${more}" >> "${log}" 2>&1
    echo "[smoke] ${arm}: resume exit $?" | tee -a "${log}"
done
