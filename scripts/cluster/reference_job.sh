#!/bin/bash
# sbatch template for the base-case reference posterior (scripts/calibrate.py --name reference).
# Submit with scripts/cluster/submit.sh, which creates outputs/cluster/logs/ first.
#
# Placeholders to replace before submitting:
#   <account>  the MASSIVE project account (the original analysis used sh30)
#   <repo>     absolute path of this repository on the cluster
#
# 8 NUTS chains, one per core, until split R-hat <= 1.01 and bulk and tail ESS >= 1600 for
# every parameter, from starts spread 0.5 logit units around the optima. Checkpointed:
# resubmitting resumes it. Output: outputs/calibrate/reference/.
#SBATCH --job-name=reference
#SBATCH --account=<account>
#SBATCH --time=48:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=4G
#SBATCH --output=<repo>/outputs/cluster/logs/reference_%j.out
#SBATCH --mail-type=BEGIN,END,FAIL

cd <repo>
export OMP_NUM_THREADS=1
pixi run python scripts/calibrate.py --name reference --chains 8 --ess 1600 --warmup 600 \
    --jitter 0.5 --seed 101 --max-hours 44 --full-runs 0
