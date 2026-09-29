# Vendored from monash-emu/kiribati_tb_modelling@46a63f0 code/tbh/paths.py.
# Modified: the original finds ``data/`` from the git root of the working directory. The
# vendored copy points at this repository's copied ``data/`` directly so ``make_golden.py``
# does not depend on the directory it is launched from.
from pathlib import Path

REPO_ROOT_PATH = Path(__file__).resolve().parents[2]
OUTPUT_PARENT_FOLDER = REPO_ROOT_PATH / "remote_cluster" / "outputs"
DATA_FOLDER = REPO_ROOT_PATH / "data"
