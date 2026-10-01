"""Repository paths."""

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA = REPO_ROOT / "data"
GOLDEN = REPO_ROOT / "tests" / "golden"
OUTPUTS = REPO_ROOT / "outputs"


def run_name(run: str | None = None) -> str:
    """The reference-run set: ``run`` if given, else the ``RUN`` environment variable, else ``""``.

    ``scripts/cluster/submit_reference.sh`` takes the same ``RUN`` to name a set of runs.
    """
    return os.environ.get("RUN", "") if run is None else run


def reference_root(run: str | None = None) -> Path:
    """Where a set of reference runs lives: ``outputs/reference[/<run>]``."""
    name = run_name(run)
    return OUTPUTS / "reference" / name if name else OUTPUTS / "reference"


def compare_dir(kind: str, run: str | None = None) -> Path:
    """Comparison output for a set of runs: ``outputs/<kind>[/<run>]``."""
    name = run_name(run)
    return OUTPUTS / kind / name if name else OUTPUTS / kind
