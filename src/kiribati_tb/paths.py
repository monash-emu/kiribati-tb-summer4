"""Repository paths."""

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA = REPO_ROOT / "data"
GOLDEN = REPO_ROOT / "tests" / "golden"
