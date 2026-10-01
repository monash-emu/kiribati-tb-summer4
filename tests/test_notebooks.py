"""Notebooks are committed clean, and each one executes (slow)."""

from __future__ import annotations

from pathlib import Path

import nbformat
import pytest

from kiribati_tb.paths import REPO_ROOT

NOTEBOOKS = sorted((REPO_ROOT / "notebooks").glob("*.ipynb"))


@pytest.mark.parametrize("path", NOTEBOOKS, ids=lambda p: p.name)
def test_notebook_is_clean(path: Path) -> None:
    nb = nbformat.read(path, as_version=4)
    for cell in nb.cells:
        if cell.cell_type == "code":
            assert not cell.outputs, f"{path.name} has outputs"
            assert cell.execution_count is None, f"{path.name} has execution counts"


@pytest.mark.slow
@pytest.mark.parametrize("path", NOTEBOOKS, ids=lambda p: p.name)
def test_notebook_executes(path: Path) -> None:
    from nbclient import NotebookClient

    nb = nbformat.read(path, as_version=4)
    # Gate notebooks over hours-long runs list the outputs they read; skip until they exist.
    missing = [
        p
        for p in nb.metadata.get("kiribati", {}).get("requires", [])
        if not (REPO_ROOT / p).exists()
    ]
    if missing:
        pytest.skip(f"{path.name} needs {missing}; see its first cell for the commands")
    NotebookClient(nb, timeout=3600, resources={"metadata": {"path": str(path.parent)}}).execute()
