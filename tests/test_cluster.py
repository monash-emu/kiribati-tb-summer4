"""K6: the cluster drivers reproduce the original's task grids."""

from __future__ import annotations

import sys

from kiribati_tb.calibration import calibration_setup
from kiribati_tb.paths import REPO_ROOT

sys.path.insert(0, str(REPO_ROOT / "scripts" / "cluster"))

import massiverun  # noqa: E402
import massiverun_sas  # noqa: E402


def test_grid_is_the_published_four_by_four() -> None:
    grid = massiverun.build_param_grid()
    assert len(grid) == 16
    assert grid[0] == {
        "clinical_regression_rate": 0.5,
        "infectiousness_loss_rate": 0.5,
        "rel_sus_unreachable": 1.0,
    }
    assert grid[-1]["infectiousness_loss_rate"] == 3.0 and grid[-1]["rel_sus_unreachable"] == 3.0


def test_every_task_builds_a_calibration() -> None:
    for overrides in massiverun.build_param_grid():
        setup = calibration_setup(param_overrides=overrides)
        assert setup.params["rel_sus_unreachable"] == overrides["rel_sus_unreachable"]
    for analysis in massiverun_sas.SA_BY_TASK_ID.values():
        calibration_setup(analysis)
