"""K0: summer4 imports and the committed goldens are readable and complete."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import yaml

import summer4
from kiribati_tb.paths import GOLDEN

FIXTURES = (
    "no_transmission",
    "homog_baseline",
    "hetero_baseline",
    "hetero_scenario_3",
    "hetero_scenario_8",
)
YEARS = np.arange(1850.0, 2036.0)


def test_summer4_imports() -> None:
    assert summer4.FlowModel is not None


@pytest.mark.parametrize("fixture", FIXTURES)
def test_golden_shape(fixture: str) -> None:
    folder = GOLDEN / fixture
    comps = pd.read_parquet(folder / "compartments.parquet")
    derived = pd.read_parquet(folder / "derived.parquet")
    params = yaml.safe_load((folder / "params.yaml").read_text())
    assert comps.shape == (len(YEARS), 1 + 160)
    np.testing.assert_array_equal(comps["time"].to_numpy(), YEARS)
    np.testing.assert_array_equal(derived["time"].to_numpy(), YEARS)
    assert params["fixture"] == fixture
    assert np.isfinite(comps.drop(columns="time").to_numpy()).all()
    assert "population" in derived.columns
    assert ("mixing_matrix_distance" in derived.columns) == fixture.startswith("hetero")
