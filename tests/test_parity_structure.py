"""K1: structure and demography match the ``no_transmission`` golden.

With ``raw_transmission_rate = 0`` and homogeneous mixing the original has no infection. The
port's K1 model has no infection flows, so this fixture checks the map, ageing, the care
cascade, deaths recycled to age 0 and births before transmission is added.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import yaml

from kiribati_tb.model import compartment_labels, compile_model, pmap, run_compartments
from kiribati_tb.paths import GOLDEN
from parity import assert_parity, read_golden

FIXTURE = "no_transmission"
ATOL = 1e-6  # persons


@pytest.fixture(scope="module")
def compartments() -> pd.DataFrame:
    params = yaml.safe_load((GOLDEN / FIXTURE / "params.yaml").read_text())["params"]
    return run_compartments(compile_model(), params)


def test_map_has_the_original_compartments() -> None:
    labels = compartment_labels(pmap())
    golden = read_golden(FIXTURE, "compartments")
    assert len(labels) == 160
    assert set(labels) == set(golden.columns)


def test_initial_population_is_exact(compartments: pd.DataFrame) -> None:
    golden = read_golden(FIXTURE, "compartments")
    np.testing.assert_array_equal(
        compartments.loc[1850.0, golden.columns].to_numpy(), golden.loc[1850.0].to_numpy()
    )


def test_compartments_match_golden(compartments: pd.DataFrame) -> None:
    report = assert_parity(compartments, read_golden(FIXTURE, "compartments"), ATOL)
    print(report.head(10).to_string())
