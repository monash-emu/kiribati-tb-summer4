"""K2: transmission and mixing match the baseline goldens.

``homog_baseline`` has one mixing pool; ``hetero_baseline`` uses the time-varying age mixing
matrix. Neither has screening.
"""

from __future__ import annotations

import pandas as pd
import pytest
import yaml

from kiribati_tb.model import ModelConfig, compile_model, run_compartments
from kiribati_tb.paths import GOLDEN
from parity import assert_parity, read_golden

ATOL = 1e-6  # persons
FIXTURES = {
    "homog_baseline": ModelConfig(heterogeneous_mixing=False),
    "hetero_baseline": ModelConfig(heterogeneous_mixing=True),
}


@pytest.mark.parametrize("fixture", sorted(FIXTURES))
def test_compartments_match_golden(fixture: str) -> None:
    params = yaml.safe_load((GOLDEN / fixture / "params.yaml").read_text())["params"]
    got = run_compartments(compile_model(FIXTURES[fixture]), params)
    report = assert_parity(got, read_golden(fixture, "compartments"), ATOL)
    print(fixture, report.head(5).to_string(), sep="\n")
