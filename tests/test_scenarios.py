"""K5: screening scenarios match the scenario goldens."""

from __future__ import annotations

import pandas as pd
import pytest
import yaml

from kiribati_tb.model import ModelConfig, compile_model, run_compartments
from kiribati_tb.outputs import build_outputs, run_outputs
from kiribati_tb.paths import GOLDEN
from kiribati_tb.scenarios import SCENARIOS, SCENARIOS_BY_ID
from parity import assert_parity, read_golden

FIXTURES = {"hetero_scenario_3": "scenario_3", "hetero_scenario_8": "scenario_8"}


def test_scenario_ids_and_names_are_the_originals() -> None:
    ids = [s.sc_id for s in SCENARIOS]
    assert ids == [f"scenario_{i}" for i in (1, 2, 3, 6, 7, 8, 16, 17, 18)]
    assert SCENARIOS_BY_ID["scenario_3"].sc_name == "3. PEARL / Vhigh"
    assert SCENARIOS_BY_ID["scenario_8"].sc_name == "8. CXR-TST / 84%"
    assert SCENARIOS_BY_ID["scenario_17"].sc_name_2 == "Drop Xpert & TST | 75%"


@pytest.fixture(scope="module", params=sorted(FIXTURES))
def scenario_run(request: pytest.FixtureRequest) -> tuple[str, pd.DataFrame, pd.DataFrame]:
    fixture = request.param
    params = yaml.safe_load((GOLDEN / fixture / "params.yaml").read_text())["params"]
    config = ModelConfig(screening=SCENARIOS_BY_ID[FIXTURES[fixture]].programs)
    compiled = compile_model(config)
    return (
        fixture,
        run_compartments(compiled, params),
        run_outputs(compiled, build_outputs(config), params),
    )


def test_compartments_match_golden(scenario_run: tuple[str, pd.DataFrame, pd.DataFrame]) -> None:
    fixture, compartments, _ = scenario_run
    report = assert_parity(compartments, read_golden(fixture, "compartments"), 1e-6)
    print(fixture, report.head(5).to_string(), sep="\n")


def test_outputs_match_golden(scenario_run: tuple[str, pd.DataFrame, pd.DataFrame]) -> None:
    fixture, _, derived = scenario_run
    golden = read_golden(fixture, "derived")
    assert sorted(derived.columns) == sorted(golden.columns)
    report = assert_parity(derived, golden, 1e-8)
    print(fixture, report.head(5).to_string(), sep="\n")
