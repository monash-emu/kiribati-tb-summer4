"""K3: every golden derived output is reproduced, by the same name."""

from __future__ import annotations

import jax
import numpy as np
import pandas as pd
import pytest
import yaml

from summer4 import SavePlan
from kiribati_tb.model import SAVE_TIMES, ModelConfig, compile_model, solve
from kiribati_tb.outputs import build_outputs, run_outputs
from kiribati_tb.paths import GOLDEN
from parity import assert_parity, read_golden

ATOL = 1e-8
# Without transmission the disease states are solver noise (~1e-10 persons, and negative) once
# the seed dies out; per-capita outputs of them are compared at the compartment floor instead.
ATOL_NO_TRANSMISSION = 1e-6
FIXTURES = {
    "no_transmission": ModelConfig(heterogeneous_mixing=False),
    "homog_baseline": ModelConfig(heterogeneous_mixing=False),
    "hetero_baseline": ModelConfig(heterogeneous_mixing=True),
}


def _params(fixture: str) -> dict[str, float]:
    return yaml.safe_load((GOLDEN / fixture / "params.yaml").read_text())["params"]


@pytest.fixture(scope="module", params=sorted(FIXTURES))
def derived(request: pytest.FixtureRequest) -> tuple[str, pd.DataFrame]:
    fixture = request.param
    config = FIXTURES[fixture]
    frame = run_outputs(compile_model(config), build_outputs(config), _params(fixture))
    return fixture, frame


def test_names_are_the_originals(derived: tuple[str, pd.DataFrame]) -> None:
    fixture, frame = derived
    golden = read_golden(fixture, "derived")
    assert sorted(frame.columns) == sorted(golden.columns)


def test_every_output_matches(derived: tuple[str, pd.DataFrame]) -> None:
    fixture, frame = derived
    golden = read_golden(fixture, "derived")
    if fixture == "no_transmission":
        # With no infection the disease states fall below 1e-3 persons within about 15 years of
        # the seed, after which ratios of them (for example % subclinical) are ratios of solver
        # noise in both models. Compare those ratios only while there is disease to divide.
        noise = [c for c in golden.columns if c.startswith(("perc_prev_", "perc_notif"))]
        early = golden["tb_prevalence"] > 1e-3
        assert_parity(frame.loc[early, noise], golden.loc[early, noise], ATOL)
        golden = golden.drop(columns=noise)
    atol = ATOL_NO_TRANSMISSION if fixture == "no_transmission" else ATOL
    report = assert_parity(frame, golden, atol)
    print(fixture, report.head(5).to_string(), sep="\n")


def test_output_set_evaluates_under_jit() -> None:
    config = FIXTURES["hetero_baseline"]
    compiled = compile_model(config)
    outputs = build_outputs(config)
    plan = outputs.plan(SavePlan(ts=SAVE_TIMES))
    targets = ["tb_incidence_per100k", "notifications", "mixing_matrix_distance"]

    @jax.jit
    def run(params: dict[str, float]) -> dict[str, jax.Array]:
        named = outputs.evaluate(solve(compiled, params, plan), params)
        return {name: named[name].values for name in targets}

    params = _params("hetero_baseline")
    got = run(params)
    golden = read_golden("hetero_baseline", "derived")
    for name in targets:
        np.testing.assert_allclose(np.asarray(got[name]), golden[name].to_numpy(), rtol=1e-5)
