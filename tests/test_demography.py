"""K1: the port's time-varying inputs equal the original summer2 functions.

``tests/golden/demography.parquet`` evaluates the original's per-age death rates, births entry
rate, treatment success, negative treatment outcomes and passive detection rate every half
year from 1935 to 2035. Each is compared with the port's rate expression evaluated through a
probe model: one exit flow per per-age quantity on a one-person-per-age map, one entry flow per
scalar quantity.
"""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
import pytest
import yaml

from summer4 import EntryFlow, ExitFlow, FlowModel, Property, PropertyMap
from kiribati_tb.demography import AGE_GROUPS, load_demography
from kiribati_tb.model import (
    AGE,
    births_entry_rate,
    death_rate,
    passive_detection_rate,
    treatment_outcomes,
    treatment_success,
)
from kiribati_tb.paths import GOLDEN

RTOL = 1e-9


@pytest.fixture(scope="module")
def golden() -> pd.DataFrame:
    return pd.read_parquet(GOLDEN / "demography.parquet")


@pytest.fixture(scope="module")
def probe(golden: pd.DataFrame) -> dict[str, np.ndarray]:
    """Evaluate every port expression at the golden times."""
    demog = load_demography()
    background = death_rate(demog)
    success = treatment_success()
    outcomes = treatment_outcomes(background, success)
    one = Property("one", ("x",))
    age_map = PropertyMap.from_property(one).stratify(AGE)
    model = FlowModel(age_map)
    for name, rate in (
        ("death_rate", background),
        ("tx_relapse", outcomes.relapse),
        ("tx_death", outcomes.death),
    ):
        model.add_flow(ExitFlow(name, one["x"], rate))
    for name, rate in (
        ("entry_rate", births_entry_rate(demog)),
        ("tx_success", success),
        ("passive_detection_rate", passive_detection_rate()),
    ):
        model.add_flow(EntryFlow(name, one["x"] & AGE[AGE_GROUPS[0]], rate))
    compiled = model.compile()
    params = yaml.safe_load((GOLDEN / "hetero_baseline" / "params.yaml").read_text())["params"]
    y = jnp.ones(age_map.size)

    def flows_at(t: Any) -> dict[str, Any]:
        return dict(compiled.observe(t, y, params).flows)

    flows = jax.jit(jax.vmap(flows_at))(jnp.asarray(golden["time"].to_numpy()))
    return {name: np.asarray(value) for name, value in flows.items()}


@pytest.mark.parametrize("name", ["death_rate", "tx_relapse", "tx_death"])
def test_per_age_series(name: str, golden: pd.DataFrame, probe: dict[str, np.ndarray]) -> None:
    expected = golden[[f"{name}_{age}" for age in AGE_GROUPS]].to_numpy()
    np.testing.assert_allclose(probe[name], expected, rtol=RTOL, atol=1e-12)


@pytest.mark.parametrize("name", ["entry_rate", "tx_success", "passive_detection_rate"])
def test_scalar_series(name: str, golden: pd.DataFrame, probe: dict[str, np.ndarray]) -> None:
    np.testing.assert_allclose(probe[name][:, 0], golden[name].to_numpy(), rtol=RTOL, atol=1e-12)
