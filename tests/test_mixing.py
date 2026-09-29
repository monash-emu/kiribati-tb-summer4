"""K2: the vectorised yearly mixing stack equals the original builder and is differentiable."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
import pytest

from kiribati_tb.mixing import (
    START_YEAR,
    YEARS,
    canberra_distance,
    contact_rates,
    default_mixing_inputs,
    read_conmat_matrix,
    spectral_radius,
    yearly_matrices,
)
from kiribati_tb.paths import GOLDEN

N_BANDS = 8


@pytest.fixture(scope="module")
def golden() -> pd.DataFrame:
    return pd.read_parquet(GOLDEN / "mixing.parquet")


def _params(bg: float, spread: float, pc: float) -> dict[str, float]:
    return {"bg_mixing": bg, "a_spread": spread, "pc_strength": pc}


def test_stack_equals_original_builder(golden: pd.DataFrame) -> None:
    """Five parameter draws × ten times, including within-year times and the 1950 data start."""
    columns = [f"m_{i}_{j}" for i in range(N_BANDS) for j in range(N_BANDS)]
    for (bg, spread, pc), rows in golden.groupby(["bg_mixing", "a_spread", "pc_strength"]):
        stack = np.asarray(yearly_matrices(_params(bg, spread, pc)))
        years = np.floor(rows["time"].to_numpy()).astype(int) - START_YEAR
        expected = rows[columns].to_numpy().reshape(-1, N_BANDS, N_BANDS)
        np.testing.assert_allclose(stack[years], expected, rtol=1e-12, atol=1e-15)


def test_eigvalsh_radius_equals_eigvals_radius() -> None:
    """``rho(S diag N)`` from the symmetric form equals the general eigenvalue routine."""
    inputs = default_mixing_inputs()
    rng = np.random.default_rng(20260929)
    populations = jnp.asarray(inputs.populations)
    for _ in range(20):
        params = _params(rng.uniform(0.01, 0.05), rng.uniform(2.0, 15.0), rng.uniform(0.01, 1.0))
        contacts = contact_rates(params, inputs)
        unnormalised = np.asarray(contacts * populations[:, None, :])
        general = np.max(np.abs(np.linalg.eigvals(unnormalised)), axis=-1)
        np.testing.assert_allclose(
            np.asarray(spectral_radius(contacts, populations)), general, rtol=1e-12
        )


def test_stack_is_differentiable_in_every_mixing_parameter() -> None:
    def total(values: jnp.ndarray) -> jnp.ndarray:
        params = dict(zip(("bg_mixing", "a_spread", "pc_strength"), values))
        return jnp.sum(yearly_matrices(params) ** 2)

    grad = np.asarray(jax.grad(total)(jnp.asarray([0.03, 8.5, 0.5])))
    assert np.all(np.isfinite(grad))
    assert np.all(grad != 0.0)


def test_build_does_not_unroll_over_band_pairs() -> None:
    """One batched build: the jaxpr is a few dozen equations, not 36 pair loops."""
    jaxpr = jax.make_jaxpr(yearly_matrices)(_params(0.03, 8.5, 0.5))
    assert len(jaxpr.jaxpr.eqns) < 60


def test_distance_to_conmat_by_year() -> None:
    """Canberra distance to the conmat matrix, per year, as the original computed value."""
    stack = yearly_matrices(_params(0.03, 8.5, 0.505))
    distance = np.asarray(canberra_distance(stack, read_conmat_matrix()))
    derived = pd.read_parquet(GOLDEN / "hetero_baseline" / "derived.parquet").set_index("time")
    np.testing.assert_allclose(
        distance, derived.loc[YEARS.astype(float), "mixing_matrix_distance"], rtol=1e-12
    )
