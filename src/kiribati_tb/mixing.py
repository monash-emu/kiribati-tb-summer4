"""Time-varying age mixing: one normalised matrix per model year, built before the solve.

Ports ``tbh/age_mixing.py``. The original evaluates ``build_mixing_matrix(bg_mixing, a_spread,
pc_strength, t)`` inside the vector field, with a Python loop over the 36 age-band pairs and a
``vmap`` per pair, and normalises by the spectral radius from ``jnp.linalg.eigvals`` (not
reverse-differentiable). Everything in it depends on ``t`` only through ``int(t)``, so the port
builds the whole ``(n_years, 8, 8)`` stack once per parameter set, vectorised, in the compiled
model's ``prepare_fn``; the vector field gathers the year's matrix with
``Lookup(Param("mixing_stack"), floor(Time() - START_YEAR))``.

For band lower bounds ``i, j`` and single ages ``a in i``, ``b in j`` with within-band population
weights ``w``, the per-pair contact rate is::

    S[i, j] = bg_mixing
              + sum_ab w_a w_b exp(-|a - b| / a_spread) / a_spread
              + pc_strength * sum_ab w_a w_b P(birth year of min(a, b), gap |a - b|)

where ``P`` is the normalised fertility-by-age-of-mother table. ``C = S diag(N)`` is then
normalised by its spectral radius. ``S`` is symmetric and ``N > 0``, so ``C`` is similar to the
symmetric ``diag(sqrt N) S diag(sqrt N)`` and ``jnp.linalg.eigvalsh`` gives the same radius with
gradients.

The parent-child term does not depend on any parameter, so it is summed on the host once.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cache
from typing import Any

import jax.numpy as jnp
import numpy as np
import pandas as pd

from kiribati_tb.demography import AGE_GROUPS, ISO3, MAX_AGE, Demography, load_demography
from kiribati_tb.paths import DATA

START_YEAR = 1850
END_YEAR = 2035
YEARS: np.ndarray = np.arange(START_YEAR, END_YEAR + 1)
MIXING_PARAMS: tuple[str, ...] = ("bg_mixing", "a_spread", "pc_strength")


def band_index(age_groups: tuple[str, ...] = AGE_GROUPS, max_age: int = MAX_AGE) -> np.ndarray:
    """Band index of every single age ``0 .. max_age``."""
    lbs = np.asarray([int(a) for a in age_groups])
    return np.searchsorted(lbs, np.arange(max_age + 1), side="right") - 1


def age_weight_table(single_age: pd.DataFrame, age_groups: tuple[str, ...]) -> pd.DataFrame:
    """Each single age's share of its band's population, by year (``build_age_weight_lookup``)."""
    pivot = single_age.pivot(index="Time", columns="Age", values="Pop").fillna(0.0)
    bands = band_index(age_groups, int(pivot.columns.max()))
    totals = pivot.T.groupby(bands).sum().T  # (years, bands)
    denom = totals.replace(0.0, 1.0).to_numpy()[:, bands]
    return pd.DataFrame(pivot.to_numpy() / denom, index=pivot.index, columns=pivot.columns)


def fertility_table(iso3: str = ISO3) -> pd.DataFrame:
    """Fertility by age of mother, normalised to sum to one each year (``build_agegap_lookup``)."""
    fert = pd.read_csv(DATA / f"un_fertility_rates_{iso3}.csv", index_col=0)
    fert = fert.div(fert.sum(axis=1), axis=0)
    fert.columns = fert.columns.astype(int)
    return fert


def _clamped_rows(table: pd.DataFrame, years: np.ndarray) -> np.ndarray:
    """Rows of ``table`` for ``years``, clamping to the first and last year (as the original)."""
    idx = np.clip(years - int(table.index.min()), 0, len(table.index) - 1)
    return table.to_numpy()[idx]


@dataclass(frozen=True)
class MixingInputs:
    """Constant arrays the per-parameter mixing build needs, for years ``YEARS``.

    Attributes:
        band_weights: ``(n_years, n_bands, n_ages)`` within-band population weight of each
            single age, zero outside its band.
        gap: ``(n_ages, n_ages)`` absolute age gap.
        parent_child: ``(n_years, n_bands, n_bands)`` parameter-free parent-child sum.
        populations: ``(n_years, n_bands)`` band populations.
    """

    band_weights: np.ndarray
    gap: np.ndarray
    parent_child: np.ndarray
    populations: np.ndarray


def mixing_inputs(
    demog: Demography, iso3: str = ISO3, age_groups: tuple[str, ...] = AGE_GROUPS
) -> MixingInputs:
    """Prepare the constant arrays of the mixing build, on the host."""
    weights = _clamped_rows(age_weight_table(demog.single_age, age_groups), YEARS)
    n_ages = weights.shape[1]
    bands = band_index(age_groups, n_ages - 1)
    onehot = bands[None, :] == np.arange(len(age_groups))[:, None]  # (bands, ages)
    band_weights = weights[:, None, :] * onehot[None, :, :]

    ages = np.arange(n_ages)
    gap = np.abs(ages[:, None] - ages[None, :])
    child = np.minimum(ages[:, None], ages[None, :])
    fert = fertility_table(iso3)
    year0, age0 = int(fert.index.min()), int(fert.columns.min())
    n_fert_years, n_fert_ages = fert.shape
    gap_idx = gap - age0
    valid = (gap_idx >= 0) & (gap_idx < n_fert_ages)
    gap_idx = np.clip(gap_idx, 0, n_fert_ages - 1)
    fert_values = fert.to_numpy()
    parent_child = np.empty((len(YEARS), len(age_groups), len(age_groups)))
    for k, year in enumerate(YEARS):
        year_idx = np.clip(year - child - year0, 0, n_fert_years - 1)
        prob = np.where(valid, fert_values[year_idx, gap_idx], 0.0)
        parent_child[k] = band_weights[k] @ prob @ band_weights[k].T

    populations = _clamped_rows(demog.grouped[list(age_groups)], YEARS)
    return MixingInputs(band_weights, gap.astype(float), parent_child, populations)


@cache
def default_mixing_inputs() -> MixingInputs:
    """:func:`mixing_inputs` for the model's country and age bands."""
    return mixing_inputs(load_demography())


def contact_rates(params: Any, inputs: MixingInputs) -> Any:
    """Symmetric per-pair contact rates ``S``, shape ``(n_years, n_bands, n_bands)``."""
    spread = params["a_spread"]
    kernel = jnp.exp(-jnp.asarray(inputs.gap) / spread) / spread
    weights = jnp.asarray(inputs.band_weights)
    assortative = jnp.einsum("yia,ab,yjb->yij", weights, kernel, weights)
    return (
        params["bg_mixing"] + assortative + params["pc_strength"] * jnp.asarray(inputs.parent_child)
    )


def spectral_radius(contacts: Any, populations: Any) -> Any:
    """Spectral radius of ``S diag(N)`` from the symmetric ``diag(sqrt N) S diag(sqrt N)``."""
    root = jnp.sqrt(populations)
    symmetric = root[..., :, None] * contacts * root[..., None, :]
    return jnp.max(jnp.abs(jnp.linalg.eigvalsh(symmetric)), axis=-1)


def yearly_matrices(params: Any, inputs: MixingInputs | None = None) -> Any:
    """Normalised mixing matrices ``C / rho(C)`` for every year in ``YEARS``.

    ``C[i, j] = S[i, j] * N[j]``: rows are the infected band, columns the infecting band, as in
    summer2 (``mixing_matrix @ (I / N ** exp)``).
    """
    inputs = default_mixing_inputs() if inputs is None else inputs
    contacts = contact_rates(params, inputs)
    populations = jnp.asarray(inputs.populations)
    unnormalised = contacts * populations[:, None, :]
    return unnormalised / spectral_radius(contacts, populations)[:, None, None]


def read_conmat_matrix(iso3: str = ISO3, age_groups: tuple[str, ...] = AGE_GROUPS) -> np.ndarray:
    """The ``conmat`` synthetic contact matrix for the model's bands, spectral radius one."""
    frame = pd.read_csv(DATA / "Rscript" / f"conmat_all_{iso3}.csv", index_col=0)
    lower = lambda label: label[1:].split(",")[0]  # noqa: E731 - "[15,18)" -> "15"
    breaks = {lower(s) for s in frame["age_group_from"].unique()}
    if breaks != set(age_groups):
        raise ValueError("conmat age bands do not match the model's")
    matrix = np.zeros((len(age_groups), len(age_groups)))
    for src, dst, contacts in frame[["age_group_from", "age_group_to", "contacts"]].itertuples(
        index=False
    ):
        matrix[age_groups.index(lower(src)), age_groups.index(lower(dst))] = contacts
    return matrix / np.max(np.abs(np.linalg.eigvals(matrix)))


def canberra_distance(m1: Any, m2: Any) -> Any:
    """Canberra distance between matrices, over their last two axes."""
    return jnp.sum(jnp.abs(m1 - m2) / (jnp.abs(m1) + jnp.abs(m2) + 1e-10), axis=(-2, -1))
