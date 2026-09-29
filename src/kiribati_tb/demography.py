"""Demographic inputs: UN population and mortality, prepared host-side.

Ports ``tbh/demographic_tools.py``. Everything here is data preparation that runs once, in
pandas and NumPy, and hands summer4 constant arrays: a per-age death-rate table, the
population-growth entry rate, and the lookups the mixing builder gathers from.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cache

import numpy as np
import pandas as pd

from summer4 import Property
from summer4.data import Data, TableData

from kiribati_tb.paths import DATA

AGE_GROUPS: tuple[str, ...] = ("0", "3", "5", "10", "15", "18", "40", "65")
MAX_AGE = 120
ISO3 = "KIR"
START_TIME = 1850.0
END_TIME = 2035.0
# 40,483 people in South Tarawa in 2023, excluding Betio, Bairiki and Nanikai, out of a total
# Kiribati population of 119,438 (DEFAULT_MODEL_CONFIG in the original runner_tools.py).
POP_SCALING = 40483.0 / 119438.0


def _group_bounds(age_groups: tuple[str, ...], max_age: int) -> list[tuple[int, int, str]]:
    lbs = [int(a) for a in age_groups]
    return [
        (lb, (lbs[i + 1] - 1) if i < len(lbs) - 1 else max_age, age_groups[i])
        for i, lb in enumerate(lbs)
    ]


@cache
def _un_population(iso3: str) -> pd.DataFrame:
    frame = pd.read_csv(DATA / "un_population.csv")
    return frame[frame["ISO3_code"] == iso3][["Time", "AgeGrp", "PopTotal"]]


def get_population_over_time(
    iso3: str = ISO3,
    age_groups: tuple[str, ...] = AGE_GROUPS,
    max_age: int = MAX_AGE,
    scaling_factor: float = POP_SCALING,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Single-age and grouped population by year, as ``get_population_over_time``.

    UN five-year groups are spread evenly over their single ages; ``"100+"`` runs to
    ``max_age``.

    Returns:
        ``(single_age, grouped)``: ``single_age`` has columns ``Time``, ``Age``, ``Pop``;
        ``grouped`` is indexed by ``Time`` with one column per age group.
    """
    pop = _un_population(iso3)
    bounds = [
        (int(g[:-1]), max_age) if g.endswith("+") else tuple(int(x) for x in g.split("-"))
        for g in pop["AgeGrp"].astype(str)
    ]
    a0 = np.array([lo for lo, _ in bounds])
    a1 = np.array([hi for _, hi in bounds])
    n_ages = a1 - a0 + 1
    per_age = pop["PopTotal"].to_numpy() * 1000.0 * scaling_factor / n_ages
    single = pd.DataFrame(
        {
            "Time": np.repeat(pop["Time"].to_numpy(), n_ages),
            "Age": np.concatenate([np.arange(lo, hi + 1) for lo, hi in zip(a0, a1)]),
            "Pop": np.repeat(per_age, n_ages),
        }
    )
    grouped = pd.DataFrame(
        {
            label: single[(single["Age"] >= lb) & (single["Age"] <= ub)]
            .groupby("Time")["Pop"]
            .sum()
            for lb, ub, label in _group_bounds(age_groups, max_age)
        }
    ).sort_index()
    grouped.index.name = "Time"
    return single, grouped


def get_death_rates_by_age(
    grouped: pd.DataFrame,
    iso3: str = ISO3,
    age_groups: tuple[str, ...] = AGE_GROUPS,
    start_time: float = START_TIME,
) -> pd.DataFrame:
    """Per-group death rates by year (deaths / population), as ``get_death_rates_by_age``."""
    bins = [int(a) for a in age_groups]
    mort = pd.read_csv(DATA / "un_mortality.csv")
    mort = mort[(mort["ISO3_code"] == iso3) & (mort["Time"] >= start_time)]
    mort = mort[["Time", "AgeGrp", "DeathTotal"]].copy()
    mort["DeathTotal"] *= 1000.0
    ages = mort["AgeGrp"].astype(str).str.rstrip("+").astype(int).to_numpy()
    group_idx = np.searchsorted(np.asarray(bins), ages, side="right") - 1
    mort["age_group"] = np.asarray(bins)[group_idx]
    mort = mort.groupby(["Time", "age_group"], as_index=False).agg({"DeathTotal": "sum"})
    mort = mort.pivot(index="Time", columns="age_group", values="DeathTotal")
    mort.columns = mort.columns.astype(str)
    return mort.div(grouped, axis=0).dropna()[list(age_groups)]


def population_entry(grouped: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Knots for the births entry rate: yearly population increments, zero the year before.

    The original linearly fills missing years, differences successive totals, and prepends a
    zero increment one year before the first difference.
    """
    total = grouped.sum(axis=1)
    full_index = pd.Index(range(int(total.index.min()), int(total.index.max()) + 1))
    increments = total.reindex(full_index).interpolate().diff().dropna()
    times = np.concatenate([[increments.index.min() - 1], increments.index.to_numpy()])
    values = np.concatenate([[0.0], increments.to_numpy()])
    return times.astype(float), values.astype(float)


@dataclass(frozen=True)
class Demography:
    """Everything the model takes from the UN tables, prepared once."""

    single_age: pd.DataFrame
    grouped: pd.DataFrame
    death_rates: pd.DataFrame
    entry_times: np.ndarray
    entry_values: np.ndarray

    @property
    def total(self) -> pd.Series:
        """Total population by year."""
        return self.grouped.sum(axis=1)

    @property
    def initial_population(self) -> float:
        """First year's total population, used as the 1850 population."""
        return float(self.total.iloc[0])

    @property
    def pop_2020(self) -> float:
        """Total population in 2020, used to rescale the transmission rate."""
        return float(self.total.loc[2020])

    def death_rate_table(self, age: Property) -> TableData:
        """Per-age death-rate table over ``age`` (one column per trait)."""
        return Data.table(self.death_rates.index.to_numpy(dtype=float), self.death_rates, over=age)


@cache
def load_demography(
    iso3: str = ISO3,
    age_groups: tuple[str, ...] = AGE_GROUPS,
    scaling_factor: float = POP_SCALING,
) -> Demography:
    """Load and prepare the UN inputs for one country and age grouping."""
    single, grouped = get_population_over_time(iso3, age_groups, MAX_AGE, scaling_factor)
    death_rates = get_death_rates_by_age(grouped, iso3, age_groups)
    entry_times, entry_values = population_entry(grouped)
    return Demography(single, grouped, death_rates, entry_times, entry_values)
