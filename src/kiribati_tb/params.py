"""Parameters from ``data/parameters.xlsx``.

Ports ``get_parameters_and_priors`` from ``tbh/runner_tools.py`` without estival: the constant
sheet becomes a parameter dict and a prior table, the time-variant sheet a dict of series.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cache

import pandas as pd

from kiribati_tb.paths import DATA

PARAMETERS_XLSX = DATA / "parameters.xlsx"


@dataclass(frozen=True)
class ParameterSheet:
    """The two sheets of ``parameters.xlsx``.

    Attributes:
        constants: Fixed value of every constant parameter, by name.
        priors: Rows of the constant sheet that carry a prior: ``parameter``, ``distribution``,
            ``distri_param1``, ``distri_param2``.
        time_variant: Each time-variant column as a series indexed by year, NaNs dropped.
    """

    constants: dict[str, float]
    priors: pd.DataFrame
    time_variant: dict[str, pd.Series]

    def midpoint_params(self) -> dict[str, float]:
        """Constants with every uniform prior at the midpoint of its range."""
        params = dict(self.constants)
        for _, row in self.priors.iterrows():
            if row["distribution"] != "uniform":
                raise ValueError(f"Unsupported prior {row['distribution']!r}")
            lo, hi = float(row["distri_param1"]), float(row["distri_param2"])
            params[str(row["parameter"])] = 0.5 * (lo + hi)
        return params


@cache
def read_parameter_sheet() -> ParameterSheet:
    """Read ``parameters.xlsx`` exactly as the original ``get_parameters_and_priors`` does."""
    frame = pd.read_excel(PARAMETERS_XLSX, sheet_name="constant")
    constants = {str(k): float(v) for k, v in zip(frame["parameter"], frame["value"])}
    priors = frame[frame["distribution"].notna()][
        ["parameter", "distribution", "distri_param1", "distri_param2"]
    ].reset_index(drop=True)
    tv_frame = pd.read_excel(PARAMETERS_XLSX, sheet_name="time_variant", index_col=0)
    time_variant = {str(col): tv_frame[col].dropna() for col in tv_frame.columns}
    return ParameterSheet(constants, priors, time_variant)
