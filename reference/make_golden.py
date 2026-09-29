"""Regenerate parity goldens with the original summer2gen model.

Run only via ``pixi run -e reference golden``. The reference environment pins the stack of
``monash-emu/kiribati_tb_modelling@46a63f0``'s ``pixi.lock`` (python 3.10, summer2gen
``aad6d4b``, jax 0.4.38) and never shares a solve with summer4.

The model code is vendored read-only under ``reference/tbh/`` (see ``reference/README.md`` for
the two files changed and why). Scenarios come from the copied ``data/scenarios.py``.

Each fixture uses the constant sheet of ``data/parameters.xlsx`` with every prior at the
midpoint of its range, then the scenario's ``params_ow``. The solver is summer2gen's default
JAX ``odeint`` (Dormand-Prince, ``max_step=1``) with ``rtol = atol = 1e-8`` instead of the
default ``1.4e-4``, so that reference error sits well under the parity tolerance.

Every derived output is saved, including the ``save_results=False`` intermediates, because the
port reproduces those names too.

Writes, per fixture, ``tests/golden/<fixture>/``:

- ``compartments.parquet``: ``time`` then one column per compartment, ``str(Compartment)``;
- ``derived.parquet``: ``time`` then one column per derived output, sorted by name;
- ``params.yaml``: the exact parameter dict and model config used.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "reference"))
sys.path.insert(0, str(ROOT))

from tbh.model import get_tb_model  # noqa: E402

from data.scenarios import SCENARIOS  # noqa: E402

OUT = ROOT / "tests" / "golden"
SOLVER_ARGS = {"rtol": 1e-8, "atol": 1e-8}

# DEFAULT_MODEL_CONFIG from tbh/runner_tools.py (not vendored: it imports pymc and estival).
MODEL_CONFIG: dict[str, Any] = {
    "start_time": 1850,
    "end_time": 2035,
    "seed": 100,
    "iso3": "KIR",
    "age_groups": ["0", "3", "5", "10", "15", "18", "40", "65"],
    "pop_scaling": 40483.0 / 119438.0,
    "heterogeneous_mixing": True,
}

FIXTURES: dict[str, dict[str, Any]] = {
    "no_transmission": {
        "heterogeneous_mixing": False,
        "scenario": None,
        "params": {"raw_transmission_rate": 0.0},
    },
    "homog_baseline": {"heterogeneous_mixing": False, "scenario": None, "params": {}},
    "hetero_baseline": {"heterogeneous_mixing": True, "scenario": None, "params": {}},
    "hetero_scenario_3": {"heterogeneous_mixing": True, "scenario": "scenario_3", "params": {}},
    "hetero_scenario_8": {"heterogeneous_mixing": True, "scenario": "scenario_8", "params": {}},
}


def read_parameters() -> tuple[dict[str, float], dict[str, pd.Series]]:
    """Read ``parameters.xlsx`` as ``get_parameters_and_priors`` does, priors at midpoints."""
    path = ROOT / "data" / "parameters.xlsx"
    df = pd.read_excel(path, sheet_name="constant")
    params: dict[str, float] = {}
    for _, row in df.iterrows():
        value = float(row["value"])
        if isinstance(row["distribution"], str):
            if row["distribution"] != "uniform":
                raise ValueError(f"unsupported prior {row['distribution']}")
            value = 0.5 * (float(row["distri_param1"]) + float(row["distri_param2"]))
        params[str(row["parameter"])] = value
    tv_df = pd.read_excel(path, sheet_name="time_variant", index_col=0)
    tv_params = {col: tv_df[col].dropna() for col in tv_df.columns}
    return params, tv_params


def _write_parquet(frame: pd.DataFrame, path: Path) -> None:
    table = pa.Table.from_pandas(frame, preserve_index=False)
    pq.write_table(table, path, compression="snappy")


def build_and_run(fixture: dict[str, Any]) -> tuple[Any, dict[str, float], dict[str, Any]]:
    """Build the original model for one fixture and run it with every output saved."""
    params, tv_params = read_parameters()
    config = MODEL_CONFIG | {"heterogeneous_mixing": fixture["heterogeneous_mixing"]}
    programs: list[Any] = []
    if fixture["scenario"] is not None:
        (scenario,) = [s for s in SCENARIOS if s.sc_id == fixture["scenario"]]
        programs = scenario.scr_prgs
        params = params | dict(scenario.params_ow)
    params = params | fixture["params"]
    model = get_tb_model(config, tv_params, screening_programs=programs)
    for request in model._derived_output_requests.values():
        request["save_results"] = True
    model.run(params, solver_args=SOLVER_ARGS)
    return model, params, config


def save(name: str, fixture: dict[str, Any]) -> None:
    model, params, config = build_and_run(fixture)
    folder = OUT / name
    folder.mkdir(parents=True, exist_ok=True)

    comps = model.get_outputs_df()
    times = np.asarray(comps.index, dtype=float)
    comps = comps.reset_index(drop=True)
    comps.insert(0, "time", times)
    _write_parquet(comps, folder / "compartments.parquet")

    derived = model.get_derived_outputs_df()
    derived = derived[sorted(derived.columns)].reset_index(drop=True).astype(float)
    derived.insert(0, "time", times)
    _write_parquet(derived, folder / "derived.parquet")

    record = {
        "fixture": name,
        "scenario": fixture["scenario"],
        "model_config": config,
        "solver_args": SOLVER_ARGS,
        "params": {k: float(v) for k, v in sorted(params.items())},
    }
    with open(folder / "params.yaml", "w") as handle:
        yaml.safe_dump(record, handle, sort_keys=False)
    print(f"{name}: {comps.shape[1] - 1} compartments, {derived.shape[1] - 1} derived outputs")


def main() -> None:
    for name, fixture in FIXTURES.items():
        save(name, fixture)


if __name__ == "__main__":
    main()
