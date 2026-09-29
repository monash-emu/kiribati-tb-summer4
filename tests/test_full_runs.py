"""K5: full runs write the original's parquet files and schemas."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from kiribati_tb.analysis import QUANTILES, full_runs, saved_names
from kiribati_tb.calibration import calibration_setup
from kiribati_tb.paths import GOLDEN


def test_full_runs_schema(tmp_path: Path) -> None:
    setup = calibration_setup()
    params = yaml.safe_load((GOLDEN / "hetero_baseline" / "params.yaml").read_text())["params"]
    names = [p.name for p in setup.priors] + ["mixing_dist_sd"]
    rng = np.random.default_rng(0)
    draws = {k: params.get(k, 12.5) * (1.0 + 0.01 * rng.standard_normal(3)) for k in names}
    runs = full_runs(setup, draws, tmp_path, ["scenario_3"], n=3, batch_size=3)

    files = sorted(p.name for p in tmp_path.iterdir())
    assert files == [
        "details.yaml",
        "diff_quantiles_df_ref_baseline_scenario_3.parquet",
        "uncertainty_df_baseline.parquet",
        "uncertainty_df_scenario_3.parquet",
    ]
    unc = pd.read_parquet(tmp_path / "uncertainty_df_scenario_3.parquet")
    assert unc.index[0] == 1850.0 and unc.index[-1] == 2035.0
    assert unc.columns.names == ["output", "quantile"]
    assert set(unc.columns.get_level_values(1)) == {str(q) for q in QUANTILES}
    assert set(unc.columns.get_level_values(0)) == set(saved_names(setup.config))

    diff = pd.read_parquet(tmp_path / "diff_quantiles_df_ref_baseline_scenario_3.parquet")
    assert list(diff.columns) == [
        "TB_averted",
        "deaths_averted",
        "TB_averted_relative",
        "deaths_averted_relative",
    ]
    np.testing.assert_allclose(diff.index.to_numpy(), QUANTILES)
    # Averted = baseline - scenario at 2035, per draw.
    base = runs.samples("baseline", "cum_tb_incidence")[:, -1]
    scen = runs.samples("scenario_3", "cum_tb_incidence")[:, -1]
    np.testing.assert_allclose(diff.loc[0.5, "TB_averted"], np.median(base - scen))
    assert (diff["TB_averted"] > 0).all()


def test_saved_names_are_the_originals_saved_outputs() -> None:
    names = saved_names(calibration_setup().config)
    assert len(names) == 137
    assert "tb_incidence_per100k" in names and "cum_tb_mortality" in names
    assert not any(n.startswith(("prev_", "all_cause_mortality_from_")) for n in names)
