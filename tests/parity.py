"""Parity comparison helpers shared by the parity tests."""

from __future__ import annotations

import numpy as np
import pandas as pd

from kiribati_tb.paths import GOLDEN

RTOL = 1e-5


def read_golden(fixture: str, kind: str) -> pd.DataFrame:
    """A golden frame (``compartments`` or ``derived``), indexed by time."""
    return pd.read_parquet(GOLDEN / fixture / f"{kind}.parquet").set_index("time")


def parity_report(got: pd.DataFrame, expected: pd.DataFrame, atol: float) -> pd.DataFrame:
    """Max absolute and relative error per column, and whether it passes ``RTOL``/``atol``.

    Relative error is ``|got - expected| / max(|expected|, atol)``. A value that is NaN in both
    (a 0/0 ratio) counts as equal.
    """
    got_arr = got[expected.columns].to_numpy()
    exp_arr = expected.to_numpy()
    both_nan = np.isnan(got_arr) & np.isnan(exp_arr)  # 0/0 ratios at 1850 in both models
    diff = np.where(both_nan, 0.0, np.abs(got_arr - exp_arr))
    scale = np.maximum(np.where(both_nan, 1.0, np.abs(exp_arr)), atol)
    passes = diff <= atol + RTOL * np.abs(np.where(both_nan, 0.0, exp_arr))
    return pd.DataFrame(
        {
            "max_abs": diff.max(axis=0),
            "max_rel": (diff / scale).max(axis=0),
            "passes": passes.all(axis=0),
        },
        index=expected.columns,
    ).sort_values("max_rel", ascending=False)


def assert_parity(got: pd.DataFrame, expected: pd.DataFrame, atol: float) -> pd.DataFrame:
    """Fail with the per-column error table if any column misses the tolerance."""
    missing = sorted(set(expected.columns) - set(got.columns))
    assert not missing, f"missing columns: {missing[:20]} ({len(missing)} total)"
    np.testing.assert_array_equal(got.index.to_numpy(), expected.index.to_numpy())
    report = parity_report(got, expected, atol)
    failing = report[~report["passes"]]
    assert failing.empty, (
        f"{len(failing)} of {len(report)} columns miss rtol={RTOL}, atol={atol}:\n"
        f"{failing.head(30).to_string()}"
    )
    return report
