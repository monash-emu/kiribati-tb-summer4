"""Compare posteriors (parameters and projections) across calibration runs.

Arms are posterior files; each gets per-parameter quantiles with Monte Carlo standard errors
and, unless ``--no-scenarios``, projections of ``--draws`` posterior draws through the
baseline and the screening scenarios. Each arm is cached under ``outputs/compare/arms/<name>``
and recomputed only when its file changes (or with ``--force``); the combined tables land in
``outputs/compare/`` for ``notebooks/06-fast-calibration.ipynb``:

- ``quantiles.parquet``: arm, parameter, quantile, value, mcse, ess_bulk, sd;
- ``projections.parquet``: arm, draw, scenario, metric, value (per-draw headline metrics);
- ``arms.json``: where each arm came from, its draw counts and the projection seed.

    pixi run python scripts/compare_posteriors.py [--arm NAME=outputs/calibrate/base] \\
        [--draws 1000] [--published-draws 2000]

Default arms, each included when its file exists: the four reference arms
(``outputs/reference/<arm>/mcmc/idata.nc``, from ``scripts/reference_run.py``), ``published``
(``reference/published/idata.nc``, draws 10,000-19,999 of each chain as the paper used, 2,000
of them projected), and ``aies_demo`` (the earlier unconverged AIES run).
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpyro  # noqa: F401 - import before arviz; see docs/summer4-workarounds.md, S4
import arviz as az
import numpy as np
import pandas as pd

from kiribati_tb.analysis import QUANTILES, run_scenarios
from kiribati_tb.calibration import calibration_setup
from kiribati_tb.paths import REPO_ROOT

OUT = REPO_ROOT / "outputs" / "compare"
PUBLISHED = REPO_ROOT / "reference" / "published" / "idata.nc"
PUBLISHED_BURN_IN = 10000
PUBLISHED_SEED = 20260819  # the published file's creation date; recorded in arms.json
DEFAULT_ARMS: dict[str, Path] = {
    **{
        arm: REPO_ROOT / "outputs" / "reference" / arm / "mcmc" / "idata.nc"
        for arm in ("nuts_td8", "nuts_td5", "sa", "ess")
    },
    "published": PUBLISHED,
    "aies_demo": REPO_ROOT / "outputs" / "calibrate" / "local_demo" / "idata.nc",
}
# The paper's scenarios: PEARL 65/85% (1, 3), CXR-TST 65/85% (6, 8), CXR only 85% (18).
SCENARIOS: tuple[str, ...] = ("scenario_1", "scenario_3", "scenario_6", "scenario_8", "scenario_18")
START_2026 = 2026.0


def load_posterior(name: str, path: Path) -> Any:
    """The arm's ``posterior`` group, post burn-in for the published file."""
    idata = az.from_netcdf(path)
    post = idata.posterior
    if name == "published":
        post = post.isel(draw=slice(PUBLISHED_BURN_IN, None))
    return post


def quantile_rows(name: str, post: Any) -> list[dict[str, Any]]:
    rows = []
    for param in sorted(post.data_vars):
        x = np.asarray(post[param].values)
        ess = float(az.ess(x, method="bulk"))
        sd = float(np.std(x))
        for q in QUANTILES:
            rows.append(
                {
                    "arm": name,
                    "parameter": param,
                    "quantile": q,
                    "value": float(np.quantile(x, q)),
                    "mcse": float(az.mcse(x, method="quantile", prob=q)),
                    "ess_bulk": ess,
                    "sd": sd,
                }
            )
    return rows


def projection_rows(name: str, runs: Any) -> list[dict[str, Any]]:
    """Per-draw headline metrics (the paper's definitions) from ``PosteriorRuns``."""
    times = np.asarray(runs.times)

    def at(year: float) -> int:
        return int(np.argmin(np.abs(times - year)))

    rows: list[dict[str, Any]] = []

    def add(scenario: str, metric: str, values: np.ndarray) -> None:
        for d, v in enumerate(np.asarray(values)):
            rows.append(
                {"arm": name, "draw": d, "scenario": scenario, "metric": metric, "value": float(v)}
            )

    t26, t28, t35, t25 = at(START_2026), at(2028.0), at(2035.0), at(2025.0)
    base_inc = runs.samples("baseline", "cum_tb_incidence")[:, t35]
    base_mort = runs.samples("baseline", "cum_tb_mortality")[:, t35]
    for output, metric in (
        ("tb_prevalence_per100k", "prevalence_per100k_2026"),
        ("tb_incidence_per100k", "incidence_per100k_2026"),
        ("viable_tbi_prevalence_perc", "viable_infection_perc_2026"),
    ):
        add("baseline", metric, runs.samples("baseline", output)[:, t26])
    add(
        "baseline",
        "incidence_per100k_2035",
        runs.samples("baseline", "tb_incidence_per100k")[:, t35],
    )
    add(
        "baseline",
        "unreachable_share_of_incidence_2025",
        runs.samples("baseline", "prop_tb_incidenceXreach_unreachable")[:, t25],
    )
    for sc in runs.scenario_names:
        if sc == "baseline":
            continue
        inc = runs.samples(sc, "cum_tb_incidence")[:, t35]
        mort = runs.samples(sc, "cum_tb_mortality")[:, t35]
        add(sc, "episodes_averted_perc", 100.0 * (base_inc - inc) / base_inc)
        add(sc, "deaths_averted_perc", 100.0 * (base_mort - mort) / base_mort)
        if sc == "scenario_3":
            add(sc, "incidence_per100k_2028", runs.samples(sc, "tb_incidence_per100k")[:, t28])
            add(sc, "mortality_per100k_2028", runs.samples(sc, "tb_mortality_per100k")[:, t28])
            add(sc, "incidence_per100k_2035", runs.samples(sc, "tb_incidence_per100k")[:, t35])
            add(
                sc,
                "unreachable_share_of_incidence_2028",
                runs.samples(sc, "prop_tb_incidenceXreach_unreachable")[:, t28],
            )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", action="append", default=[], metavar="NAME=FOLDER_OR_FILE")
    parser.add_argument("--draws", type=int, default=1000)
    parser.add_argument("--published-draws", type=int, default=2000)
    parser.add_argument("--no-scenarios", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--force", action="store_true", help="recompute cached arms")
    args = parser.parse_args()
    arms = dict(DEFAULT_ARMS)
    for item in args.arm:
        name, path = item.split("=", 1)
        p = Path(path)
        arms[name] = p / "idata.nc" if p.is_dir() else p
    arms = {k: v for k, v in arms.items() if v.exists()}
    OUT.mkdir(parents=True, exist_ok=True)
    setup = calibration_setup()
    meta_path = OUT / "arms.json"
    meta: dict[str, Any] = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    for name, path in arms.items():
        cache = OUT / "arms" / name
        stamp = path.stat().st_mtime
        if not args.force and meta.get(name, {}).get("mtime") == stamp and cache.exists():
            print(f"{name}: cached", flush=True)
            continue
        cache.mkdir(parents=True, exist_ok=True)
        post = load_posterior(name, path)
        pd.DataFrame(quantile_rows(name, post)).to_parquet(cache / "quantiles.parquet")
        n = args.published_draws if name == "published" else args.draws
        seed = PUBLISHED_SEED if name == "published" else args.seed
        info = {
            "path": str(path.relative_to(REPO_ROOT)),
            "mtime": stamp,
            "chains": int(post.sizes["chain"]),
            "draws_per_chain": int(post.sizes["draw"]),
            "burn_in_dropped": PUBLISHED_BURN_IN if name == "published" else 0,
            "projected_draws": n,
            "projection_seed": seed,
        }
        print(name, info, flush=True)
        if not args.no_scenarios:
            idata = az.from_dict({"posterior": {k: post[k].values for k in post.data_vars}})
            start = time.time()
            runs = run_scenarios(setup, idata, list(SCENARIOS), n=n, seed=seed)
            frame = pd.DataFrame(projection_rows(name, runs))
            frame.to_parquet(cache / "projections.parquet")
            info["projection_seconds"] = round(time.time() - start)
        meta[name] = info
        meta_path.write_text(json.dumps(meta, indent=2))
    for kind in ("quantiles", "projections"):
        parts = [
            pd.read_parquet(f)
            for f in sorted((OUT / "arms").glob(f"*/{kind}.parquet"))
            if f.parent.name in meta
        ]
        if parts:
            pd.concat(parts, ignore_index=True).to_parquet(OUT / f"{kind}.parquet")


if __name__ == "__main__":
    main()
