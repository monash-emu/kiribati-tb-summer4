"""Compare the reference calibration arms with each other and with the published posterior.

    pixi run python scripts/compare_reference.py [--root outputs/reference] [--arm NAME=PATH]

Loads every arm under ``--root`` that has written at least one sampling chunk
(``<root>/<arm>/mcmc/idata.nc``; a run still in progress is read as it stands) and the published
posterior (``reference/published/idata.nc``, draws 10,000-19,999 of each chain, the paper's
burn-in), then writes to ``outputs/compare_reference/`` (or ``--out``):

- ``diagnostics.csv``: per arm and parameter, R-hat and bulk and tail ESS;
- ``efficiency.csv``: per arm, draws, decision, divergences, tree depth, acceptance, wall time,
  CPU hours, smallest ESS and smallest ESS per CPU hour;
- ``quantiles.csv``: per arm, parameter and quantile (2.5/25/50/75/97.5%), value and Monte Carlo
  standard error;
- ``agreement.csv``: per pair of arms, the largest |z| over parameters and quantiles
  (difference / combined MCSE), the share above 3, and the largest difference in posterior sds;
- ``marginals.html``, ``efficiency.html``, ``agreement.html``: the figures.

``notebooks/06-fast-calibration.ipynb`` reads these files.
"""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path
from typing import Any

import numpyro  # noqa: F401 - import before arviz; see docs/summer4-workarounds.md, S4
import arviz as az
import numpy as np
import pandas as pd
import plotly.express as px

from kiribati_tb.paths import REPO_ROOT
from kiribati_tb.pipeline import diagnostics

ARMS: tuple[str, ...] = ("nuts_td8", "nuts_td5", "sa", "ess")
PUBLISHED = REPO_ROOT / "reference" / "published" / "idata.nc"
PUBLISHED_BURN_IN = 10000
# The published run: 72,130 s on 8 cores (the paper's cluster jobs had 8 cores).
PUBLISHED_CPU_HOURS = 72130.0 * 8 / 3600.0
QUANTILES: tuple[float, ...] = (0.025, 0.25, 0.5, 0.75, 0.975)


def load_arms(root: Path, extra: dict[str, Path]) -> dict[str, dict[str, Any]]:
    """Posterior draws and run records for every arm that has any."""
    arms: dict[str, dict[str, Any]] = {}
    for arm in ARMS:
        folder = root / arm / "mcmc"
        if (folder / "idata.nc").exists():
            post = az.from_netcdf(folder / "idata.nc").posterior
            progress = pd.read_csv(folder / "progress.csv")
            arms[arm] = {"posterior": post, "last": progress.iloc[-1].to_dict()}
    for name, path in extra.items():
        arms[name] = {"posterior": az.from_netcdf(path).posterior, "last": {}}
    if PUBLISHED.exists():
        post = az.from_netcdf(PUBLISHED).posterior.isel(draw=slice(PUBLISHED_BURN_IN, None))
        arms["published"] = {
            "posterior": post,
            "last": {"seconds": 72130.0, "cpu_hours": PUBLISHED_CPU_HOURS, "decision": "fixed"},
        }
    return arms


def tables(arms: dict[str, dict[str, Any]]) -> dict[str, pd.DataFrame]:
    diag_rows, eff_rows, q_rows = [], [], []
    for arm, rec in arms.items():
        post = rec["posterior"]
        params = sorted(post.data_vars)
        diag = diagnostics({p: post[p].values for p in params})
        for p, row in diag.iterrows():
            diag_rows.append({"arm": arm, "parameter": p, **row.to_dict()})
        last = rec["last"]
        ess_min = float(diag[["ess_bulk", "ess_tail"]].min().min())
        cpu_hours = last.get("cpu_hours")
        eff_rows.append(
            {
                "arm": arm,
                "chains": int(post.sizes["chain"]),
                "draws_per_chain": int(post.sizes["draw"]),
                "decision": last.get("decision"),
                "rhat_max": float(diag["rhat"].max()),
                "ess_min": ess_min,
                "divergence_frac": last.get("divergence_frac"),
                "treedepth_mean": last.get("chunk_treedepth_mean"),
                "accept_mean": last.get("chunk_accept_mean"),
                "sampling_hours": None if last.get("seconds") is None else last["seconds"] / 3600,
                "cpu_hours": cpu_hours,
                "ess_min_per_cpu_hour": (None if not cpu_hours else ess_min / float(cpu_hours)),
            }
        )
        for p in params:
            x = np.asarray(post[p].values)
            sd = float(np.std(x))
            for q in QUANTILES:
                q_rows.append(
                    {
                        "arm": arm,
                        "parameter": p,
                        "quantile": q,
                        "value": float(np.quantile(x, q)),
                        "mcse": float(az.mcse(x, method="quantile", prob=q)),
                        "sd": sd,
                    }
                )
    quantiles = pd.DataFrame(q_rows)
    agree_rows = []
    wide = quantiles.pivot_table(
        index=["parameter", "quantile"], columns="arm", values=["value", "mcse", "sd"]
    )
    for a, b in itertools.combinations(list(arms), 2):
        diff = wide["value"][a] - wide["value"][b]
        z = diff / np.sqrt(wide["mcse"][a] ** 2 + wide["mcse"][b] ** 2)
        sd = 0.5 * (wide["sd"][a] + wide["sd"][b])
        agree_rows.append(
            {
                "arm_a": a,
                "arm_b": b,
                "max_abs_z": float(z.abs().max()),
                "share_abs_z_over_3": float((z.abs() > 3).mean()),
                "max_abs_diff_in_sd": float((diff / sd).abs().max()),
                "worst": "{} q={}".format(*z.abs().idxmax()),
            }
        )
    return {
        "diagnostics": pd.DataFrame(diag_rows),
        "efficiency": pd.DataFrame(eff_rows),
        "quantiles": quantiles,
        "agreement": pd.DataFrame(agree_rows),
    }


def figures(arms: dict[str, dict[str, Any]], out: dict[str, pd.DataFrame], folder: Path) -> None:
    from kiribati_tb.calibration import calibration_setup
    from summer4.epi.calibration import Uniform

    setup = calibration_setup()
    bounds = {p.name: (p.lo, p.hi) for p in setup.priors if isinstance(p, Uniform)}
    bounds["mixing_dist_sd"] = (5.0, 20.0)
    rng = np.random.default_rng(0)
    rows = []
    for arm, rec in arms.items():
        post = rec["posterior"]
        for p in sorted(post.data_vars):
            x = post[p].values.ravel()
            x = rng.choice(x, size=min(len(x), 8000), replace=False)
            lo, hi = bounds[p]
            rows.append(
                pd.DataFrame({"arm": arm, "parameter": p, "position": (x - lo) / (hi - lo)})
            )
    fig = px.histogram(
        pd.concat(rows),
        x="position",
        color="arm",
        facet_col="parameter",
        facet_col_wrap=4,
        histnorm="probability density",
        barmode="overlay",
        opacity=0.45,
        nbins=40,
        height=1500,
        title="Posterior marginals by arm, each parameter scaled to its prior range",
    )
    fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))
    fig.write_html(folder / "marginals.html")
    eff = out["efficiency"].dropna(subset=["ess_min_per_cpu_hour"])
    px.bar(
        eff,
        x="arm",
        y="ess_min_per_cpu_hour",
        log_y=True,
        title="Smallest ESS (bulk or tail, over parameters) per CPU hour of sampling",
        labels={"ess_min_per_cpu_hour": "ESS per CPU hour"},
    ).write_html(folder / "efficiency.html")
    px.bar(
        out["agreement"],
        x=out["agreement"]["arm_a"] + " vs " + out["agreement"]["arm_b"],
        y="max_abs_z",
        title="Largest |quantile difference| / combined MCSE, per pair of arms (3 = agreement)",
        labels={"x": "pair", "max_abs_z": "max |z|"},
    ).write_html(folder / "agreement.html")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=str(REPO_ROOT / "outputs" / "reference"))
    parser.add_argument("--out", default=str(REPO_ROOT / "outputs" / "compare_reference"))
    parser.add_argument("--arm", action="append", default=[], metavar="NAME=IDATA")
    parser.add_argument("--no-figures", action="store_true")
    args = parser.parse_args()
    extra = {k: Path(v) for k, v in (item.split("=", 1) for item in args.arm)}
    arms = load_arms(Path(args.root), extra)
    if not arms:
        raise SystemExit("No arms found.")
    folder = Path(args.out)
    folder.mkdir(parents=True, exist_ok=True)
    out = tables(arms)
    for name, frame in out.items():
        frame.to_csv(folder / f"{name}.csv", index=False)
    if not args.no_figures:
        figures(arms, out, folder)
    (folder / "arms.json").write_text(json.dumps(sorted(arms), indent=2))
    with pd.option_context("display.width", 200, "display.max_columns", 20):
        print(out["efficiency"].round(3).to_string(index=False))
        print(out["agreement"].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
