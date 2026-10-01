"""Figures and tables for docs/neutra.md from NeuTra runs and the published posterior.

    pixi run python scripts/neutra_report.py [--run outputs/neutra/base] \\
        [--arm reference=outputs/calibrate/reference/idata.nc] [--out docs/figures/neutra]

Writes to ``--out``: ``elbo.png`` (SVI loss and the flow's PSIS ``k_hat``), ``marginals.png``
(every parameter, each arm against the published draws 10,000-19,999), ``ridge.png`` (the
transmission ridge), ``tradeoffs.png`` (``rel_sus_children``, ``bg_mixing``, ``pc_strength``)
and ``report.json`` (flow and sampler measurements, and quantiles with Monte Carlo standard
errors for every arm, with each quantile's distance from the published one in units of the
combined MCSE).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpyro  # noqa: F401 - import before arviz; see docs/summer4-workarounds.md, S4
import arviz as az
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from kiribati_tb.neutra import efficiency  # noqa: E402
from kiribati_tb.paths import REPO_ROOT  # noqa: E402

PUBLISHED = REPO_ROOT / "reference" / "published" / "idata.nc"
BURN_IN = 10000
QUANTILES = (0.025, 0.25, 0.5, 0.75, 0.975)
COLOURS = ("#1f6feb", "#d1495b", "#2a9d8f", "#e9c46a")


def load(path: Path, *, published: bool = False) -> dict[str, np.ndarray]:
    post = az.from_netcdf(path).posterior
    if published:
        post = post.isel(draw=slice(BURN_IN, None))
    return {k: np.asarray(post[k].values) for k in post.data_vars}


def quantile_table(name: str, draws: dict[str, np.ndarray]) -> pd.DataFrame:
    rows = []
    for param in sorted(draws):
        x = draws[param]
        for q in QUANTILES:
            rows.append(
                {
                    "arm": name,
                    "parameter": param,
                    "quantile": q,
                    "value": float(np.quantile(x, q)),
                    "mcse": float(az.mcse(x, method="quantile", prob=q)),
                    "ess_bulk": float(az.ess(x, method="bulk")),
                    "rhat": float(az.rhat(x)),
                }
            )
    return pd.DataFrame(rows)


def compare(table: pd.DataFrame, base: str = "published") -> pd.DataFrame:
    ref = table[table["arm"] == base].set_index(["parameter", "quantile"])
    out = []
    for arm, frame in table[table["arm"] != base].groupby("arm"):
        f = frame.set_index(["parameter", "quantile"])
        z = (f["value"] - ref["value"]) / np.sqrt(f["mcse"] ** 2 + ref["mcse"] ** 2)
        out.append(
            pd.DataFrame(
                {"arm": arm, "difference": f["value"] - ref["value"], "z": z}
            ).reset_index()
        )
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


def plot_flow(run: Path, out: Path) -> dict[str, Any]:
    flow = run / "flow"
    losses = np.load(flow / "losses.npy")
    progress = pd.read_csv(flow / "progress.csv")
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    good = np.where(np.isfinite(losses) & (losses < 1e20), losses, np.nan)
    axes[0].plot(np.arange(1, len(good) + 1), good, lw=0.4, color="#999999", label="per step")
    smooth = pd.Series(good).rolling(50, min_periods=10).median()
    axes[0].plot(np.arange(1, len(good) + 1), smooth, color=COLOURS[0], label="rolling median (50)")
    lo, hi = np.nanpercentile(good, [1, 99])
    axes[0].set_ylim(lo - 0.1 * (hi - lo), hi)
    axes[0].set(xlabel="SVI step", ylabel="negative ELBO (nats)", title="SVI loss")
    axes[0].legend()
    checked = progress.dropna(subset=["k_hat"])
    axes[1].plot(checked["step"], checked["k_hat"], "o-", color=COLOURS[1])
    axes[1].axhline(0.7, ls="--", color="k", lw=0.8)
    axes[1].axhline(0.5, ls=":", color="k", lw=0.8)
    axes[1].set(
        xlabel="SVI step",
        ylabel="PSIS k-hat",
        title="Flow as an importance proposal (k-hat < 0.7 usable)",
    )
    fig.tight_layout()
    fig.savefig(out / "elbo.png", dpi=130)
    plt.close(fig)
    return {
        "steps": int(len(losses)),
        "failed_steps": int(np.sum(~np.isfinite(losses) | (losses >= 1e20))),
        "checks": checked[["step", "elbo", "k_hat", "is_ess_frac", "failed"]].to_dict("records"),
    }


def plot_marginals(arms: dict[str, dict[str, np.ndarray]], out: Path) -> None:
    params = sorted(arms["published"])
    ncol = 4
    nrow = int(np.ceil(len(params) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4 * ncol, 2.6 * nrow))
    for ax, p in zip(axes.ravel(), params):
        allx = np.concatenate([a[p].ravel() for a in arms.values()])
        bins = np.linspace(allx.min(), allx.max(), 50)
        for colour, (name, draws) in zip(COLOURS, arms.items()):
            ax.hist(
                draws[p].ravel(),
                bins=bins,
                density=True,
                histtype="step",
                lw=1.3,
                color=colour,
                label=name,
            )
        ax.set_title(p, fontsize=9)
        ax.tick_params(labelsize=7)
    for ax in axes.ravel()[len(params) :]:
        ax.axis("off")
    axes.ravel()[0].legend(fontsize=7)
    fig.suptitle("Marginal posteriors (published: draws 10,000-19,999 of 8 chains)")
    fig.tight_layout()
    fig.savefig(out / "marginals.png", dpi=110)
    plt.close(fig)


def _thin(x: np.ndarray, n: int, seed: int = 0) -> np.ndarray:
    flat = x.ravel()
    idx = np.random.default_rng(seed).choice(len(flat), size=min(n, len(flat)), replace=False)
    return flat[idx]


def plot_pairs(arms: dict[str, dict[str, np.ndarray]], out: Path) -> None:
    fig, axes = plt.subplots(1, len(arms), figsize=(4.5 * len(arms), 4), sharex=True, sharey=True)
    for ax, colour, (name, draws) in zip(np.atleast_1d(axes), COLOURS, arms.items()):
        s = _thin(draws["infection_pop_scale"], 3000)
        r = _thin(draws["raw_transmission_rate"], 3000)
        ax.scatter(s, r, s=3, alpha=0.3, color=colour)
        ax.set_yscale("log")
        ax.set(title=name, xlabel="infection_pop_scale", ylabel="raw_transmission_rate")
    fig.suptitle("The transmission ridge (3,000 draws per arm)")
    fig.tight_layout()
    fig.savefig(out / "ridge.png", dpi=120)
    plt.close(fig)

    pairs = [
        ("rel_sus_children", "bg_mixing"),
        ("rel_sus_children", "pc_strength"),
        ("bg_mixing", "pc_strength"),
    ]
    fig, axes = plt.subplots(len(arms), 3, figsize=(12, 3.4 * len(arms)), squeeze=False)
    for i, (colour, (name, draws)) in enumerate(zip(COLOURS, arms.items())):
        for j, (a, b) in enumerate(pairs):
            ax = axes[i, j]
            ax.scatter(_thin(draws[a], 3000), _thin(draws[b], 3000), s=3, alpha=0.3, color=colour)
            ax.set(xlabel=a, ylabel=b, title=name if j == 0 else None)
    fig.suptitle("Trade-offs among the mixing and susceptibility parameters")
    fig.tight_layout()
    fig.savefig(out / "tradeoffs.png", dpi=110)
    plt.close(fig)


def plot_quantile_z(cmp: pd.DataFrame, out: Path) -> None:
    if cmp.empty:
        return
    arms = sorted(cmp["arm"].unique())
    fig, ax = plt.subplots(figsize=(10, 5))
    params = sorted(cmp["parameter"].unique())
    for k, (colour, arm) in enumerate(zip(COLOURS[1:], arms)):
        f = cmp[cmp["arm"] == arm]
        y = np.asarray([params.index(p) for p in f["parameter"]]) + 0.15 * k
        ax.scatter(f["z"], y, s=12, color=colour, label=arm)
    ax.axvline(-2, ls="--", color="k", lw=0.8)
    ax.axvline(2, ls="--", color="k", lw=0.8)
    ax.set_yticks(range(len(params)), params, fontsize=8)
    ax.set_xlabel("(quantile - published quantile) / combined MCSE, quantiles 2.5-97.5%")
    ax.set_title("Quantile differences from the published posterior, in Monte Carlo errors")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "quantile_z.png", dpi=120)
    plt.close(fig)


def sampler_summary(run: Path) -> dict[str, Any]:
    nuts = run / "nuts"
    out: dict[str, Any] = {}
    if (nuts / "warmup.csv").exists():
        out["warmup"] = pd.read_csv(nuts / "warmup.csv").to_dict("records")
    if (nuts / "progress.csv").exists():
        out["progress"] = pd.read_csv(nuts / "progress.csv").to_dict("records")
    if (run / "summary.json").exists():
        out["summary"] = json.loads((run / "summary.json").read_text())
    if (run / "stage_seconds.json").exists():
        out["stage_seconds"] = json.loads((run / "stage_seconds.json").read_text())
    if (nuts / "num_steps.npy").exists():
        steps = np.load(nuts / "num_steps.npy")
        out["num_steps"] = {
            "mean": float(steps.mean()),
            "median": float(np.median(steps)),
            "max": int(steps.max()),
            "tree_depth_hist": {
                int(d): int(c)
                for d, c in zip(*np.unique(np.ceil(np.log2(steps + 1)), return_counts=True))
            },
        }
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", default="outputs/neutra/base", help="the NeuTra run")
    parser.add_argument(
        "--baseline", default="outputs/neutra/baseline_shear", help="the matched no-flow run"
    )
    parser.add_argument("--arm", action="append", default=[], metavar="NAME=IDATA")
    parser.add_argument("--out", default="docs/figures/neutra")
    args = parser.parse_args()
    run = REPO_ROOT / args.run
    out = REPO_ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)
    arms = {"published": load(PUBLISHED, published=True)}
    runs = {"neutra": run, "shear_nuts": REPO_ROOT / args.baseline}
    report: dict[str, Any] = {"flow": plot_flow(run, out), "sampler": {}}
    for name, folder in runs.items():
        if not (folder / "nuts" / "idata.nc").exists():
            continue
        arms[name] = load(folder / "nuts" / "idata.nc")
        summary = sampler_summary(folder)
        steps_path = folder / "nuts" / "num_steps.npy"
        if steps_path.exists():
            steps = np.load(steps_path)
            draws = {k: v[:, : steps.shape[1]] for k, v in arms[name].items()}
            summary["efficiency"] = efficiency(draws, steps)
        report["sampler"][name] = summary
    for item in args.arm:
        name, path = item.split("=", 1)
        arms[name] = load(REPO_ROOT / path)
    table = pd.concat([quantile_table(n, d) for n, d in arms.items()], ignore_index=True)
    cmp = compare(table)
    report["quantiles"] = table.to_dict("records")
    if not cmp.empty:
        report["comparison"] = cmp.to_dict("records")
        report["comparison_summary"] = {
            arm: {
                "abs_z_median": float(f["z"].abs().median()),
                "abs_z_max": float(f["z"].abs().max()),
                "frac_abs_z_gt_2": float((f["z"].abs() > 2).mean()),
                "worst": f.loc[f["z"].abs().idxmax(), ["parameter", "quantile", "z"]].to_dict(),
            }
            for arm, f in cmp.groupby("arm")
        }
    plot_marginals(arms, out)
    plot_pairs(arms, out)
    plot_quantile_z(cmp, out)
    (out / "report.json").write_text(json.dumps(report, indent=2, default=float))
    print(json.dumps(report.get("comparison_summary", {}), indent=2, default=float))


if __name__ == "__main__":
    main()
