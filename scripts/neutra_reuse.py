"""Can one flow serve the whole grid? Score a trained flow under other configurations.

    pixi run python scripts/neutra_reuse.py [--flow-run outputs/neutra/base] [--draws 128]

For each of the published grid's 16 configurations (``scripts/cluster/massiverun.py``) and the
sensitivity analyses with the same parameters (``tpt_60``, ``subclinical_50``; the
``homogeneous_mixing`` analysis drops four sites, so a 19-dimensional flow cannot serve it),
the base run's coordinates and flow are kept and :func:`kiribati_tb.neutra.flow_check` scores
them under that configuration's posterior: the ELBO and the PSIS ``k_hat`` of the flow as an
importance proposal. Results are appended to ``outputs/neutra/reuse/checks.jsonl`` (one line
per configuration; configurations already there are skipped, so a killed run resumes).
"""

from __future__ import annotations

import argparse
import json
import time
from itertools import product
from pathlib import Path

REGRESSION_RATE_VALUES: tuple[float, ...] = (0.5, 1.0, 2.0, 3.0)
REL_SUS_UNREACHABLE_VALUES: tuple[float, ...] = (1.0, 1.5, 2.0, 3.0)


def configurations() -> list[tuple[str, str | None, dict[str, float]]]:
    """``(label, sensitivity analysis, overrides)`` for the grid and the usable analyses."""
    grid = [
        (
            f"grid_rate{rate}_relsus{rel_sus}",
            None,
            {
                "clinical_regression_rate": rate,
                "infectiousness_loss_rate": rate,
                "rel_sus_unreachable": rel_sus,
            },
        )
        for rate, rel_sus in product(REGRESSION_RATE_VALUES, REL_SUS_UNREACHABLE_VALUES)
    ]
    sas = [("sa_tpt_60", "tpt_60", {}), ("sa_subclinical_50", "subclinical_50", {})]
    return [("base", None, {})] + grid + sas


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--flow-run", default="outputs/neutra/base")
    parser.add_argument("--draws", type=int, default=128)
    parser.add_argument("--only", default=None, help="comma-separated labels")
    parser.add_argument("--solver", default="calibration", choices=["calibration", "original"])
    args = parser.parse_args()

    import numpyro  # noqa: F401
    import jax
    import jax.numpy as jnp

    from kiribati_tb.calibration import bayesian_model, calibration_setup
    from kiribati_tb.neutra import (
        NeutraConfig,
        WhitenedCoordinates,
        flat_model,
        flow_check,
        load_params,
        make_guide,
    )
    from kiribati_tb.calibration import ORIGINAL_CALIBRATION_SOLVER, forward_mode_solver
    from kiribati_tb.paths import REPO_ROOT
    from kiribati_tb.pipeline import forward_log_density

    original = ORIGINAL_CALIBRATION_SOLVER if args.solver == "original" else None
    source = REPO_ROOT / args.flow_run
    coords = WhitenedCoordinates.from_json(json.loads((source / "coords.json").read_text()))
    saved = json.loads((source / "args.json").read_text())
    config = NeutraConfig(
        flow=saved["flow"],
        num_flows=saved["num_flows"],
        hidden=tuple(int(h) for h in saved["hidden"].split(",")),
    )
    params = load_params(source / "flow")
    out = REPO_ROOT / "outputs" / "neutra" / "reuse"
    out.mkdir(parents=True, exist_ok=True)
    path = out / "checks.jsonl"
    done = set()
    if path.exists():
        done = {json.loads(line)["label"] for line in path.read_text().splitlines() if line}
    wanted = None if args.only is None else set(args.only.split(","))
    for label, analysis, overrides in configurations():
        if label in done or (wanted is not None and label not in wanted):
            continue
        start = time.perf_counter()
        setup = calibration_setup(analysis, overrides)
        bm = bayesian_model(setup, solver=forward_mode_solver(original))
        if tuple(sorted(bm.prior_names())) != coords.sites:
            print(f"[reuse] {label}: different sites, skipped", flush=True)
            continue
        # Forward evaluations only. The model is built with diffrax's forward-mode adjoint so
        # numpyro's start-up check takes a forward-mode gradient: with the original solver every
        # reverse-mode gradient is NaN at regression rates >= 2.5 and BayesianModel.potential_fn
        # cannot be built there (docs/summer4-workarounds.md, S12).
        density = forward_log_density(bm)
        potential = coords.pull_back(lambda z: -density(z))
        # The guide only needs the site's shape; its set-up trace (a reverse-mode check) runs
        # on a stand-in Gaussian so the forward-mode model is never differentiated in reverse.
        stand_in = flat_model(lambda x: 0.5 * jnp.sum(x**2), coords.dim)
        guide = make_guide(stand_in, config, dim=coords.dim)
        numpyro.handlers.seed(guide, 0)()  # set up the guide's prototype (runs the model once)
        check = flow_check(guide, params, potential, n=args.draws, seed=1)
        row = {
            "label": label,
            "analysis": analysis,
            **overrides,
            **check,
            "seconds": time.perf_counter() - start,
        }
        with path.open("a") as f:
            f.write(json.dumps(row) + "\n")
        print(f"[reuse] {json.dumps(row)}", flush=True)
        jax.clear_caches()


if __name__ == "__main__":
    main()
