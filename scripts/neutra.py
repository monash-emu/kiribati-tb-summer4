"""NeuTra calibration: fit a normalising flow by SVI, then NUTS in its warped coordinates.

    pixi run python scripts/neutra.py --name base [--from outputs/neutra/inputs] \\
        [--svi-steps 2000] [--particles 4] [--flow iaf] [--chains 4] [--ess 400]

``--from`` is a folder holding ``optima.npz`` and ``metric.npy`` (the multi-start optimum and
the Laplace covariance at it, as ``kiribati_tb.pipeline.calibrate`` writes them); the flow is
fitted in coordinates whitened by that covariance (after the ridge shear unless ``--no-shear``).
``--flow none --dense-mass`` is the matched baseline: the same NUTS, warmup and stop rule
on the whitened coordinates without a flow (the pipeline's Laplace-started dense-metric NUTS,
sheared unless ``--no-shear``). ``--warm-start <run folder>`` instead reuses that run's coordinates and starts SVI from its
flow (with ``--svi-steps 0`` the flow is used unchanged), for configurations of the grid or the
sensitivity analyses. Output goes to ``outputs/neutra/<name>/``: ``coords.json``, ``flow/``
(SVI state, losses, ``progress.csv`` with the ELBO and the flow's PSIS ``k_hat``), ``nuts/``
(warmup rounds, checkpointed draws, ``num_steps.npy``) and ``summary.json``. Every stage is
checkpointed: rerunning the same command after a kill resumes it.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", default="base")
    parser.add_argument("--from", dest="source", default="outputs/neutra/inputs")
    parser.add_argument("--warm-start", default=None, help="a previous run folder")
    parser.add_argument("--analysis", default=None)
    parser.add_argument("--override", action="append", default=[], metavar="NAME=VALUE")
    parser.add_argument("--no-shear", action="store_true")
    parser.add_argument(
        "--solver",
        default="calibration",
        choices=["calibration", "original"],
        help="calibration_solver (Bosh3, default) or the original's Dopri5 (docs/neutra.md runs)",
    )
    parser.add_argument("--flow", default="iaf", choices=["iaf", "bnaf", "none"])
    parser.add_argument("--num-flows", type=int, default=2)
    parser.add_argument("--hidden", default="38,38")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--particles", type=int, default=4)
    parser.add_argument("--svi-steps", type=int, default=2000)
    parser.add_argument("--svi-chunk", type=int, default=50)
    parser.add_argument("--check-draws", type=int, default=128)
    parser.add_argument("--chains", type=int, default=4)
    parser.add_argument("--chain-method", default="parallel", choices=["parallel", "vectorized"])
    parser.add_argument("--warmup-rounds", default="100,100,200")
    parser.add_argument("--chunk", type=int, default=100)
    parser.add_argument("--max-tree-depth", type=int, default=8)
    parser.add_argument("--dense-mass", action="store_true", help="dense metric (baseline)")
    parser.add_argument("--rhat", type=float, default=1.01)
    parser.add_argument("--ess", type=float, default=400.0)
    parser.add_argument("--max-samples", type=int, default=5000)
    parser.add_argument("--max-hours", type=float, default=None)
    parser.add_argument(
        "--max-divergence-frac",
        type=float,
        default=0.02,
        help="stop when divergences exceed this fraction (the pipeline's default); 1 disables",
    )
    parser.add_argument("--fit-only", action="store_true", help="stop after the flow")
    parser.add_argument("--progress", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    if args.chain_method == "parallel":
        if "jax" in sys.modules:
            raise RuntimeError("Set XLA host devices before importing jax.")
        flag = f"--xla_force_host_platform_device_count={int(args.chains)}"
        os.environ["XLA_FLAGS"] = (os.environ.get("XLA_FLAGS", "") + " " + flag).strip()

    import numpyro  # noqa: F401 - import before arviz; see docs/summer4-workarounds.md, S4
    import numpy as np

    from summer4.epi.calibration import workflow as wf

    from kiribati_tb.calibration import (
        ORIGINAL_CALIBRATION_SOLVER,
        bayesian_model,
        calibration_setup,
    )
    from kiribati_tb.neutra import (
        NeutraConfig,
        WhitenedCoordinates,
        fit_flow,
        load_params,
        run_neutra,
    )
    from kiribati_tb.paths import REPO_ROOT
    from kiribati_tb.pipeline import StopCriteria, ridge_shear

    overrides = {k: float(v) for k, v in (item.split("=", 1) for item in args.override)}
    solver = ORIGINAL_CALIBRATION_SOLVER if args.solver == "original" else None
    bm = bayesian_model(calibration_setup(args.analysis, overrides), solver=solver)
    folder = REPO_ROOT / "outputs" / "neutra" / args.name
    init_flow = None
    if (folder / "coords.json").exists():
        coords = WhitenedCoordinates.from_json(json.loads((folder / "coords.json").read_text()))
    elif args.warm_start is not None:
        source = Path(args.warm_start)
        coords = WhitenedCoordinates.from_json(json.loads((source / "coords.json").read_text()))
    else:
        source = REPO_ROOT / args.source
        best = wf.Candidates.load(source / "optima.npz").best(1)
        mode = {k: float(np.asarray(v)[0]) for k, v in best.z.items()}
        cov = np.load(source / "metric.npy")
        shear = None if args.no_shear else ridge_shear(bm, mode, cov)
        coords = WhitenedCoordinates.from_laplace(mode, cov, shear=shear)
    if args.warm_start is not None:
        init_flow = load_params(Path(args.warm_start) / "flow")
    config = NeutraConfig(
        flow=args.flow,
        num_flows=args.num_flows,
        hidden=tuple(int(h) for h in args.hidden.split(",")),
        learning_rate=args.lr,
        particles=args.particles,
        svi_steps=args.svi_steps,
        svi_chunk=args.svi_chunk,
        check_draws=args.check_draws,
        num_chains=args.chains,
        chain_method=args.chain_method,
        warmup_rounds=tuple(int(n) for n in args.warmup_rounds.split(",")),
        chunk=args.chunk,
        max_tree_depth=args.max_tree_depth,
        dense_mass=args.dense_mass,
        progress_bar=args.progress,
    )
    criteria = StopCriteria(
        rhat=args.rhat,
        ess=args.ess,
        max_samples=args.max_samples,
        max_seconds=None if args.max_hours is None else args.max_hours * 3600.0,
        max_divergence_frac=args.max_divergence_frac,
    )
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "args.json").write_text(json.dumps(vars(args), indent=2))
    if args.fit_only:
        (folder / "coords.json").write_text(json.dumps(coords.to_json()))
        fit_flow(bm, coords, folder, config, init_flow=init_flow, seed=args.seed)
        return
    run = run_neutra(
        bm, coords, folder, config, init_flow=init_flow, criteria=criteria, seed=args.seed
    )
    print(json.dumps(run.summary(), indent=2, default=float), flush=True)


if __name__ == "__main__":
    main()
