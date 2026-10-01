"""Maximum a posteriori fit with optax (replaces the original's nevergrad notebook, KI20).

Starts from the prior midpoints, runs Adam on the unconstrained joint density, and writes the
fitted parameters and the fit to the targets to ``outputs/find_map/``.

    pixi run python scripts/find_map.py [--steps 300] [--lr 0.05] [--analysis tpt_60]
"""

from __future__ import annotations

import argparse
import json
import time

import jax
import optax

from summer4.epi.calibration import Uniform

from kiribati_tb.calibration import bayesian_model, calibration_setup
from kiribati_tb.model import compile_model
from kiribati_tb.outputs import build_outputs, run_outputs
from kiribati_tb.paths import REPO_ROOT


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--analysis", default=None)
    args = parser.parse_args()
    out = REPO_ROOT / "outputs" / "find_map" / (args.analysis or "base")
    out.mkdir(parents=True, exist_ok=True)

    setup = calibration_setup(args.analysis)
    bm = bayesian_model(setup)
    start = {p.name: 0.5 * (p.lo + p.hi) for p in bm._sites if isinstance(p, Uniform)}
    z0 = bm.unconstrain(start)
    density = jax.jit(bm.log_density)
    t0 = time.time()
    fitted = bm.find_map(z0, steps=args.steps, optimizer=optax.adam(args.lr)).best_params
    elapsed = time.time() - t0
    fitted = {k: float(v) for k, v in fitted.items()}
    params = {**setup.params, **fitted}
    frame = run_outputs(compile_model(setup.config), build_outputs(setup.config), params)
    fit = {
        t.key: {
            "times": t.times.tolist(),
            "observed": t.values.tolist(),
            "modelled": frame.loc[t.times, t.key].tolist(),
        }
        for t in setup.targets
    }
    record = {
        "elapsed_s": elapsed,
        "args": vars(args),
        "log_density_start": float(density(z0)),
        "log_density_map": float(density(bm.unconstrain(fitted))),
        "params": fitted,
        "fit": fit,
    }
    (out / "map.json").write_text(json.dumps(record, indent=2))
    print(json.dumps({k: v for k, v in record.items() if k != "fit"}, indent=2))


if __name__ == "__main__":
    main()
