"""Does solver error, rather than posterior geometry, set NUTS's step size?

    pixi run bench-energy [--points 4] [--steps 48] [--eps 0.1,0.2]

From ``--points`` published-posterior draws, with momenta drawn for the Laplace metric
(``outputs/explore/laplace_cov.npy``, the metric the pipeline's warmup starts from), runs
``--steps`` leapfrog steps at each step size three times: with the original's solver, with
``CALIBRATION_SOLVER`` and with a tight solver (Dopri5, ``rtol = atol = 1e-8``, mixing jumps
given to the controller). Same starts, same momenta. If the energy errors agree, the
trajectory's energy error comes from the posterior's curvature, and the step size NUTS adapts
to (and so its tree depth) is set by geometry rather than by solver noise; a looser or tighter
solver then changes only the cost per gradient.

Writes ``outputs/bench/energy_error.json``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent))

import diffrax  # noqa: E402
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

from common import pid, published_points, setup_and_model, solver_kwargs, write_json  # noqa: E402
from kiribati_tb.paths import REPO_ROOT  # noqa: E402


def make_value_and_grad(bm: Any, sites: tuple[str, ...]) -> Any:
    """Batched ``value_and_grad`` of the log density over flat vectors (sites in order)."""

    def flat(vec: Any) -> Any:
        return bm.log_density({s: vec[i] for i, s in enumerate(sites)})

    return jax.jit(jax.vmap(jax.value_and_grad(flat)))


def leapfrog(
    vg: Any, z0: np.ndarray, p0: np.ndarray, inv_mass: np.ndarray, eps: float, n: int
) -> dict[str, Any]:
    """``n`` leapfrog steps for every row of ``z0``; energy error after every step."""
    z = jnp.asarray(z0)
    p = jnp.asarray(p0)
    minv = jnp.asarray(inv_mass)

    def kinetic(pp: Any) -> Any:
        return 0.5 * jnp.einsum("bi,ij,bj->b", pp, minv, pp)

    logp, grad = vg(z)
    h0 = -logp + kinetic(p)
    errors = []
    for _ in range(n):
        p = p + 0.5 * eps * grad
        z = z + eps * p @ minv.T
        logp, grad = vg(z)
        p = p + 0.5 * eps * grad
        errors.append(np.asarray(-logp + kinetic(p) - h0))
    err = np.stack(errors, axis=1)  # (points, steps)
    return {"delta_h": err, "z_end": np.asarray(z)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--points", type=int, default=4)
    parser.add_argument("--steps", type=int, default=48)
    parser.add_argument("--eps", default="0.1,0.2")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    _setup, base = setup_and_model("original")
    _setup, after = setup_and_model("calibration")
    _setup, tight = setup_and_model(
        solver_kwargs(
            diffrax.Dopri5(),
            pid(1e-8, jumps=True),
            diffrax.RecursiveCheckpointAdjoint(checkpoints=8192),
            max_steps=32768,
        )
    )
    sites = tuple(sorted(base.prior_names()))
    cov = np.load(REPO_ROOT / "outputs" / "explore" / "laplace_cov.npy")
    mass = np.linalg.inv(cov)
    chol = np.linalg.cholesky(mass)
    points = published_points(base, args.points, seed=2)
    z0 = np.stack([[float(p[s]) for s in sites] for p in points])
    rng = np.random.default_rng(args.seed)
    p0 = rng.normal(size=z0.shape) @ chol.T
    vg = {
        "original": make_value_and_grad(base, sites),
        "calibration": make_value_and_grad(after, sites),
        "tight_1e-8": make_value_and_grad(tight, sites),
    }

    report: dict[str, Any] = {"points": args.points, "steps": args.steps, "runs": {}}
    for eps in [float(e) for e in args.eps.split(",")]:
        runs = {name: leapfrog(f, z0, p0, cov, eps, args.steps) for name, f in vg.items()}
        tight = runs["tight_1e-8"]
        row: dict[str, Any] = {}
        for name, run in runs.items():
            dh = run["delta_h"]
            row[name] = {
                "max_abs_dH": float(np.nanmax(np.abs(dh))),
                "rms_dH": float(np.sqrt(np.nanmean(dh**2))),
                "final_dH": dh[:, -1].tolist(),
                "nonfinite": int(np.sum(~np.isfinite(dh))),
            }
            if name != "tight_1e-8":
                row[name]["max_abs_dH_minus_tight"] = float(
                    np.nanmax(np.abs(dh - tight["delta_h"]))
                )
                row[name]["end_point_distance_to_tight"] = float(
                    np.max(np.linalg.norm(run["z_end"] - tight["z_end"], axis=1))
                )
        report["runs"][str(eps)] = row
        print("eps", eps, row, flush=True)
    print("wrote", write_json("energy_error", report))


if __name__ == "__main__":
    main()
