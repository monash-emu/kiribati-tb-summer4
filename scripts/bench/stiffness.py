"""How stiff is the model? Jacobian eigenvalues of the vector field along the trajectory.

    pixi run bench-stiffness [--points 8]

At the MAP, at ``--points`` published-posterior draws and at the prior's fastest corner (every
rate prior at its upper bound), solves 1850-2035 and evaluates the vector field's Jacobian
(``jax.jacfwd``, 160 x 160) every five years. The largest ``|Re lambda|`` bounds the step an
explicit method can take: a Runge-Kutta method with real-axis stability boundary ``b`` needs
``dt <= b / |lambda|``. Reports, per point, the largest ``|Re lambda|`` and the explicit-step
limits it implies for Dopri5 (b = 3.31), Tsit5 (3.5, approximately), Bosh3 (2.51) and Heun (2.0), and how
many vector-field evaluations each needs for 185 years at that limit.

Writes ``outputs/bench/stiffness.json``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent))

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

from summer4 import Compartments, SavePlan, SaveRequest  # noqa: E402
from summer4.jax.propertydata import PropertyData  # noqa: E402

from common import map_z, published_points, setup_and_model, write_json  # noqa: E402
from kiribati_tb.calibration import CALIBRATION_SOLVER  # noqa: E402
from kiribati_tb.model import END_TIME, START_TIME  # noqa: E402

# Real-axis stability boundary and new vector-field evaluations per step (FSAL counted once).
METHODS: dict[str, tuple[float, int]] = {
    "dopri5": (3.31, 6),
    "tsit5": (3.5, 6),
    "bosh3": (2.51, 3),
    "heun": (2.0, 2),
}
TIMES = np.arange(START_TIME, END_TIME + 0.1, 5.0)


def spectrum(bm: Any, z: Any) -> dict[str, Any]:
    """Largest ``|Re lambda|`` of the Jacobian at every five-year point of the trajectory."""
    compiled = bm.compiled
    params = bm.merge_params(bm.constrain(z))
    prepared = compiled.prepare(params)
    plan = SavePlan(requests={"y": SaveRequest(Compartments())}, ts=TIMES)
    run = {k: v for k, v in bm.run_kwargs.items()}
    result = compiled.run(params, save=plan, **run)
    ys = jnp.asarray(result["y"].values.data)

    def vf(t: Any, y: Any) -> Any:
        return compiled.vector_field(t, PropertyData(compiled.pmap, y), prepared).data

    jac = jax.vmap(jax.jacfwd(vf, argnums=1))(jnp.asarray(TIMES) + 0.5, ys)
    eig = np.linalg.eigvals(np.asarray(jac))
    re = np.max(np.abs(np.minimum(eig.real, 0.0)), axis=1)
    return {"max_abs_re": float(re.max()), "by_time": dict(zip(TIMES.tolist(), re.tolist()))}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--points", type=int, default=8)
    args = parser.parse_args()
    setup, bm = setup_and_model(dict(CALIBRATION_SOLVER))
    named = {"map": map_z(bm)}
    for i, z in enumerate(published_points(bm, args.points, seed=3)):
        named[f"posterior_{i}"] = z
    fast = {p.name: 0.999 * p.hi + 0.001 * p.lo for p in setup.priors if hasattr(p, "hi")}
    mid = {p.name: 0.5 * (p.hi + p.lo) for p in bm.site_priors() if hasattr(p, "hi")}
    named["prior_upper_corner"] = {
        k: jnp.asarray(v) for k, v in bm.unconstrain({**mid, **fast}).items()
    }
    report: dict[str, Any] = {"methods": METHODS, "points": {}}
    span = END_TIME - START_TIME
    for name, z in named.items():
        row = spectrum(bm, z)
        lam = row["max_abs_re"]
        row["explicit_limits"] = {
            m: {"dt_max": b / lam, "vf_evals_185y": int(np.ceil(span * lam / b)) * n}
            for m, (b, n) in METHODS.items()
        }
        report["points"][name] = row
        print(
            name,
            round(lam, 2),
            {m: v["vf_evals_185y"] for m, v in row["explicit_limits"].items()},
            flush=True,
        )
    print("wrote", write_json("stiffness", report))


if __name__ == "__main__":
    main()
