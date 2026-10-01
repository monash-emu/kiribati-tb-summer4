"""How often is a reverse-mode gradient non-finite, per solver configuration?

    pixi run bench-nan [--only base,pi_jumps] [--n 256]

docs/summer4-workarounds.md S5: with the committed solver, reverse-mode gradients are NaN at
17% of the 256-point prior design (``outputs/explore/design.npz``) although the log density is
finite, traced to rejected trial steps that visit invalid states. A non-finite gradient inside
a NUTS trajectory is a divergence. Fewer rejected steps (a PI controller) or a different solver
may change that rate, so this counts, for each ``levers.VARIANTS`` configuration, the design
points with a finite log density and a non-finite gradient, and the solver failures.

Writes ``outputs/bench/nan_rate.json``.
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

from common import setup_and_model, write_json  # noqa: E402
from kiribati_tb.paths import REPO_ROOT  # noqa: E402
import levers  # noqa: E402


def design_points(names: list[str], n: int) -> dict[str, Any]:
    """The first ``n`` prior design points, unconstrained, stacked."""
    data = np.load(REPO_ROOT / "outputs" / "explore" / "design.npz")
    return {k: jnp.asarray(data[f"z/{k}"][:n]) for k in names}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", default="base,pi_jumps,bosh3_pi_jumps")
    parser.add_argument("--n", type=int, default=256)
    parser.add_argument("--batch", type=int, default=16)
    args = parser.parse_args()
    report: dict[str, Any] = {"n": args.n}
    for name in args.only.split(","):
        variant = levers.VARIANTS[name]
        _s, bm = setup_and_model(variant.solver())
        z = design_points(list(bm.prior_names()), args.n)
        vg = jax.jit(
            lambda zz: jax.lax.map(jax.value_and_grad(bm.log_density), zz, batch_size=args.batch)
        )
        values, grads = vg(z)
        values = np.asarray(values)
        g = np.stack([np.asarray(grads[k]) for k in bm.prior_names()], axis=1)
        failed = values <= -1e29
        finite_ld = np.isfinite(values) & ~failed
        bad_grad = finite_ld & ~np.all(np.isfinite(g), axis=1)
        report[name] = {
            "solver_failures": int(failed.sum()),
            "finite_log_density": int(finite_ld.sum()),
            "nonfinite_grad_at_finite_ld": int(bad_grad.sum()),
            "nonfinite_fraction": float(bad_grad.sum() / max(1, finite_ld.sum())),
        }
        print(name, report[name], flush=True)
    print("wrote", write_json("nan_rate", report))


if __name__ == "__main__":
    main()
