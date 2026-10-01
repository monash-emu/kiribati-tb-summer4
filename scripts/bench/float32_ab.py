"""In-process A/B: the calibration log density and gradient in float64 against float32.

    pixi run bench-float32 [--rounds 20] [--points 6]

``kiribati_tb`` turns on ``jax_enable_x64`` at import (summer2gen's precision). Here the float32
model is traced and run inside ``jax.experimental.disable_x64()``, so both versions live in one
process and are timed interleaved. JAX's trace cache keys on the x64 setting, so the two
compiled programs do not collide. Also reported: float32's log-density and gradient error
against float64 at the same solver settings, at the MAP and at ``--points`` posterior draws.

Writes ``outputs/bench/float32_ab.json``.
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
from jax.experimental import disable_x64  # noqa: E402

from common import flat_grad, interleaved, map_z, published_points, setup_and_model  # noqa: E402
from common import time_once, write_json  # noqa: E402


class Float32:
    """Call a jitted function with float32 inputs inside ``disable_x64``."""

    def __init__(self, fn: Any) -> None:
        self.fn = fn

    def __call__(self, z: Any) -> Any:
        with disable_x64():
            z32 = {k: jnp.asarray(np.asarray(v, dtype=np.float32)) for k, v in z.items()}
            return self.fn(z32)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rounds", type=int, default=20)
    parser.add_argument("--points", type=int, default=6)
    args = parser.parse_args()

    _s, bm = setup_and_model()
    names = list(bm.prior_names())
    z = map_z(bm)
    points = [z] + published_points(bm, args.points, seed=4)
    f64 = {"eval": jax.jit(bm.log_density), "grad": jax.jit(jax.value_and_grad(bm.log_density))}
    f32 = {
        "eval": Float32(jax.jit(lambda zz: bm.log_density(zz))),
        "grad": Float32(jax.jit(jax.value_and_grad(lambda zz: bm.log_density(zz)))),
    }
    for fns in (f64, f32):
        for fn in fns.values():
            time_once(fn, z)

    ld_err, g_err = [], []
    for p in points:
        v64, g64 = f64["grad"](p)
        v32, g32 = f32["grad"](p)
        a, b = flat_grad(g64, names), flat_grad(g32, names)
        ld_err.append(abs(float(v32) - float(v64)))
        g_err.append(float(np.linalg.norm(b - a) / np.linalg.norm(a)))
    calls = {f"float64/{k}": (fn, (z,)) for k, fn in f64.items()}
    calls |= {f"float32/{k}": (fn, (z,)) for k, fn in f32.items()}
    timings = interleaved(calls, args.rounds)
    report: dict[str, Any] = {
        "timings": timings,
        "ld_abs_diff": {"median": float(np.median(ld_err)), "max": float(np.max(ld_err))},
        "grad_rel_diff": {"median": float(np.median(g_err)), "max": float(np.max(g_err))},
    }
    for kind in ("eval", "grad"):
        a, b = timings[f"float64/{kind}"], timings[f"float32/{kind}"]
        report[f"{kind}_speedup"] = a["median_s"] / b["median_s"]
        report[f"{kind}_cpu_speedup"] = a["cpu_median_s"] / b["cpu_median_s"]
    for key, value in report.items():
        if key != "timings":
            print(key, value)
    print("wrote", write_json("float32_ab", report))


if __name__ == "__main__":
    main()
