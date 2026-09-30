"""Time one log-density evaluation against its gradient under several solver/adjoint choices.

Every variant scores the same point (the MAP from ``outputs/find_map/base/map.json`` if present,
else prior midpoints) and reports compile time, the median of ``--repeats`` timed calls, the
number of solver steps and the log-density error against a ``rtol = atol = 1e-10`` solve.

    pixi run python scripts/bench_gradients.py [--repeats 5] [--only eval,rev_default,...]

Results are written to ``outputs/bench/gradients.json``.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from collections.abc import Callable
from typing import Any

import diffrax
import jax
import numpy as np

from summer4.solvers import Diffrax

from kiribati_tb.calibration import bayesian_model, calibration_setup, map_point
from kiribati_tb.paths import REPO_ROOT

YEARS = np.arange(1851.0, 2035.0)


def pid(rtol: float, *, jumps: bool = False) -> Any:
    return diffrax.PIDController(rtol=rtol, atol=rtol, jump_ts=YEARS if jumps else None)


def solver(rtol: float, adjoint: Any = None, *, jumps: bool = False, max_steps: int | None = None):
    out: dict[str, Any] = {
        "solver": Diffrax(
            diffrax.Dopri5(), stepsize_controller=pid(rtol, jumps=jumps), adjoint=adjoint
        )
    }
    if max_steps is not None:
        out["max_steps"] = max_steps
    return out


# name -> (solver kwargs, mode) where mode is "eval", "rev" or "fwd"
VARIANTS: dict[str, tuple[Callable[[], dict[str, Any]], str]] = {
    "eval": (lambda: solver(1.4e-4), "eval"),
    "eval_jumps": (lambda: solver(1.4e-4, jumps=True), "eval"),
    "eval_1e-3_jumps": (lambda: solver(1e-3, jumps=True), "eval"),
    "eval_1e-6": (lambda: solver(1e-6), "eval"),
    "rev_default": (lambda: solver(1.4e-4), "rev"),
    "rev_1e-8": (lambda: solver(1e-8), "rev"),
    "rev_1e-6": (lambda: solver(1e-6), "rev"),
    "rev_max_steps_2048": (lambda: solver(1.4e-4, max_steps=2048), "rev"),
    "rev_checkpoints_1024": (
        lambda: solver(
            1.4e-4, diffrax.RecursiveCheckpointAdjoint(checkpoints=1024), max_steps=2048
        ),
        "rev",
    ),
    "rev_direct_2048": (lambda: solver(1.4e-4, diffrax.DirectAdjoint(), max_steps=2048), "rev"),
    "fwd": (lambda: solver(1.4e-4, diffrax.ForwardMode()), "fwd"),
    "fwd_max_steps_2048": (lambda: solver(1.4e-4, diffrax.ForwardMode(), max_steps=2048), "fwd"),
    "fwd_jumps": (lambda: solver(1.4e-4, diffrax.ForwardMode(), jumps=True), "fwd"),
    "fwd_1e-3_jumps": (lambda: solver(1e-3, diffrax.ForwardMode(), jumps=True), "fwd"),
}


def forward_log_density(bm: Any) -> Callable[[Any], Any]:
    """``bm.log_density`` built with numpyro's forward-mode initialisation.

    WORKAROUND(summer4): ``BayesianModel.potential_fn`` calls ``initialize_model`` without
    ``forward_mode_differentiation=True``, whose validity check takes a reverse-mode gradient,
    so it fails for a model solved with ``diffrax.ForwardMode()``.
    """
    from numpyro.infer.util import initialize_model

    potential = initialize_model(
        jax.random.PRNGKey(0), bm.numpyro_model(), forward_mode_differentiation=True
    )[1]
    return lambda z: -potential(z)


def block(tree: Any) -> Any:
    return jax.tree.map(lambda x: x.block_until_ready(), tree)


def time_fn(fn: Callable[[Any], Any], z: Any, repeats: int) -> tuple[float, float, Any]:
    start = time.perf_counter()
    out = block(fn(z))
    compile_s = time.perf_counter() - start
    times = []
    for _ in range(repeats):
        start = time.perf_counter()
        out = block(fn(z))
        times.append(time.perf_counter() - start)
    return compile_s, statistics.median(times), out


def solver_steps(bm: Any, z: Any) -> int | None:
    params = bm.merge_params(bm.constrain(z))
    info = jax.jit(lambda p: bm._run(p).solver)(params)
    return None if info is None else int(info.num_steps)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--only", default=None, help="comma-separated variant names")
    args = parser.parse_args()
    names = list(VARIANTS) if args.only is None else args.only.split(",")
    setup = calibration_setup()
    ref_bm = bayesian_model(setup, solver={"solver": "dopri5", "rtol": 1e-10, "atol": 1e-10})
    z = ref_bm.unconstrain(map_point(setup))
    ref = float(jax.jit(ref_bm.log_density)(z))
    rows: dict[str, dict[str, Any]] = {}
    for name in names:
        make, mode = VARIANTS[name]
        bm = bayesian_model(setup, solver=make())
        fn: Callable[[Any], Any]
        if mode == "eval":
            fn = jax.jit(bm.log_density)
        elif mode == "rev":
            fn = jax.jit(jax.value_and_grad(bm.log_density))
        else:
            density = forward_log_density(bm)
            fn = jax.jit(lambda z_, d=density: (d(z_), jax.jacfwd(d)(z_)))
        try:
            compile_s, median_s, out = time_fn(fn, z, args.repeats)
        except Exception as exc:  # noqa: BLE001 - recorded verbatim
            rows[name] = {"mode": mode, "error": f"{type(exc).__name__}: {exc}"[:2000]}
            print(name, rows[name], flush=True)
            continue
        value = float(out if mode == "eval" else out[0])
        row: dict[str, Any] = {
            "mode": mode,
            "compile_s": round(compile_s, 2),
            "median_s": round(median_s, 4),
            "log_density": value,
            "error_vs_1e-10": value - ref,
            "steps": solver_steps(bm, z),
        }
        if mode != "eval":
            grad = out[1]
            row["grad_norm"] = float(np.sqrt(sum(float(v) ** 2 for v in grad.values())))
            row["grad"] = {k: float(v) for k, v in grad.items()}
        rows[name] = row
        print(name, {k: v for k, v in row.items() if k != "grad"}, flush=True)
    out_dir = REPO_ROOT / "outputs" / "bench"
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "gradients.json"
    existing = json.loads(path.read_text()) if path.exists() else {}
    existing.update(rows)
    existing["_reference_log_density"] = ref
    path.write_text(json.dumps(existing, indent=2))


if __name__ == "__main__":
    main()
