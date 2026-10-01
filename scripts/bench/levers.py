"""Solver-level levers: time per eval and gradient, steps, and gradient accuracy.

    pixi run bench-levers [--rounds 10] [--points 8] [--only base,pi_jumps,...]

Every variant is one ``BayesianModel`` differing only in its solver backend (and, for
``sparse_saves``, its save grid). For each variant this records:

- eval and ``value_and_grad`` time at the MAP, interleaved across variants for ``--rounds``
  rounds (median and interquartile range);
- first-call (trace + compile) time of the gradient;
- accepted and rejected steps at the MAP and at ``--points`` draws of the published posterior;
- log-density error and relative gradient error ``|g - g_ref| / |g_ref|`` at the same points,
  against a reference gradient from Dopri5 at ``rtol = atol = 1e-10`` with the yearly mixing
  jumps given to the controller (``--check-fd`` also checks that reference against central
  finite differences of a 1e-10 log density).

Writes ``outputs/bench/levers.json``.
"""

from __future__ import annotations

import argparse
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent))

import diffrax  # noqa: E402
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

from common import (  # noqa: E402
    flat_grad,
    interleaved,
    map_z,
    pid,
    published_points,
    setup_and_model,
    solver_kwargs,
    time_once,
    write_json,
)

CKPT = diffrax.RecursiveCheckpointAdjoint(checkpoints=1024)
# Hairer & Wanner's PI controller for step sizes limited by stability, as diffrax's
# PIDController documentation suggests for stiff problems: damps the accept/reject cycling.
PI = {"pcoeff": 0.4, "icoeff": 0.3}


@dataclass(frozen=True)
class Variant:
    """One solver configuration: a ``run`` kwargs factory, a mode, and a save-grid choice."""

    solver: Callable[[], dict[str, Any]]
    mode: str = "rev"
    sparse_saves: bool = False
    note: str = ""


def _s(solver: Any = None, controller: Any = None, adjoint: Any = None, **kw: Any) -> Callable:
    return lambda: solver_kwargs(solver, controller, adjoint, **kw)


VARIANTS: dict[str, Variant] = {
    "base": Variant(_s(), note="committed: Dopri5, I-controller 1.4e-4, default checkpoints"),
    "ckpt1024": Variant(_s(adjoint=CKPT), note="RecursiveCheckpointAdjoint(checkpoints=1024)"),
    "fwd": Variant(_s(adjoint=diffrax.ForwardMode()), mode="fwd", note="ForwardMode, jacfwd"),
    "jumps": Variant(_s(controller=pid(1.4e-4, jumps=True)), note="jump_ts at integer years"),
    "pi": Variant(_s(controller=pid(1.4e-4, **PI)), note="PI controller 0.4/0.3"),
    "pi_jumps": Variant(_s(controller=pid(1.4e-4, jumps=True, **PI))),
    "pi_jumps_ckpt": Variant(_s(controller=pid(1.4e-4, jumps=True, **PI), adjoint=CKPT)),
    "dtmax1_jumps": Variant(_s(controller=pid(1.4e-4, jumps=True, dtmax=1.0))),
    "tol1e-3_pi_jumps": Variant(_s(controller=pid(1e-3, jumps=True, **PI))),
    "tol1e-5_pi_jumps": Variant(_s(controller=pid(1e-5, jumps=True, **PI))),
    "tol1e-6_pi_jumps": Variant(_s(controller=pid(1e-6, jumps=True, **PI))),
    "tsit5_pi_jumps": Variant(_s(diffrax.Tsit5(), pid(1.4e-4, jumps=True, **PI))),
    "bosh3_pi_jumps": Variant(_s(diffrax.Bosh3(), pid(1.4e-4, jumps=True, **PI))),
    "bosh3_pi_jumps_ckpt": Variant(_s(diffrax.Bosh3(), pid(1.4e-4, jumps=True, **PI), CKPT)),
    "heun_pi_jumps": Variant(_s(diffrax.Heun(), pid(1.4e-4, jumps=True, **PI))),
    "ralston_pi_jumps": Variant(_s(diffrax.Ralston(), pid(1.4e-4, jumps=True, **PI))),
    "kvaerno5_jumps": Variant(_s(diffrax.Kvaerno5(), pid(1.4e-4, jumps=True))),
    "kvaerno3_jumps": Variant(_s(diffrax.Kvaerno3(), pid(1.4e-4, jumps=True))),
    # Constant steps that divide one year land on every mixing jump; they need an explicit
    # save grid (the default grid is every step).
    "const_dopri5_dt0.25": Variant(
        lambda: {**solver_kwargs(diffrax.Dopri5(), diffrax.ConstantStepSize()), "dt": 0.25},
        sparse_saves=True,
        note="no error control; stable while |lambda| < 13",
    ),
    "const_bosh3_dt0.2": Variant(
        lambda: {**solver_kwargs(diffrax.Bosh3(), diffrax.ConstantStepSize()), "dt": 0.2},
        sparse_saves=True,
        note="no error control; stable while |lambda| < 12.5",
    ),
    "sparse_saves": Variant(_s(), sparse_saves=True, note="save only target years and year-1"),
    "recommended": Variant(
        _s(diffrax.Bosh3(), pid(1.4e-4, jumps=True, **PI), CKPT),
        sparse_saves=True,
        note="Bosh3, PI + jumps, checkpoints=1024, target-year saves",
    ),
}

REFERENCE = Variant(
    lambda: solver_kwargs(
        diffrax.Dopri5(),
        pid(1e-10, jumps=True),
        diffrax.RecursiveCheckpointAdjoint(checkpoints=4096),
        max_steps=65536,
    )
)


def build(name: str, variant: Variant) -> dict[str, Any]:
    """Model, jitted eval and gradient, and a step counter for ``variant``."""
    setup, bm = setup_and_model(variant.solver(), target_years=variant.sparse_saves)
    if variant.mode == "fwd":
        from numpyro.infer.util import initialize_model

        potential = initialize_model(
            jax.random.PRNGKey(0), bm.numpyro_model(), forward_mode_differentiation=True
        )[1]
        density = lambda z: -potential(z)  # noqa: E731
        grad_fn = jax.jit(lambda z: (density(z), jax.jacfwd(density)(z)))
        eval_fn = jax.jit(density)
    else:
        grad_fn = jax.jit(jax.value_and_grad(bm.log_density))
        eval_fn = jax.jit(bm.log_density)

    def stats(z: Any) -> Any:
        info = bm._run(bm.merge_params(bm.constrain(z))).solver
        return info.num_accepted_steps, info.num_rejected_steps

    return {"bm": bm, "grad": grad_fn, "eval": eval_fn, "stats": jax.jit(stats), "name": name}


def central_fd(
    eval_fn: Callable[[Any], Any], z: dict[str, Any], h: float = 1e-5
) -> dict[str, float]:
    """Central finite differences of ``eval_fn`` in every unconstrained coordinate."""
    out = {}
    for k in z:
        up = {**z, k: z[k] + h}
        dn = {**z, k: z[k] - h}
        out[k] = (float(eval_fn(up)) - float(eval_fn(dn))) / (2 * h)
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rounds", type=int, default=10)
    parser.add_argument("--points", type=int, default=8)
    parser.add_argument("--only", default=None)
    parser.add_argument("--check-fd", action="store_true")
    parser.add_argument("--out", default="levers")
    args = parser.parse_args()
    names = list(VARIANTS) if args.only is None else args.only.split(",")

    ref = build("reference", REFERENCE)
    bm0 = ref["bm"]
    names_z = list(bm0.prior_names())
    z_map = map_z(bm0)
    points = [z_map] + published_points(bm0, args.points, seed=0)
    ref_vals, ref_grads = [], []
    start = time.perf_counter()
    for z in points:
        v, g = ref["grad"](z)
        ref_vals.append(float(v))
        ref_grads.append(flat_grad(g, names_z))
    report: dict[str, Any] = {
        "points": len(points),
        "reference": {"seconds": time.perf_counter() - start, "log_density_map": ref_vals[0]},
    }
    if args.check_fd:
        fd = central_fd(ref["eval"], z_map)
        g = ref_grads[0]
        fdv = flat_grad(fd, names_z)
        report["reference"]["fd_rel_error"] = float(np.linalg.norm(g - fdv) / np.linalg.norm(fdv))
        report["reference"]["fd_max_abs_error"] = float(np.max(np.abs(g - fdv)))
        print("reference vs FD", report["reference"], flush=True)

    built: dict[str, dict[str, Any]] = {}
    rows: dict[str, dict[str, Any]] = {}
    for name in names:
        variant = VARIANTS[name]
        try:
            b = build(name, variant)
            first, _ = time_once(b["grad"], z_map)
            time_once(b["eval"], z_map)
            errs, lderr, acc, rej, finite = [], [], [], [], []
            for z, v_ref, g_ref in zip(points, ref_vals, ref_grads):
                v, g = b["grad"](z)
                gv = flat_grad(g, names_z)
                finite.append(bool(np.all(np.isfinite(gv))))
                errs.append(float(np.linalg.norm(gv - g_ref) / np.linalg.norm(g_ref)))
                lderr.append(abs(float(v) - v_ref))
                a, r = b["stats"](z)
                acc.append(int(a))
                rej.append(int(r))
        except Exception as exc:  # noqa: BLE001 - recorded verbatim
            rows[name] = {"error": f"{type(exc).__name__}: {exc}"[:3000], "note": variant.note}
            print(name, "FAILED", rows[name]["error"][:500], flush=True)
            continue
        built[name] = b
        rows[name] = {
            "note": variant.note,
            "mode": variant.mode,
            "grad_first_call_s": first,
            "accepted_map": acc[0],
            "rejected_map": rej[0],
            "accepted_median": float(np.median(acc)),
            "rejected_median": float(np.median(rej)),
            "grad_rel_err_map": errs[0],
            "grad_rel_err_median": float(np.median(errs)),
            "grad_rel_err_max": float(np.max(errs)),
            "ld_abs_err_map": lderr[0],
            "ld_abs_err_median": float(np.median(lderr)),
            "ld_abs_err_max": float(np.max(lderr)),
            "all_finite": all(finite),
        }
        print(name, rows[name], flush=True)

    calls = {}
    for name, b in built.items():
        calls[f"{name}/eval"] = (b["eval"], (z_map,))
        calls[f"{name}/grad"] = (b["grad"], (z_map,))
    timings = interleaved(calls, args.rounds)
    base_grad = timings.get("base/grad", {})
    for name in built:
        rows[name]["eval"] = timings[f"{name}/eval"]
        rows[name]["grad"] = timings[f"{name}/grad"]
        if base_grad:
            g = timings[f"{name}/grad"]
            rows[name]["grad_speedup_vs_base"] = base_grad["median_s"] / g["median_s"]
            rows[name]["grad_cpu_speedup_vs_base"] = base_grad["cpu_median_s"] / g["cpu_median_s"]
    report["variants"] = rows
    path = write_json(args.out, report)
    print(
        f"{'variant':24s} {'eval ms':>8s} {'grad ms':>8s} {'cpu ms':>8s} {'x base':>7s} "
        f"{'x cpu':>6s} {'acc':>5s} {'rej':>5s} {'g err med':>10s} {'g err max':>10s} "
        f"{'ld err med':>10s}"
    )
    for name, r in rows.items():
        if "error" in r:
            print(f"{name:24s} ERROR")
            continue
        print(
            f"{name:24s} {1e3 * r['eval']['median_s']:8.1f} {1e3 * r['grad']['median_s']:8.1f} "
            f"{1e3 * r['grad']['cpu_median_s']:8.1f} "
            f"{r.get('grad_speedup_vs_base', float('nan')):7.2f} "
            f"{r.get('grad_cpu_speedup_vs_base', float('nan')):6.2f} {r['accepted_map']:5d} "
            f"{r['rejected_map']:5d} {r['grad_rel_err_median']:10.2e} {r['grad_rel_err_max']:10.2e} "
            f"{r['ld_abs_err_median']:10.2e}"
        )
    print("wrote", path)


if __name__ == "__main__":
    main()
