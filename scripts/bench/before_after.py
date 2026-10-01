"""Before and after: the original's calibration solve against ``CALIBRATION_SOLVER``.

    pixi run bench-before-after [--rounds 30] [--points 24]

``before`` is the original's Dopri5 at ``rtol = atol = 1.4e-4`` with diffrax's default adjoint,
saving every year; ``after`` is ``kiribati_tb.calibration.CALIBRATION_SOLVER`` (Bosh3, PI
controller with the yearly mixing jumps, 1,024 checkpoints) saving only the target years, as
``bayesian_model`` now builds it. Both are timed interleaved in one process at the MAP
(eval and gradient). At the MAP and ``--points`` published-posterior draws, both are compared
with a reference gradient (Dopri5 at 1e-10, ``levers.REFERENCE``): relative gradient error,
log-density error, steps and finiteness.

Writes ``outputs/bench/before_after.json``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent))

import jax  # noqa: E402
import numpy as np  # noqa: E402

from common import flat_grad, interleaved, map_z, published_points, setup_and_model  # noqa: E402
from common import time_once, write_json  # noqa: E402
import levers  # noqa: E402


def _forward() -> dict[str, Any]:
    """The reference solve's backend with diffrax's forward-mode adjoint."""
    import dataclasses

    import diffrax

    backend = levers.REFERENCE.solver()["solver"]
    return {"solver": dataclasses.replace(backend, adjoint=diffrax.ForwardMode())}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rounds", type=int, default=30)
    parser.add_argument("--points", type=int, default=24)
    args = parser.parse_args()

    _s, ref = setup_and_model(levers.REFERENCE.solver())
    models = {"before": setup_and_model("original")[1], "after": setup_and_model("calibration")[1]}
    names = list(ref.prior_names())
    z_map = map_z(ref)
    points = [z_map] + published_points(ref, args.points, seed=5)
    ref_vg = jax.jit(jax.value_and_grad(ref.log_density))
    refs = [(float(v), flat_grad(g, names)) for v, g in (ref_vg(z) for z in points)]
    # The 1e-10 Dopri5 reverse-mode gradient is itself NaN at some draws (S5); there the
    # reference is the same solve differentiated in forward mode.
    fallback = None
    n_forward = 0
    for i, z in enumerate(points):
        if not np.all(np.isfinite(refs[i][1])):
            if fallback is None:
                fwd = levers.Variant(
                    lambda: {**levers.REFERENCE.solver(), **_forward()}, mode="fwd"
                )
                fallback = levers.build("reference_fwd", fwd)["grad"]
            v, g = fallback(z)
            refs[i] = (float(v), flat_grad(g, names))
            n_forward += 1

    report: dict[str, Any] = {"points": len(points), "rounds": args.rounds}
    report["reference_forward_mode_points"] = n_forward
    calls = {}
    for label, bm in models.items():
        vg = jax.jit(jax.value_and_grad(bm.log_density))
        ev = jax.jit(bm.log_density)
        first, _ = time_once(vg, z_map)
        time_once(ev, z_map)
        steps = jax.jit(lambda z, b=bm: b._run(b.merge_params(b.constrain(z))).solver)
        errs, lderr, n_steps, finite = [], [], [], []
        for z, (v_ref, g_ref) in zip(points, refs):
            v, g = vg(z)
            gv = flat_grad(g, names)
            finite.append(bool(np.all(np.isfinite(gv))))
            errs.append(float(np.linalg.norm(gv - g_ref) / np.linalg.norm(g_ref)))
            lderr.append(abs(float(v) - v_ref))
            n_steps.append(int(steps(z).num_steps))
        report[label] = {
            "per_point_grad_rel_err": errs,
            "nonfinite_grad_points": int(sum(not f for f in finite)),
            "grad_first_call_s": first,
            # Over the points where this configuration's gradient is finite.
            "grad_rel_err": {
                "median": float(np.nanmedian(errs)),
                "p90": float(np.nanpercentile(errs, 90)),
                "max": float(np.nanmax(errs)),
            },
            "ld_abs_err": {"median": float(np.median(lderr)), "max": float(np.max(lderr))},
            "steps": {"median": float(np.median(n_steps)), "max": int(np.max(n_steps))},
            "all_finite": all(finite),
        }
        print(label, report[label], flush=True)
        calls[f"{label}/eval"] = (ev, (z_map,))
        calls[f"{label}/grad"] = (vg, (z_map,))
    if args.rounds == 0:
        print("wrote", write_json("before_after_accuracy", report))
        return
    timings = interleaved(calls, args.rounds)
    report["timings"] = timings
    for kind in ("eval", "grad"):
        a, b = timings[f"before/{kind}"], timings[f"after/{kind}"]
        report[f"{kind}_speedup"] = {
            "wall_median": a["median_s"] / b["median_s"],
            "wall_p10": a["p10_s"] / b["p10_s"],
            "cpu_median": a["cpu_median_s"] / b["cpu_median_s"],
        }
        print(
            kind,
            {k: round(v["median_s"], 4) for k, v in ((l, timings[f"{l}/{kind}"]) for l in models)},
            report[f"{kind}_speedup"],
        )
    print("wrote", write_json("before_after", report))


if __name__ == "__main__":
    main()
