"""Where does one calibration gradient go? A breakdown of the default configuration.

    pixi run bench-profile [--rounds 15] [--vf-calls 2000]

At the MAP point, with ``--solver original`` (the original's Dopri5 at rtol = atol = 1.4e-4,
diffrax's default ``RecursiveCheckpointAdjoint``, every year saved) or ``--solver calibration``
(``kiribati_tb.calibration.CALIBRATION_SOLVER``, target years saved), this times, interleaved:

- ``log_density`` and ``value_and_grad(log_density)``;
- the solve alone with the calibration save plan (``run``) and with one save point
  (``run_1save``); output evaluation plus likelihood is ``log_density`` less ``run``;
- the gradient of the solve alone (sum of the final state) with one save point;
- the run-start stage (``prepare``: mixing stack and hoisted rates) and its gradient;
- one vector-field call and one vector-field VJP (averaged over ``--vf-calls`` calls in a
  ``lax.scan``, at varying ``t``).

It also records the solver's accepted/rejected steps, the jaxpr size (equations, nested
sub-jaxprs included) of the vector field, its VJP, ``log_density`` and the gradient, and the
trace+lower and XLA compile times. Writes ``outputs/bench/profile_<solver>.json``.
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

from common import (  # noqa: E402
    compile_seconds,
    count_eqns,
    interleaved,
    map_z,
    setup_and_model,
    time_once,
    write_json,
)
from kiribati_tb.model import END_TIME, START_TIME  # noqa: E402


def build(bm: Any, vf_calls: int) -> dict[str, Any]:
    """Jitted pieces of one log-density evaluation, keyed by name."""
    compiled = bm.compiled
    run_kwargs = dict(bm.run_kwargs)
    one_save = SavePlan(requests={"final": SaveRequest(Compartments())}, ts=np.asarray([END_TIME]))

    def params_of(z: Any) -> Any:
        return bm.merge_params(bm.constrain(z))

    def run(z: Any) -> Any:
        return bm._run(params_of(z))

    def run_1save(z: Any) -> Any:
        params = params_of(z)
        out = compiled.run(params, save=one_save, **run_kwargs)
        return out["final"].values.data

    def prepare(z: Any) -> Any:
        prepared = compiled.prepare(params_of(z))
        return prepared

    def prepare_sum(z: Any) -> Any:
        prepared = compiled.prepare(params_of(z))
        leaves = jax.tree.leaves(prepared)
        return sum(jnp.sum(jnp.asarray(x, dtype=float)) for x in leaves)

    ts = START_TIME + (END_TIME - START_TIME) * (jnp.arange(vf_calls) + 0.5) / vf_calls

    def vf_scan(prepared: Any, y: Any) -> Any:
        def body(acc: Any, t: Any) -> Any:
            dy = compiled.vector_field(t, PropertyData(compiled.pmap, y + 1e-9 * acc), prepared)
            return acc + dy.data, None

        acc, _ = jax.lax.scan(body, jnp.zeros_like(y), ts)
        return acc

    def vjp_scan(prepared: Any, y: Any) -> Any:
        def f(t: Any, yy: Any, pp: Any) -> Any:
            return compiled.vector_field(t, PropertyData(compiled.pmap, yy), pp).data

        def body(acc: Any, t: Any) -> Any:
            _, pull = jax.vjp(lambda yy, pp: f(t, yy, pp), y, prepared)
            gy, gp = pull(jnp.ones_like(y) + 1e-9 * acc)
            return acc + gy, None

        acc, _ = jax.lax.scan(body, jnp.zeros_like(y), ts)
        return acc

    return {
        "params_of": params_of,
        "run": run,
        "run_1save": run_1save,
        "prepare": prepare,
        "prepare_sum": prepare_sum,
        "vf_scan": vf_scan,
        "vjp_scan": vjp_scan,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rounds", type=int, default=15)
    parser.add_argument("--vf-calls", type=int, default=2000)
    parser.add_argument("--solver", default="original", choices=["original", "calibration"])
    args = parser.parse_args()

    _setup, bm = setup_and_model(args.solver)
    z = map_z(bm)
    parts = build(bm, args.vf_calls)
    report: dict[str, Any] = {
        "solver": args.solver,
        "rounds": args.rounds,
        "vf_calls": args.vf_calls,
    }

    ld = jax.jit(bm.log_density)
    vg = jax.jit(jax.value_and_grad(bm.log_density))
    run = jax.jit(parts["run"])
    run_1save = jax.jit(parts["run_1save"])
    grad_run_1save = jax.jit(jax.grad(lambda zz: jnp.sum(parts["run_1save"](zz))))
    prepare = jax.jit(parts["prepare"])
    grad_prepare = jax.jit(jax.grad(parts["prepare_sum"]))
    vf_scan = jax.jit(parts["vf_scan"])
    vjp_scan = jax.jit(parts["vjp_scan"])

    # Compile and warm everything once; record compile wall times (trace + lower + compile).
    first: dict[str, float] = {}
    first["log_density"], value = time_once(ld, z)
    first["grad"], (value2, grad) = time_once(vg, z)
    first["run"], result = time_once(run, z)
    first["run_1save"], _ = time_once(run_1save, z)
    first["grad_run_1save"], _ = time_once(grad_run_1save, z)
    first["prepare"], prepared = time_once(prepare, z)
    first["grad_prepare"], _ = time_once(grad_prepare, z)
    y_mid = (
        jnp.asarray(result["compartments"].values.data)[150]
        if "compartments" in result.outputs
        else None
    )
    if y_mid is None:
        y_mid = jnp.asarray(run_1save(z))[-1]
    first["vf_scan"], _ = time_once(vf_scan, prepared, y_mid)
    first["vjp_scan"], _ = time_once(vjp_scan, prepared, y_mid)
    report["first_call_s"] = first
    report["log_density"] = float(value)
    report["grad_norm"] = float(np.sqrt(sum(float(v) ** 2 for v in grad.values())))
    info = result.solver
    report["steps"] = {
        "num_steps": int(info.num_steps),
        "accepted": int(info.num_accepted_steps),
        "rejected": int(info.num_rejected_steps),
    }
    report["save_keys"] = list(result.outputs)
    report["save_points"] = int(np.asarray(result.times.values).size)

    calls = {
        "log_density": (ld, (z,)),
        "grad": (vg, (z,)),
        "run": (run, (z,)),
        "run_1save": (run_1save, (z,)),
        "grad_run_1save": (grad_run_1save, (z,)),
        "prepare": (prepare, (z,)),
        "grad_prepare": (grad_prepare, (z,)),
        "vf_scan": (vf_scan, (prepared, y_mid)),
        "vjp_scan": (vjp_scan, (prepared, y_mid)),
    }
    timings = interleaved(calls, args.rounds)
    for key in ("vf_scan", "vjp_scan"):
        timings[key]["per_call_us"] = 1e6 * timings[key]["median_s"] / args.vf_calls
    report["timings"] = timings

    # Jaxpr sizes.
    params = parts["params_of"](z)
    vf = lambda t, y, pp: bm.compiled.vector_field(t, PropertyData(bm.compiled.pmap, y), pp).data
    t0 = jnp.asarray(2000.5)
    sizes = {
        "vector_field": count_eqns(jax.make_jaxpr(vf)(t0, y_mid, prepared)),
        "vector_field_vjp": count_eqns(
            jax.make_jaxpr(lambda t, y, pp: jax.vjp(lambda a, b: vf(t, a, b), y, pp)[1](y))(
                t0, y_mid, prepared
            )
        ),
        "prepare": count_eqns(jax.make_jaxpr(lambda p: bm.compiled.prepare(p))(params)),
        "log_density": count_eqns(jax.make_jaxpr(bm.log_density)(z)),
        "grad": count_eqns(jax.make_jaxpr(jax.value_and_grad(bm.log_density))(z)),
    }
    report["jaxpr_eqns"] = sizes
    report["compile"] = {
        "log_density": compile_seconds(bm.log_density, z),
        "grad": compile_seconds(jax.value_and_grad(bm.log_density), z),
    }

    t = {k: v["median_s"] for k, v in timings.items()}
    vf_us = timings["vf_scan"]["per_call_us"]
    vjp_us = timings["vjp_scan"]["per_call_us"]
    n_steps = report["steps"]["num_steps"]
    # New vector-field evaluations per attempted step (first-same-as-last counted once).
    backend = bm.run_kwargs["solver"]
    name = backend if isinstance(backend, str) else backend.name
    per_step = {"dopri5": 6, "tsit5": 6, "bosh3": 3, "heun": 2, "ralston": 2}[name]
    report["vf_evals_per_step"] = per_step
    vf_in_solve = per_step * n_steps * vf_us * 1e-6
    report["breakdown"] = {
        "eval_s": t["log_density"],
        "grad_s": t["grad"],
        "grad_over_eval": t["grad"] / t["log_density"],
        "solve_with_saves_s": t["run"],
        "solve_1save_s": t["run_1save"],
        "saving_cost_s": t["run"] - t["run_1save"],
        # Output evaluation, likelihood and priors: everything log_density adds to the solve.
        "score_s": t["log_density"] - t["run"],
        "prepare_s": t["prepare"],
        "grad_prepare_s": t["grad_prepare"],
        "vf_evals_in_solve_s": vf_in_solve,
        "vf_share_of_solve_1save": vf_in_solve / t["run_1save"],
        "per_step_overhead_us": 1e6 * (t["run_1save"] - vf_in_solve) / n_steps,
        "backward_s": t["grad"] - t["log_density"],
        "grad_solve_1save_s": t["grad_run_1save"],
        "vjp_evals_lower_bound_s": per_step * n_steps * vjp_us * 1e-6,
        "vjp_over_vf": vjp_us / vf_us,
    }
    path = write_json(f"profile_{args.solver}", report)
    for key, value in report.items():
        if key != "save_keys":
            print(key, value)
    print("wrote", path)


if __name__ == "__main__":
    main()
