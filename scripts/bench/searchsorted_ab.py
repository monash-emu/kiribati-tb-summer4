"""In-process A/B: ``jnp.searchsorted``'s ``while``-loop binary search against ``compare_all``.

    pixi run bench-searchsorted [--rounds 20]

summer4 (``_eval_interp`` at ``c9548d5``) finds an interpolation knot with
``jnp.searchsorted`` (sigmoidal and step tables) and ``jnp.interp`` (linear tables), whose
default ``method="scan"`` lowers to an XLA ``while`` loop. The Kiribati vector field has three
such loops (death-rate table, births, treatment success): each is a sequence of small kernels
per iteration in every one of the ~4,300 vector-field calls of a solve, and of its reverse pass.

Model A is built first and compiled. Then ``jnp.searchsorted`` is patched to default to
``method="compare_all"`` (one fused comparison against every knot; summer4's prototype on
``perf/gradient-performance`` does the same in ``_eval_interp``), and model B is compiled
with a fresh ``prepare_fn`` object, which changes the model digest so diffrax's ``filter_jit``
traces it again rather than reusing A's trace. A and B are then timed interleaved: one
vector-field call (in a ``lax.scan``), one log density, one gradient. Also reported: the
answers (must be identical) and XLA kernels per Runge-Kutta stage.

Writes ``outputs/bench/searchsorted_ab.json``.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent))

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

from summer4.jax.propertydata import PropertyData  # noqa: E402

from common import interleaved, map_z, setup_and_model, time_once, write_json  # noqa: E402
from kiribati_tb import model as km  # noqa: E402
from kiribati_tb.model import END_TIME, START_TIME  # noqa: E402
from process_ab import patch_searchsorted  # noqa: E402


def vf_scan(bm: Any, calls: int) -> Any:
    """Jitted: ``calls`` vector-field evaluations at spread-out times, summed."""
    compiled = bm.compiled
    ts = START_TIME + (END_TIME - START_TIME) * (jnp.arange(calls) + 0.5) / calls

    def run(z: Any) -> Any:
        prepared = compiled.prepare(bm.merge_params(bm.constrain(z)))
        y = compiled.initial_state(prepared).data

        def body(acc: Any, t: Any) -> Any:
            dy = compiled.vector_field(t, PropertyData(compiled.pmap, y + 1e-12 * acc), prepared)
            return acc + dy.data, None

        return jax.lax.scan(body, jnp.zeros_like(y), ts)[0]

    return jax.jit(run)


def loop_kernels(fn: Any, z: Any) -> dict[str, int]:
    """``while`` loops and the largest loop body's kernel count in the compiled HLO."""
    text = jax.jit(fn).lower(z).compile().as_text()
    bodies: dict[str, int] = {}
    current = None
    for line in text.splitlines():
        head = re.match(r"^(?:ENTRY )?%?([\w.\-]+) .*\{$", line)
        if head:
            current = head.group(1)
            bodies[current] = 0
        elif current and line.startswith("  ") and "=" in line:
            if not re.search(r" (parameter|constant|get-tuple-element|tuple|bitcast)\(", line):
                bodies[current] += 1
    whiles = re.findall(r" while\(.*?body=%?([\w.\-]+)", text)
    return {"while_loops": len(whiles), "largest_loop_body_kernels": max(bodies[b] for b in whiles)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rounds", type=int, default=20)
    parser.add_argument("--vf-calls", type=int, default=1000)
    args = parser.parse_args()

    models: dict[str, Any] = {}
    _s, models["scan"] = setup_and_model()
    z = map_z(models["scan"])
    fns: dict[str, dict[str, Any]] = {}

    def compile_all(name: str) -> None:
        bm = models[name]
        fns[name] = {
            "vf": vf_scan(bm, args.vf_calls),
            "eval": jax.jit(bm.log_density),
            "grad": jax.jit(jax.value_and_grad(bm.log_density)),
        }
        for fn in fns[name].values():
            time_once(fn, z)

    compile_all("scan")
    kernels = {"scan": loop_kernels(models["scan"].log_density, z)}
    patch_searchsorted()
    original = km.add_mixing_stack
    km.compile_model.__globals__["add_mixing_stack"] = lambda p: original(p)  # new digest
    _s, models["compare_all"] = setup_and_model()
    km.compile_model.__globals__["add_mixing_stack"] = original
    assert models["compare_all"].compiled._digest != models["scan"].compiled._digest
    compile_all("compare_all")
    kernels["compare_all"] = loop_kernels(models["compare_all"].log_density, z)

    answers = {}
    for name in models:
        value, grad = fns[name]["grad"](z)
        answers[name] = (float(value), np.asarray([float(grad[k]) for k in sorted(grad)]))
    calls = {f"{name}/{kind}": (fn, (z,)) for name in fns for kind, fn in fns[name].items()}
    timings = interleaved(calls, args.rounds)
    report: dict[str, Any] = {"rounds": args.rounds, "kernels": kernels, "timings": timings}
    for kind in ("vf", "eval", "grad"):
        a, b = timings[f"scan/{kind}"], timings[f"compare_all/{kind}"]
        report[f"{kind}_speedup"] = a["median_s"] / b["median_s"]
        report[f"{kind}_cpu_speedup"] = a["cpu_median_s"] / b["cpu_median_s"]
    report["vf_us_per_call"] = {
        name: 1e6 * timings[f"{name}/vf"]["median_s"] / args.vf_calls for name in fns
    }
    report["log_density_diff"] = answers["compare_all"][0] - answers["scan"][0]
    report["grad_max_abs_diff"] = float(
        np.max(np.abs(answers["compare_all"][1] - answers["scan"][1]))
    )
    for key, value in report.items():
        if key != "timings":
            print(key, value)
    print("wrote", write_json("searchsorted_ab", report))


if __name__ == "__main__":
    main()
