"""A/B process-level settings on one eval and one gradient: one subprocess per configuration.

    pixi run bench-ab [--rounds 3] [--reps 15] [--only default,searchsorted]

Some levers cannot be compared inside one process: ``XLA_FLAGS`` is read when JAX starts,
``jax_enable_x64`` is global, and diffrax's ``filter_jit`` caches a trace by model structure,
so a monkeypatch applied after the first trace is silently ignored. Each configuration runs
in its own subprocess, interleaved (A, B, C, A, B, C, ...) for ``--rounds`` rounds. Each
subprocess compiles, then times ``--reps`` evaluations and gradients at the MAP and reports
the log density and gradient, so a configuration that changes the answer is visible.

Configurations:

- ``default``: the calibration solver as committed.
- ``single_thread``: XLA CPU without the Eigen thread pool (one intra-op thread).
- ``legacy_runtime``: ``--xla_cpu_use_thunk_runtime=false``.
- ``no_fusion_emitters``: ``--xla_cpu_use_fusion_emitters=false``.
- ``searchsorted``: ``jnp.searchsorted`` (and so ``jnp.interp``) defaults to
  ``method="compare_all"``, which XLA fuses into one kernel, instead of ``"scan"``, which is a
  ``while`` loop of a binary search in every vector-field call. The summer4 prototype on
  ``perf/gradient-performance`` makes the same change in ``_eval_interp``.
- ``float32``: ``jax_enable_x64`` off after import (state and parameters in float32).
- ``summer4_src``: summer4 imported from ``$SUMMER4_SRC`` (a local checkout's ``src``).

Writes ``outputs/bench/process_ab.json``.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

CONFIGS: dict[str, dict[str, Any]] = {
    "default": {"env": {}, "args": []},
    "single_thread": {
        "env": {"XLA_FLAGS": "--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1"},
        "args": [],
    },
    "legacy_runtime": {"env": {"XLA_FLAGS": "--xla_cpu_use_thunk_runtime=false"}, "args": []},
    "no_fusion_emitters": {"env": {"XLA_FLAGS": "--xla_cpu_use_fusion_emitters=false"}, "args": []},
    "searchsorted": {"env": {}, "args": ["--patch", "searchsorted"]},
    "float32": {"env": {}, "args": ["--patch", "float32"]},
    # summer4 from a local checkout (e.g. the perf/gradient-performance branch) ahead of the
    # pinned one: set SUMMER4_SRC=/path/to/summer4/src. Skipped when unset.
    "summer4_src": {"env": {"PYTHONPATH": os.environ.get("SUMMER4_SRC", "")}, "args": []},
}


def patch_searchsorted() -> None:
    """Make ``method="compare_all"`` the default for ``jnp.searchsorted`` and ``jnp.interp``."""
    import functools

    import jax.numpy as jnp
    from jax._src.numpy import lax_numpy

    original = lax_numpy.searchsorted

    @functools.wraps(original)
    def compare_all(
        a: Any, v: Any, side: str = "left", sorter: Any = None, *, method: str = "compare_all"
    ) -> Any:
        return original(a, v, side=side, sorter=sorter, method=method)

    lax_numpy.searchsorted = compare_all
    jnp.searchsorted = compare_all


def worker(reps: int, patch: str | None, solver: str | None) -> None:
    """Time ``reps`` evals and gradients in this process; print one JSON line."""
    sys.path.insert(0, str(Path(__file__).parent))
    import jax
    import numpy as np

    import kiribati_tb  # noqa: F401 - enables x64
    import summer4

    print("SUMMER4 " + str(Path(summer4.__file__).parent), file=sys.stderr)

    if patch == "searchsorted":
        patch_searchsorted()
    elif patch == "float32":
        jax.config.update("jax_enable_x64", False)

    from common import last_cpu, map_z, setup_and_model, time_once
    import levers

    kwargs = None if solver is None else levers.VARIANTS[solver].solver()
    _setup, bm = setup_and_model(kwargs)
    z = map_z(bm)
    if patch == "float32":
        z = {k: jax.numpy.asarray(v, dtype=jax.numpy.float32) for k, v in z.items()}
    ld = jax.jit(bm.log_density)
    vg = jax.jit(jax.value_and_grad(bm.log_density))
    ld_first, _ = time_once(ld, z)
    vg_first, (value, grad) = time_once(vg, z)
    evals, grads, evals_cpu, grads_cpu = [], [], [], []
    for _ in range(reps):
        evals.append(time_once(ld, z)[0])
        evals_cpu.append(last_cpu())
        grads.append(time_once(vg, z)[0])
        grads_cpu.append(last_cpu())
    row = {
        "eval": evals,
        "grad": grads,
        "eval_cpu": evals_cpu,
        "grad_cpu": grads_cpu,
        "grad_first_s": vg_first,
        "ld_first_s": ld_first,
        "log_density": float(value),
        "grad_vec": {k: float(v) for k, v in grad.items()},
        "grad_norm": float(np.sqrt(sum(float(v) ** 2 for v in grad.values()))),
    }
    print("RESULT " + json.dumps(row))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--reps", type=int, default=15)
    parser.add_argument("--only", default=None)
    parser.add_argument("--solver", default=None, help="a levers.VARIANTS name for every config")
    parser.add_argument("--out", default="process_ab")
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--patch", default=None)
    args = parser.parse_args()
    if args.worker:
        worker(args.reps, args.patch, args.solver)
        return

    sys.path.insert(0, str(Path(__file__).parent))
    from common import quantiles, write_json

    names = list(CONFIGS) if args.only is None else args.only.split(",")
    if not os.environ.get("SUMMER4_SRC") and "summer4_src" in names:
        names.remove("summer4_src")
    samples: dict[str, dict[str, Any]] = {
        n: {
            "eval": [],
            "grad": [],
            "eval_cpu": [],
            "grad_cpu": [],
            "grad_first_s": [],
            "last": None,
        }
        for n in names
    }
    errors: dict[str, str] = {}
    for _ in range(args.rounds):
        for name in names:
            cmd = [sys.executable, __file__, "--worker", "--reps", str(args.reps)]
            cmd += CONFIGS[name]["args"]
            if args.solver:
                cmd += ["--solver", args.solver]
            env = {**os.environ, **CONFIGS[name]["env"]}
            proc = subprocess.run(cmd, env=env, capture_output=True, text=True)
            line = [l for l in proc.stdout.splitlines() if l.startswith("RESULT ")]
            if proc.returncode != 0 or not line:
                errors[name] = proc.stderr[-4000:]
                print(name, "FAILED\n", proc.stderr[-4000:], flush=True)
                continue
            row = json.loads(line[0][len("RESULT ") :])
            for key in ("eval", "grad", "eval_cpu", "grad_cpu"):
                samples[name][key] += row[key]
            samples[name]["grad_first_s"].append(row["grad_first_s"])
            samples[name]["last"] = row
            med = {
                k: round(sorted(row[k])[len(row[k]) // 2], 4)
                for k in ("eval", "grad", "eval_cpu", "grad_cpu")
            }
            print(name, med, "ld", row["log_density"], flush=True)
    report: dict[str, Any] = {"rounds": args.rounds, "reps": args.reps, "solver": args.solver}
    base = samples.get("default", {}).get("last")
    for name in names:
        last = samples[name]["last"]
        if last is None:
            report[name] = {"error": errors.get(name, "")}
            continue
        entry: dict[str, Any] = {
            "config": CONFIGS[name],
            "eval": quantiles(samples[name]["eval"], samples[name]["eval_cpu"]),
            "grad": quantiles(samples[name]["grad"], samples[name]["grad_cpu"]),
            "grad_first_call_s": quantiles(samples[name]["grad_first_s"]),
            "log_density": last["log_density"],
            "grad_norm": last["grad_norm"],
        }
        if base is not None:
            import numpy as np

            keys = sorted(base["grad_vec"])
            g0 = np.asarray([base["grad_vec"][k] for k in keys])
            g1 = np.asarray([last["grad_vec"][k] for k in keys])
            entry["log_density_minus_default"] = last["log_density"] - base["log_density"]
            entry["grad_rel_diff_vs_default"] = float(np.linalg.norm(g1 - g0) / np.linalg.norm(g0))
        report[name] = entry
    print(json.dumps(report, indent=2))
    print("wrote", write_json(args.out, report))


if __name__ == "__main__":
    main()
