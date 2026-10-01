"""Gradients for several chains at once: ``vmap`` in one device against ``pmap`` across devices.

    pixi run bench-parallel [--batches 1,2,4,8,16] [--devices 1,2,4,8] [--rounds 5]

``vmap``: one jitted ``vmap(value_and_grad(log_density))`` over ``B`` published-posterior
points, for each ``B`` in ``--batches``, timed interleaved. The adaptive solve runs every
lane until the slowest lane finishes, so per-lane cost depends on the spread in step counts.

``pmap``: one subprocess per device count ``D`` (``--xla_force_host_platform_device_count``),
``pmap(value_and_grad(log_density))`` over ``D`` points, and ``pmap(vmap(...))`` with
``--per-device`` points per device. On a shared machine the parallel numbers are bounded by
the free cores, which the report records (``os.getloadavg``).

Writes ``outputs/bench/parallel.json``.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any


def stack(points: list[dict[str, Any]]) -> dict[str, Any]:
    """Stack a list of site dicts along a new leading axis."""
    import jax.numpy as jnp

    return {k: jnp.stack([p[k] for p in points]) for k in points[0]}


def pmap_worker(devices: int, per_device: int, rounds: int, solver: str | None) -> None:
    """Time ``pmap`` (and ``pmap`` of ``vmap``) gradients on ``devices`` host devices."""
    sys.path.insert(0, str(Path(__file__).parent))
    import jax

    from common import published_points, quantiles, setup_and_model, time_once
    import levers

    kwargs = None if solver is None else levers.VARIANTS[solver].solver()
    _setup, bm = setup_and_model(kwargs)
    assert jax.device_count() == devices, jax.devices()
    points = published_points(bm, devices * per_device, seed=1)
    vg = jax.value_and_grad(bm.log_density)
    one = {k: v.reshape((devices,) + v.shape[1:]) for k, v in stack(points[:devices]).items()}
    many = {k: v.reshape((devices, per_device) + v.shape[1:]) for k, v in stack(points).items()}
    f1 = jax.pmap(vg)
    fm = jax.pmap(jax.vmap(vg))
    first1, _ = time_once(f1, one)
    firstm, _ = time_once(fm, many)
    t1, tm = [], []
    for _ in range(rounds):
        t1.append(time_once(f1, one)[0])
        tm.append(time_once(fm, many)[0])
    row = {
        "devices": devices,
        "per_device": per_device,
        "pmap": quantiles(t1),
        "pmap_vmap": quantiles(tm),
        "first_call_s": [first1, firstm],
        "loadavg": os.getloadavg(),
    }
    print("RESULT " + json.dumps(row))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batches", default="1,2,4,8,16")
    parser.add_argument("--devices", default="1,2,4,8")
    parser.add_argument("--per-device", type=int, default=4)
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--solver", default=None, help="a levers.VARIANTS name")
    parser.add_argument("--out", default="parallel")
    parser.add_argument("--pmap-worker", type=int, default=0)
    args = parser.parse_args()
    if args.pmap_worker:
        pmap_worker(args.pmap_worker, args.per_device, args.rounds, args.solver)
        return

    sys.path.insert(0, str(Path(__file__).parent))
    import jax

    from common import interleaved, published_points, setup_and_model, time_once, write_json
    import levers

    kwargs = None if args.solver is None else levers.VARIANTS[args.solver].solver()
    _setup, bm = setup_and_model(kwargs)
    batches = [int(b) for b in args.batches.split(",")]
    points = published_points(bm, max(batches), seed=1)
    vg = jax.jit(jax.vmap(jax.value_and_grad(bm.log_density)))
    calls = {}
    first = {}
    for b in batches:
        zb = stack(points[:b])
        first[b], _ = time_once(vg, zb)
        calls[f"vmap{b}"] = (vg, (zb,))
    timings = interleaved(calls, args.rounds)
    report: dict[str, Any] = {"solver": args.solver, "loadavg": os.getloadavg(), "vmap": {}}
    base = timings["vmap1"]["median_s"] if "vmap1" in timings else None
    for b in batches:
        t = timings[f"vmap{b}"]
        report["vmap"][b] = {
            **t,
            "first_call_s": first[b],
            "per_gradient_s": t["median_s"] / b,
            "throughput_vs_1": (b * base / t["median_s"]) if base else None,
        }
        print("vmap", b, report["vmap"][b], flush=True)

    report["pmap"] = {}
    for d in [int(x) for x in args.devices.split(",")]:
        env = {**os.environ, "XLA_FLAGS": f"--xla_force_host_platform_device_count={d}"}
        cmd = [
            sys.executable,
            __file__,
            "--pmap-worker",
            str(d),
            "--per-device",
            str(args.per_device),
            "--rounds",
            str(args.rounds),
        ]
        if args.solver:
            cmd += ["--solver", args.solver]
        proc = subprocess.run(cmd, env=env, capture_output=True, text=True)
        line = [l for l in proc.stdout.splitlines() if l.startswith("RESULT ")]
        if proc.returncode != 0 or not line:
            report["pmap"][d] = {"error": proc.stderr[-3000:]}
            print("pmap", d, "FAILED", proc.stderr[-3000:], flush=True)
            continue
        row = json.loads(line[0][len("RESULT ") :])
        row["per_gradient_pmap_s"] = row["pmap"]["median_s"] / d
        row["per_gradient_pmap_vmap_s"] = row["pmap_vmap"]["median_s"] / (d * args.per_device)
        report["pmap"][d] = row
        print("pmap", d, row, flush=True)
    print("wrote", write_json(args.out, report))


if __name__ == "__main__":
    main()
