"""K4 acceptance 1: recover five parameters from targets simulated at known values.

Simulates the nine Normal targets from the ``hetero_baseline`` golden parameters, calibrates a
five-parameter subset with numpyro AIES (the gradient-free stand-in for pymc's DEMetropolisZ),
and writes the posterior and a summary to ``outputs/synthetic_recovery/``.

    pixi run python scripts/synthetic_recovery.py [--warmup 1000] [--samples 1000] [--walkers 16]
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import yaml

from summer4.epi.calibration import Uniform

from kiribati_tb.calibration import (
    CalibrationSetup,
    TargetSpec,
    bayesian_model,
    calibration_setup,
    target_specs,
)
from kiribati_tb.model import compile_model
from kiribati_tb.outputs import build_outputs, run_outputs
from kiribati_tb.params import read_parameter_sheet
from kiribati_tb.paths import GOLDEN, REPO_ROOT

SUBSET: tuple[str, ...] = (
    "raw_transmission_rate",
    "recent_detection_rate",
    "clinical_progression_rate",
    "prev_se_cleared_tst",
    "prev_se_subclin_lowinf_cxr",
)
OUT = REPO_ROOT / "outputs" / "synthetic_recovery"


def synthetic_setup() -> tuple[CalibrationSetup, dict[str, float]]:
    """Targets simulated at the golden parameters; priors on the five-parameter subset."""
    truth = yaml.safe_load((GOLDEN / "hetero_baseline" / "params.yaml").read_text())["params"]
    base = calibration_setup()
    frame = run_outputs(compile_model(base.config), build_outputs(base.config), truth)
    specs = [
        TargetSpec(s.output, {y: float(frame.loc[y, s.output]) for y in s.observations}, s.tol_pct)
        for s in target_specs()
    ]
    priors = [p for p in base.priors if p.name in SUBSET]
    assert sorted(p.name for p in priors) == sorted(SUBSET)
    setup = CalibrationSetup(base.config, dict(truth), priors, [s.target() for s in specs])
    return setup, {name: float(truth[name]) for name in SUBSET}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--warmup", type=int, default=1000)
    parser.add_argument("--samples", type=int, default=1000)
    parser.add_argument("--walkers", type=int, default=16)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    setup, truth = synthetic_setup()
    bm = bayesian_model(setup)
    start = time.time()
    idata = bm.sample(
        "aies",
        num_warmup=args.warmup,
        num_samples=args.samples,
        num_chains=args.walkers,
        seed=args.seed,
    ).idata
    elapsed = time.time() - start
    OUT.mkdir(parents=True, exist_ok=True)
    idata.to_netcdf(OUT / "idata.nc")
    summary: dict[str, object] = {"elapsed_s": elapsed, "args": vars(args), "params": {}}
    for name, value in truth.items():
        draws = np.asarray(idata.posterior[name]).reshape(-1)
        lo, med, hi = np.quantile(draws, [0.025, 0.5, 0.975])
        prior = next(p for p in setup.priors if p.name == name)
        assert isinstance(prior, Uniform)
        summary["params"][name] = {  # type: ignore[index]
            "truth": value,
            "q025": float(lo),
            "median": float(med),
            "q975": float(hi),
            "prior": [float(prior.lo), float(prior.hi)],
            "covered": bool(lo <= value <= hi),
        }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
