"""The four reference calibrations of the base case, one per sampler, for MASSIVE.

Every arm samples the same posterior (``calibration_setup()``, the original's base case) with
the same front end (design, L-BFGS, Laplace metric; ``kiribati_tb.pipeline.calibrate``) and
the same stop rule: split R-hat ≤ 1.01 and bulk and tail ESS ≥ 400 for every parameter, or a
wall-time budget. They differ only in the MCMC kernel:

- ``nuts_td8``: dense-mass NUTS, ``max_tree_depth=8``, in the ridge-sheared coordinates;
- ``nuts_td5``: the same with ``max_tree_depth=5`` (cheaper iterations, more autocorrelation);
- ``sa``: numpyro's Sample Adaptive MCMC, gradient-free;
- ``ess``: numpyro's ensemble slice sampler (differential move), gradient-free, 64 walkers.

``scripts/reference_run.py`` runs one arm; ``scripts/cluster/submit_reference.sh`` submits all
four; ``scripts/compare_reference.py`` compares whatever has been written so far.
"""

from __future__ import annotations

from dataclasses import replace

from kiribati_tb.pipeline import PipelineConfig, StopCriteria

# Divergences are recorded and reported, not used to stop a reference run.
REFERENCE_CRITERIA = StopCriteria(rhat=1.01, ess=400.0, max_samples=10**7, max_divergence_frac=None)

_NUTS = PipelineConfig(
    kernel="nuts",
    num_chains=8,
    chain_method="parallel",
    cpus=8,
    warmup_rounds=(100, 100, 200, 200, 400),
    chunk=100,
    max_tree_depth=8,
    shear=True,
    criteria=REFERENCE_CRITERIA,
)

REFERENCE_ARMS: dict[str, PipelineConfig] = {
    "nuts_td8": _NUTS,
    "nuts_td5": replace(_NUTS, max_tree_depth=5),
    # One model evaluation per iteration: long chains, thinned 10x to keep checkpoints small.
    "sa": PipelineConfig(
        kernel="sa",
        num_chains=8,
        chain_method="parallel",
        cpus=8,
        warmup_rounds=(20000,),
        chunk=10000,
        thinning=10,
        criteria=REFERENCE_CRITERIA,
    ),
    # 64 walkers (over 3x the 19 dimensions) vectorised in one program on one device. One
    # iteration measured ~60 s here (every walker waits for the slowest slice), so warmup (which
    # only tunes the slice width) is short and each chunk is about 1.5 h.
    "ess": PipelineConfig(
        kernel="ess",
        num_chains=64,
        chain_method="vectorized",
        cpus=2,  # one vectorised program used ~1.4 cores here
        warmup_rounds=(300,),
        chunk=100,
        criteria=REFERENCE_CRITERIA,
    ),
}

# XLA host devices each arm needs (one per parallel chain). Scripts read this before JAX is
# imported, so it must stay a plain table: tests check it against REFERENCE_ARMS.
DEVICES: dict[str, int] = {"nuts_td8": 8, "nuts_td5": 8, "sa": 8, "ess": 1}
SMOKE_DEVICES = 2


def reference_config(arm: str, *, smoke: bool = False) -> PipelineConfig:
    """The arm's settings; ``smoke`` shrinks every stage so the whole path runs in minutes.

    Smoke NUTS arms cap the tree depth at 3, so ``nuts_td8`` and ``nuts_td5`` smoke-test the
    same path; the full arms differ only in ``max_tree_depth``.
    """
    if arm not in REFERENCE_ARMS:
        raise ValueError(f"Unknown arm {arm!r}; expected one of {sorted(REFERENCE_ARMS)}.")
    config = REFERENCE_ARMS[arm]
    criteria = config.criteria
    if not smoke:
        return config
    walkers = 8 if config.kernel == "ess" else 2
    return replace(
        config,
        design=16,
        starts=2,
        opt_steps=3,
        workers=2,
        num_chains=walkers,
        cpus=2 if config.kernel != "ess" else 1,
        warmup_rounds=(6, 6) if config.kernel == "nuts" else (20,),
        # A smoke run checks the code path, not the geometry: short trajectories keep it to
        # minutes (a depth-8 tree costs up to 255 gradients, minutes per iteration here).
        max_tree_depth=min(config.max_tree_depth, 3),
        chunk=4 if config.kernel == "nuts" else 20,
        thinning=1,
        tight_metric=False,
        criteria=replace(criteria, max_samples=8 if config.kernel == "nuts" else 40),
    )


__all__ = ["DEVICES", "REFERENCE_ARMS", "REFERENCE_CRITERIA", "SMOKE_DEVICES", "reference_config"]
