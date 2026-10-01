"""Fast, checkpointed calibration from summer4's workflow pieces.

The posterior is the original's (same priors, targets and likelihood); the sampler is not.
The pipeline is:

1. **Design**: ``wf.lhs`` over the priors, scored with ``wf.evaluate``.
2. **Multi-start optimisation**: the best design points are optimised with L-BFGS-B
   (:func:`lbfgs`), several starts at once on separate cores.
3. **Mode check**: :func:`distinct_optima` groups the optimised points, so a second mode is
   reported rather than silently mixed into (or left out of) the chains.
4. **NUTS**: chains start from the optima within :data:`SEED_WINDOW` nats of the best, warm up
   in checkpointed rounds (:class:`StagedWarmup`, dense mass matrix from a Laplace start),
   then sample in chunks with
   ``wf.sample_until``. The stop callable is a :class:`Checkpoint`: after every chunk it writes
   the draws (``idata.nc``) and the sampler state to disk, and stops on rank-normalised split
   R-hat and bulk/tail ESS over every draw so far, so a killed job resumes where it stopped.

:func:`calibrate` is the one-call convenience; its docstring lists the lines it stands for.
"""

from __future__ import annotations

import json
import math
import pickle
import time
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd

from summer4.epi.calibration import workflow as wf

# Chains are seeded from optima whose log density is within this many nats of the best. A
# 19-dimensional posterior's typical set lies ~d/2 ≈ 10 nats below its mode, so an optimum
# more than 25 nats down carries no appreciable posterior mass and would only stall R-hat.
SEED_WINDOW = 25.0


# --------------------------------------------------------------------------------------------
# Optimisation
# --------------------------------------------------------------------------------------------


def forward_log_density(bm: Any) -> Callable[[Any], Any]:
    """``bm.log_density`` for a model solved with ``diffrax.ForwardMode()``.

    WORKAROUND(summer4): ``BayesianModel.potential_fn`` builds its potential with
    ``numpyro.infer.util.initialize_model`` without ``forward_mode_differentiation=True``, and
    that call's validity check takes a reverse-mode gradient, which a forward-mode solve
    refuses. See docs/summer4-workarounds.md, W6.
    """
    from numpyro.infer.util import initialize_model

    potential = initialize_model(
        jax.random.PRNGKey(0), bm.numpyro_model(), forward_mode_differentiation=True
    )[1]
    return lambda z: -potential(z)


def lbfgs(
    bm: Any,
    starts: wf.Candidates,
    *,
    fallback: Any = None,
    workers: int = 8,
    maxiter: int = 300,
    gtol: float = 1e-4,
    batch_size: int = 16,
) -> wf.Candidates:
    """Minimise the potential from every start with scipy's L-BFGS-B; scored ``Candidates``.

    Each start runs its own host loop over one jitted value-and-gradient, and ``workers``
    threads run starts concurrently (XLA releases the GIL, so they use separate cores). Where
    the reverse-mode gradient is not finite, ``fallback`` (the same model solved with
    ``diffrax.ForwardMode()``, see :func:`kiribati_tb.calibration.forward_mode_solver`) supplies
    a forward-mode gradient. A failed solve is returned to L-BFGS as a large finite value, so
    its line search backs off.

    WORKAROUND(summer4): ``wf.optimize`` vmaps its starts into one program. With an adaptive
    solve and a line search every lane waits for the slowest, and 24 L-BFGS starts did not
    finish one 10-step chunk in 18 minutes; see docs/summer4-workarounds.md, W7.
    """
    from concurrent.futures import ThreadPoolExecutor

    from scipy.optimize import minimize

    start_time = time.perf_counter()
    sites = starts.sites

    def potential(vec: Any) -> Any:
        return bm.potential_fn({s: vec[i] for i, s in enumerate(sites)})

    value_and_grad = jax.jit(jax.value_and_grad(potential))
    forward: Any = None
    if fallback is not None:
        density = forward_log_density(fallback)
        forward = jax.jit(
            jax.jacfwd(lambda vec: -density({s: vec[i] for i, s in enumerate(sites)}))
        )

    def fun(x: np.ndarray) -> tuple[float, np.ndarray]:
        value, grad = value_and_grad(jnp.asarray(x))
        value, grad = float(value), np.asarray(grad, dtype=float)
        if not np.all(np.isfinite(grad)) and forward is not None and value < 1e29:
            grad = np.asarray(forward(jnp.asarray(x)), dtype=float)
        if not (np.isfinite(value) and np.all(np.isfinite(grad))) or value > 1e29:
            return 1e10, np.zeros_like(x)
        return value, grad

    def one(i: int) -> tuple[np.ndarray, int]:
        x0 = np.asarray([float(np.asarray(starts.z[s])[i]) for s in sites])
        res = minimize(
            fun, x0, jac=True, method="L-BFGS-B", options={"maxiter": maxiter, "gtol": gtol}
        )
        return np.asarray(res.x), int(res.nfev)

    fun(np.asarray([float(np.asarray(starts.z[s])[0]) for s in sites]))  # compile once
    with ThreadPoolExecutor(max_workers=int(workers)) as pool:
        results = list(pool.map(one, range(len(starts))))
    z = {s: np.asarray([r[0][j] for r in results]) for j, s in enumerate(sites)}
    record = wf.StageRecord(
        stage="lbfgs",
        settings={
            "maxiter": maxiter,
            "gtol": gtol,
            "workers": workers,
            "evaluations": [r[1] for r in results],
        },
        seconds=time.perf_counter() - start_time,
    )
    optimised = wf.Candidates.from_z(bm, z)
    optimised = wf.evaluate(bm, optimised, batch_size=batch_size)
    return replace(optimised, history=starts.history + (record,) + optimised.history)


def design(bm: Any, n: int, *, seed: int = 0, batch_size: int = 16) -> wf.Candidates:
    """``n`` Latin-hypercube points over the priors, scored."""
    return wf.evaluate(bm, wf.lhs(bm, n, seed=seed), batch_size=batch_size)


@dataclass(frozen=True)
class Optimum:
    """One group of optimised points that converged to the same place."""

    log_density: float
    members: tuple[int, ...]
    z: dict[str, float]


def distinct_optima(
    candidates: wf.Candidates, *, scale: Mapping[str, float] | None = None, tol: float = 0.5
) -> list[Optimum]:
    """Group optimised points that lie within ``tol`` of each other (unconstrained, scaled).

    Greedy single linkage from the best point down. ``scale`` divides each site's
    unconstrained coordinate (default 1). Returned best first.
    """
    ok = np.asarray(candidates.ok, dtype=bool)
    ld = np.where(ok, np.asarray(candidates.log_density), -np.inf)
    order = [int(i) for i in np.argsort(-ld) if np.isfinite(ld[i])]
    sites = candidates.sites
    scales = np.asarray([1.0 if scale is None else float(scale[s]) for s in sites])
    pts = np.stack([np.asarray(candidates.z[s]) for s in sites], axis=1) / scales
    groups: list[list[int]] = []
    for i in order:
        for g in groups:
            if np.min(np.linalg.norm(pts[g] - pts[i], axis=1)) < tol:
                g.append(i)
                break
        else:
            groups.append([i])
    return [
        Optimum(
            log_density=float(ld[g[0]]),
            members=tuple(g),
            z={s: float(np.asarray(candidates.z[s])[g[0]]) for s in sites},
        )
        for g in groups
    ]


def seeds(candidates: wf.Candidates, window: float = SEED_WINDOW) -> wf.Candidates:
    """The optimised points within ``window`` nats of the best, best first."""
    ld = np.where(
        np.asarray(candidates.ok, dtype=bool), np.asarray(candidates.log_density), -np.inf
    )
    keep = np.flatnonzero(ld >= np.max(ld) - window)
    return candidates.take(keep[np.argsort(-ld[keep])])


# --------------------------------------------------------------------------------------------
# NUTS, checkpointed
# --------------------------------------------------------------------------------------------


KERNELS: tuple[str, ...] = ("nuts", "sa", "ess")
# Per-iteration fields each kernel records for the checkpoint's diagnostics.
KERNEL_FIELDS: dict[str, tuple[str, ...]] = {
    "nuts": ("diverging", "num_steps", "accept_prob"),
    "sa": ("diverging", "accept_prob"),
    "ess": (),
}


def make_mcmc(
    bm: Any,
    config: PipelineConfig,
    num_warmup: int,
    *,
    inverse_mass_matrix: np.ndarray | None = None,
    step_size: float = 1.0,
    shear: RidgeShear | None = None,
) -> Any:
    """A numpyro ``MCMC`` for ``config.kernel`` on ``bm``, ``config.chunk`` draws per run.

    - ``"nuts"``: dense-mass NUTS. ``inverse_mass_matrix`` (sites sorted, e.g. from
      :func:`laplace_covariance`) and ``step_size`` are where adaptation starts: from the
      identity metric this posterior needs 255-step trajectories until the first mass-matrix
      window closes. With ``shear`` the kernel runs on ``shear.potential(bm)`` in sheared
      coordinates (:class:`RidgeShear`); draws then come back unconstrained and
      ``Checkpoint(postprocess=...)`` maps them to parameters.
    - ``"sa"``: numpyro's Sample Adaptive MCMC (gradient-free), one chain per core. Its
      initialisation fails under ``chain_method="vectorized"`` with given ``init_params``
      (numpyro 0.22), so use ``"parallel"`` or ``"sequential"``.
    - ``"ess"``: numpyro's ensemble slice sampler with the differential move,
      ``config.num_chains`` walkers vectorised in one program; ``ess_max_steps`` and
      ``ess_max_iter`` cap the stepping-out and shrinking loops, which under ``vmap`` run every
      walker to the slowest one.

    ``chain_method="parallel"`` needs one XLA host device per chain
    (``XLA_FLAGS=--xla_force_host_platform_device_count=N`` before JAX is first imported, as
    the scripts do).
    """
    from numpyro.infer import ESS, MCMC, NUTS, SA

    target = (
        {"model": bm.numpyro_model()} if shear is None else {"potential_fn": shear.potential(bm)}
    )
    chain_method = config.chain_method
    if config.kernel == "nuts":
        kernel = NUTS(
            **target,
            dense_mass=True,
            target_accept_prob=config.target_accept_prob,
            max_tree_depth=config.max_tree_depth,
            inverse_mass_matrix=(
                None if inverse_mass_matrix is None else jnp.asarray(inverse_mass_matrix)
            ),
            step_size=step_size,
        )
    elif config.kernel == "sa":
        if chain_method == "vectorized":
            raise ValueError("numpyro's SA cannot vectorise seeded chains; use 'parallel'.")
        kernel = SA(**target)
    elif config.kernel == "ess":
        kernel = ESS(
            **target,
            moves={ESS.DifferentialMove(): 1.0},
            max_steps=config.ess_max_steps,
            max_iter=config.ess_max_iter,
        )
        chain_method = "vectorized"  # numpyro runs ensemble walkers in one program
    else:
        raise ValueError(f"Unknown kernel {config.kernel!r}; expected one of {KERNELS}.")
    return MCMC(
        kernel,
        num_warmup=int(num_warmup),
        num_samples=int(config.chunk),
        num_chains=int(config.num_chains),
        thinning=int(config.thinning),
        chain_method=chain_method,
        progress_bar=config.progress_bar,
    )


def sample_chunks(
    mcmc: Any,
    stop: Callable[[Mapping[str, Any]], tuple[bool, str] | None],
    *,
    seed: int = 0,
    extra_fields: tuple[str, ...] = (),
) -> tuple[bool, str]:
    """Sample a warmed-up ``mcmc`` in chunks of ``num_samples`` until ``stop`` decides.

    ``wf.sample_until`` with the kernel's own diagnostic fields collected: it requests only
    ``diverging``, so tree depth and acceptance would be lost (docs/summer4-workarounds.md,
    W8). ``stop`` is called after every chunk with ``{"seconds": ...}`` for this call, the
    shape of an ``MCMCRun.progress`` row a :class:`Checkpoint` reads.
    """
    from jax import random

    if mcmc.post_warmup_state is None:
        raise ValueError("sample_chunks needs a warmed-up MCMC (post_warmup_state set).")
    key = random.PRNGKey(int(seed))
    seconds = 0.0
    # A fresh MCMC given a saved state (a resumed job) still initialises its kernel on the
    # first run, and a kernel on a potential function needs positions for that; they are
    # otherwise ignored in favour of the state.
    init = jax.tree.map(np.asarray, mcmc.post_warmup_state.z)
    while True:
        key, sub = random.split(key)
        start = time.perf_counter()
        mcmc.run(sub, extra_fields=extra_fields, init_params=init)
        init = None
        jax.block_until_ready(mcmc.last_state)
        seconds += time.perf_counter() - start
        mcmc.post_warmup_state = mcmc.last_state
        decision = stop({"seconds": seconds})
        if decision is not None:
            return decision


def diagnostics(samples: Mapping[str, np.ndarray]) -> pd.DataFrame:
    """Rank-normalised split R-hat, bulk and tail ESS per site (arviz), ``(chain, draw)`` arrays."""
    import arviz as az

    rows = []
    for name, arr in samples.items():
        x = np.asarray(arr)
        flat = x.reshape(x.shape[0], x.shape[1], -1)
        for i in range(flat.shape[2]):
            label = name if x.ndim == 2 else f"{name}[{i}]"
            xi = flat[:, :, i]
            rows.append(
                {
                    "site": label,
                    "rhat": float(az.rhat(xi)),
                    "ess_bulk": float(az.ess(xi, method="bulk")),
                    "ess_tail": float(az.ess(xi, method="tail", prob=(0.05, 0.95))),
                }
            )
    return pd.DataFrame(rows).set_index("site")


@dataclass
class StopCriteria:
    """When a :class:`Checkpoint` ends sampling (``None`` disables a criterion).

    Converged means every site has R-hat ≤ ``rhat`` and bulk and tail ESS ≥ ``ess``. Budgets
    count every draw and second since the run first started, across restarts.
    """

    rhat: float = 1.01
    ess: float = 400.0
    max_samples: int = 20000
    max_seconds: float | None = None
    max_divergence_frac: float | None = 0.02


@dataclass
class Checkpoint:
    """A ``wf.sample_until`` stop callable that saves the run after every chunk.

    Called once per chunk with ``wf.MCMCRun``'s progress row, it appends the chunk's draws
    (read from the caller's ``mcmc``), writes ``idata.nc``, ``mcmc_state.pkl`` (the last
    sampler state: positions, adapted step size and mass matrix, RNG keys) and
    ``progress.csv`` to ``folder`` (``postprocess`` first maps the kernel's draws to
    parameters, for a kernel on a potential function), and decides with :class:`StopCriteria` on the diagnostics
    of **every** draw so far, including draws from before a restart.

    Resume with :meth:`resume`, which reloads the draws and hands the saved state to a fresh
    ``MCMC`` as its ``post_warmup_state``.
    """

    mcmc: Any
    folder: Path
    criteria: StopCriteria = field(default_factory=StopCriteria)
    samples: dict[str, np.ndarray] = field(default_factory=dict)
    diverging: np.ndarray | None = None
    rows: list[dict[str, Any]] = field(default_factory=list)
    seconds_before: float = 0.0
    postprocess: Callable[[dict[str, Any]], dict[str, Any]] | None = None
    cpus: int = 1
    deadline: float | None = None  # wall clock (time.time()) this job must stop by

    def __post_init__(self) -> None:
        self.folder = Path(self.folder)
        self.folder.mkdir(parents=True, exist_ok=True)

    # -- the stop callable ------------------------------------------------------------------

    def __call__(self, row: Mapping[str, Any]) -> tuple[bool, str] | None:
        chunk = self.mcmc.get_samples(group_by_chain=True)
        if self.postprocess is not None:
            chunk = self.postprocess(chunk)
        for name, value in chunk.items():
            new = np.asarray(value)
            old = self.samples.get(name)
            self.samples[name] = new if old is None else np.concatenate([old, new], axis=1)
        extra = self.mcmc.get_extra_fields(group_by_chain=True)
        if "diverging" in extra:
            div = np.asarray(extra["diverging"])
            self.diverging = (
                div if self.diverging is None else np.concatenate([self.diverging, div], axis=1)
            )
        diag = diagnostics(self.samples)
        draws = int(next(iter(self.samples.values())).shape[1])
        seconds = self.seconds_before + float(row.get("seconds") or 0.0)
        steps = np.asarray(extra["num_steps"]) if "num_steps" in extra else None
        accept = np.asarray(extra["accept_prob"]) if "accept_prob" in extra else None
        ess_min = min(float(diag["ess_bulk"].min()), float(diag["ess_tail"].min()))
        record = {
            "chunk": len(self.rows) + 1,
            "draws_per_chain": draws,
            "rhat_max": float(diag["rhat"].max()),
            "ess_bulk_min": float(diag["ess_bulk"].min()),
            "ess_tail_min": float(diag["ess_tail"].min()),
            "divergence_frac": (None if self.diverging is None else float(np.mean(self.diverging))),
            "chunk_divergence_frac": (
                float(np.mean(extra["diverging"])) if "diverging" in extra else None
            ),
            "chunk_leapfrog_mean": None if steps is None else float(np.mean(steps)),
            "chunk_treedepth_mean": None if steps is None else float(np.mean(np.log2(steps + 1))),
            "chunk_accept_mean": None if accept is None else float(np.mean(accept)),
            "seconds": seconds,
            "cpu_hours": seconds * self.cpus / 3600.0,
            "ess_min_per_cpu_hour": ess_min / max(seconds * self.cpus / 3600.0, 1e-9),
        }
        decision = self.decide(record)
        previous = self.rows[-1]["seconds"] if self.rows else self.seconds_before
        chunk_seconds = max(seconds - float(previous or 0.0), 0.0)
        if (
            decision is None
            and self.deadline is not None
            and time.time() + 1.25 * chunk_seconds > self.deadline
        ):
            decision = (False, "deadline")  # another chunk would overrun this job
        record["decision"] = None if decision is None else decision[1]
        self.rows.append(record)
        self.save(seconds)
        report = {"record": record, "per_parameter": diag.round(5).to_dict(orient="index")}
        _atomic(self.folder / "diagnostics.json", lambda p: p.write_text(json.dumps(report)))
        print(f"[checkpoint] {json.dumps(record)}", flush=True)
        return decision

    def decide(self, record: Mapping[str, Any]) -> tuple[bool, str] | None:
        c = self.criteria
        div = record["divergence_frac"]
        if c.max_divergence_frac is not None and div is not None and div > c.max_divergence_frac:
            return False, "divergences"
        if (
            record["rhat_max"] <= c.rhat
            and min(record["ess_bulk_min"], record["ess_tail_min"]) >= c.ess
        ):
            return True, "diagnostics"
        if record["draws_per_chain"] >= c.max_samples:
            return False, "max_samples"
        if c.max_seconds is not None and record["seconds"] >= c.max_seconds:
            return False, "max_seconds"
        return None

    # -- persistence ------------------------------------------------------------------------

    @property
    def progress(self) -> pd.DataFrame:
        return pd.DataFrame(self.rows).set_index("chunk")

    @property
    def idata(self) -> Any:
        import arviz as az

        groups: dict[str, Any] = {"posterior": dict(self.samples)}
        if self.diverging is not None:
            groups["sample_stats"] = {"diverging": self.diverging}
        return az.from_dict(groups)

    def save(self, seconds: float) -> None:
        """Write draws, sampler state and progress; each file is replaced atomically."""
        _atomic(self.folder / "idata.nc", lambda p: self.idata.to_netcdf(p))
        state = jax.device_get(self.mcmc.last_state)
        _atomic(self.folder / "mcmc_state.pkl", lambda p: p.write_bytes(pickle.dumps(state)))
        _atomic(self.folder / "progress.csv", lambda p: self.progress.to_csv(p))
        meta = {
            "seconds": seconds,
            "chunks": len(self.rows),
            "criteria": asdict(self.criteria),
            "cpus": self.cpus,
        }
        _atomic(self.folder / "checkpoint.json", lambda p: p.write_text(json.dumps(meta)))

    @classmethod
    def resume(
        cls,
        folder: Path,
        mcmc: Any,
        criteria: StopCriteria | None = None,
        postprocess: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
    ) -> Checkpoint | None:
        """Reload a checkpoint into a fresh ``mcmc``, or ``None`` if there is none."""
        import arviz as az

        folder = Path(folder)
        if not (folder / "checkpoint.json").exists():
            return None
        meta = json.loads((folder / "checkpoint.json").read_text())
        idata = az.from_netcdf(folder / "idata.nc")
        samples = {k: np.asarray(v.values) for k, v in idata.posterior.data_vars.items()}
        diverging = None
        if "sample_stats" in idata.children and "diverging" in idata.sample_stats:
            diverging = np.asarray(idata.sample_stats["diverging"].values)
        state = pickle.loads((folder / "mcmc_state.pkl").read_bytes())
        mcmc.post_warmup_state = jax.tree.map(jnp.asarray, state)
        rows = [
            {k: (None if isinstance(v, float) and np.isnan(v) else v) for k, v in r.items()}
            for r in pd.read_csv(folder / "progress.csv").to_dict("records")
        ]
        return cls(
            mcmc=mcmc,
            folder=folder,
            criteria=criteria or StopCriteria(**meta["criteria"]),
            samples=samples,
            diverging=diverging,
            rows=rows,
            seconds_before=float(meta["seconds"]),
            postprocess=postprocess,
            cpus=int(meta.get("cpus", 1)),
        )


def _atomic(path: Path, write: Callable[[Path], Any]) -> None:
    tmp = path.with_name(path.name + ".tmp")
    write(tmp)
    tmp.replace(path)


class DeadlineReached(RuntimeError):
    """A stage stopped before the job's deadline; rerun with the same folder to resume."""


@dataclass
class StagedWarmup:
    """NUTS warmup in short rounds, each starting from the last round's metric and positions.

    numpyro adapts the step size and mass matrix inside one ``MCMC.warmup`` call whose length
    is fixed in advance, and reports nothing until it ends. Here each round is a fresh
    ``MCMC`` from ``build(num_warmup, inverse_mass_matrix, step_size)`` started where the
    previous round left off, with the chains' adapted mass matrices pooled into the next
    round's starting metric (``wf.warmup_until`` carries positions only and restarts from the
    identity metric; see docs/summer4-workarounds.md, S7). After every round a progress line
    is printed and the state is saved to ``folder``, so a killed job resumes at the next
    round. Rounds stop when ``check`` (default ``wf.WarmupRule()``) passes on the second half
    of a round's draws, or after the last entry of ``rounds``.
    """

    build: Callable[[int, np.ndarray | None, float], Any]
    folder: Path
    rounds: tuple[int, ...] = (100, 100, 200, 200, 400)
    check: Callable[[Mapping[str, Any]], Any] = field(default_factory=lambda: wf.WarmupRule())
    max_tree_depth: int = 8
    extra_fields: tuple[str, ...] = ("num_steps", "accept_prob", "diverging")
    rows: list[dict[str, Any]] = field(default_factory=list)
    mcmc: Any = None
    done: bool = False

    def __post_init__(self) -> None:
        self.folder = Path(self.folder)
        self.folder.mkdir(parents=True, exist_ok=True)

    def run(
        self,
        init_params: Mapping[str, Any] | None,
        metric: np.ndarray | None,
        *,
        step_size: float = 1.0,
        seed: int = 0,
        deadline: float | None = None,
    ) -> Any:
        """Warm up (or resume warming up); return the warmed-up ``MCMC``.

        Raises :class:`DeadlineReached` instead of starting a round that would not finish
        before ``deadline`` (``time.time()``), judged from the previous round's pace.
        """
        from jax import random

        saved = self._load()
        if saved is not None:
            init_params, metric, step_size, state = saved
            if self.done:
                self.mcmc = self.build(self.rows[-1]["num_warmup"], metric, step_size)
                self.mcmc.post_warmup_state = state
                return self.mcmc
        key = random.PRNGKey(seed + 7919 * len(self.rows))
        while not self.done:
            k = len(self.rows)
            n = self.rounds[min(k, len(self.rounds) - 1)]
            if deadline is not None and self.rows:
                pace = float(self.rows[-1]["seconds_per_iteration"])
                if time.time() + 1.25 * pace * n > deadline:
                    raise DeadlineReached(f"warmup round {k + 1} would overrun the deadline")
            mcmc = self.build(n, metric, step_size)
            start = time.perf_counter()
            key, sub = random.split(key)
            mcmc.warmup(
                sub,
                init_params=init_params,
                collect_warmup=True,
                extra_fields=self.extra_fields,
            )
            # Parallel chains return futures: wait for them, or the round looks instant.
            jax.block_until_ready(mcmc.post_warmup_state)
            seconds = time.perf_counter() - start
            row = self._row(mcmc, k + 1, n, seconds)
            state = mcmc.post_warmup_state
            failed = tuple(self.check(row))
            row["failed"] = ", ".join(failed) if failed else None
            self.done = not failed or k + 1 >= len(self.rounds)
            row["done"] = self.done
            self.rows.append(row)
            init_params = jax.tree.map(np.asarray, state.z)
            adapt = getattr(state, "adapt_state", None)
            if adapt is not None and hasattr(adapt, "inverse_mass_matrix"):  # HMC / NUTS
                metric = _pooled_metric(adapt.inverse_mass_matrix)
                step_size = float(np.median(np.asarray(adapt.step_size)))
            self._save(init_params, metric, step_size, state)
            print(f"[warmup] {json.dumps(row)}", flush=True)
            self.mcmc = mcmc
        return self.mcmc

    def _row(self, mcmc: Any, k: int, n: int, seconds: float) -> dict[str, Any]:
        """Second-half statistics of a round; fields a kernel does not record are ``None``."""
        draws = mcmc.get_samples(group_by_chain=True)
        if not isinstance(draws, Mapping):  # a kernel on a flat potential: one array
            draws = {"z": draws}
        half = {name: np.asarray(v)[:, n // 2 :] for name, v in draws.items()}
        extra = {
            name: np.asarray(v)[:, n // 2 :]
            for name, v in mcmc.get_extra_fields(group_by_chain=True).items()
        }
        adapt = getattr(mcmc.post_warmup_state, "adapt_state", None)
        steps = (
            np.atleast_1d(np.asarray(adapt.step_size))
            if adapt is not None and hasattr(adapt, "inverse_mass_matrix")
            else None
        )
        mean = lambda name: float(np.mean(extra[name])) if name in extra else None  # noqa: E731
        return {
            "round": k,
            "num_warmup": n,
            "seconds": round(seconds, 1),
            "seconds_per_iteration": round(seconds / n, 3),
            "leapfrog_mean": mean("num_steps"),
            "rhat_max": float(diagnostics(half)["rhat"].max()),
            "step_size_min": None if steps is None else float(steps.min()),
            "step_size_max": None if steps is None else float(steps.max()),
            "step_size_ratio": None if steps is None else float(steps.max() / steps.min()),
            "accept_mean": mean("accept_prob"),
            "target_accept_prob": (
                None if steps is None else float(getattr(mcmc.sampler, "_target_accept_prob", 0.8))
            ),
            "divergence_frac": mean("diverging"),
            "treedepth_frac": (
                float(np.mean(extra["num_steps"] >= 2**self.max_tree_depth - 1))
                if "num_steps" in extra
                else None
            ),
        }

    def _save(self, z: Any, metric: np.ndarray | None, step_size: float, state: Any) -> None:
        blob = {
            "z": z,
            "metric": metric,
            "step_size": step_size,
            "state": jax.device_get(state),
            "rows": self.rows,
            "done": self.done,
        }
        _atomic(self.folder / "warmup.pkl", lambda p: p.write_bytes(pickle.dumps(blob)))
        _atomic(self.folder / "warmup.csv", lambda p: self.progress.to_csv(p))

    def _load(self) -> tuple[Any, np.ndarray | None, float, Any] | None:
        path = self.folder / "warmup.pkl"
        if not path.exists():
            return None
        blob = pickle.loads(path.read_bytes())
        self.rows, self.done = list(blob["rows"]), bool(blob["done"])
        state = jax.tree.map(jnp.asarray, blob["state"])
        return blob["z"], blob["metric"], float(blob["step_size"]), state

    @property
    def progress(self) -> pd.DataFrame:
        return pd.DataFrame(self.rows).set_index("round")


def _pooled_metric(inverse_mass_matrix: Any) -> np.ndarray:
    """One dense metric from numpyro's per-chain adapted ``inverse_mass_matrix``."""
    blocks = inverse_mass_matrix
    if isinstance(blocks, Mapping):
        if len(blocks) != 1:
            raise ValueError("Expected one dense mass-matrix block over every site.")
        blocks = next(iter(blocks.values()))
    arr = np.asarray(blocks)
    return arr.mean(axis=0) if arr.ndim == 3 else arr


# --------------------------------------------------------------------------------------------
# Cheap alternative: Laplace approximation corrected by Pareto-smoothed importance sampling
# --------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class LaplacePSIS:
    """Importance draws from a Student-t fitted at a mode, with PSIS weights and ``k_hat``.

    ``k_hat < 0.7`` means the weighted draws estimate posterior expectations reliably
    (Vehtari et al., *Pareto smoothed importance sampling*); above it they do not.
    """

    sites: tuple[str, ...]
    z: np.ndarray  # (n, d) unconstrained proposal draws
    log_weights: np.ndarray  # (n,) Pareto-smoothed, normalised
    k_hat: float
    ess: float
    seconds: float
    failed: int = 0
    log_ratio: np.ndarray | None = None  # raw log p - log q, before smoothing

    def resample(self, n: int, *, seed: int = 0) -> dict[str, np.ndarray]:
        """``n`` unconstrained draws resampled by weight (with replacement), site dict."""
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(self.z), size=n, p=np.exp(self.log_weights))
        return {s: self.z[idx, i] for i, s in enumerate(self.sites)}


def hessian_fd(grad_fn: Callable[[Any], Any], z: np.ndarray, h: float = 1e-3) -> np.ndarray:
    """Central-difference Hessian from a gradient function of a flat vector."""
    d = len(z)
    cols = []
    for i in range(d):
        e = np.zeros(d)
        e[i] = h
        cols.append((np.asarray(grad_fn(z + e)) - np.asarray(grad_fn(z - e))) / (2 * h))
    hess = np.stack(cols, axis=1)
    return 0.5 * (hess + hess.T)


def laplace_covariance(
    bm: Any,
    mode: Mapping[str, Any],
    *,
    h: float = 1e-3,
    min_var: float = 1e-4,
    max_var: float = 25.0,
) -> tuple[tuple[str, ...], np.ndarray]:
    """Inverse negative Hessian of the log density at ``mode`` (unconstrained), sites sorted.

    Eigenvalues are clipped to ``[min_var, max_var]`` so a flat direction (a parameter piled
    against a prior bound) or an indefinite one still gives a usable metric. Pass a ``bm``
    with a tight solver (``rtol = 1e-8``): the Hessian is a central difference of the
    gradient, and at the calibration tolerance solver noise would swamp it.
    """
    sites = tuple(sorted(bm.prior_names()))

    def flat_density(vec: Any) -> Any:
        return bm.log_density({s: vec[i] for i, s in enumerate(sites)})

    grad = jax.jit(jax.grad(flat_density))
    z0 = np.asarray([float(mode[s]) for s in sites])
    vals, vecs = np.linalg.eigh(-hessian_fd(grad, z0, h))
    var = np.clip(1.0 / np.where(vals > 0, vals, 1.0 / max_var), min_var, max_var)
    return sites, (vecs * var) @ vecs.T


def laplace_psis(
    bm: Any,
    mode: Mapping[str, Any],
    *,
    n: int = 4000,
    df: float = 5.0,
    inflate: float = 1.2,
    seed: int = 0,
    batch_size: int = 16,
    hessian_bm: Any | None = None,
) -> LaplacePSIS:
    """Laplace at ``mode`` (unconstrained site dict), widened to a Student-t, then PSIS.

    The Hessian is a central difference of the reverse-mode gradient (``hessian_bm`` may use a
    tighter solver so solver noise does not swamp the difference). Forward-over-reverse would
    be exact but diffrax's checkpointed adjoint has no forward-mode rule.
    """
    from arviz_stats.base import array_stats

    start = time.perf_counter()
    sites = tuple(bm.prior_names())
    hb = bm if hessian_bm is None else hessian_bm

    def flat_density(vec: Any) -> Any:
        return hb.log_density({s: vec[i] for i, s in enumerate(sites)})

    grad = jax.jit(jax.grad(flat_density))
    z0 = np.asarray([float(mode[s]) for s in sites])
    neg_hess = -hessian_fd(grad, z0)
    vals, vecs = np.linalg.eigh(neg_hess)
    vals = np.maximum(vals, 1e-3 * vals.max())  # a flat or indefinite direction gets width
    cov = (vecs / vals) @ vecs.T * inflate**2
    chol = np.linalg.cholesky(cov)
    rng = np.random.default_rng(seed)
    d = len(sites)
    normal = rng.normal(size=(n, d))
    chi = rng.chisquare(df, size=(n, 1))
    draws = z0 + (normal @ chol.T) * np.sqrt(df / chi)
    # Student-t log density up to a constant (the constant cancels in normalised weights).
    maha = np.sum(np.linalg.solve(chol, (draws - z0).T) ** 2, axis=0)
    log_q = -0.5 * (df + d) * np.log1p(maha / df)
    score = jax.jit(
        lambda z: jax.lax.map(
            lambda v: bm.log_density({s: v[i] for i, s in enumerate(sites)}),
            z,
            batch_size=batch_size,
        )
    )
    log_p = np.asarray(score(jnp.asarray(draws)))
    ok = np.isfinite(log_p) & (log_p > -1e29)  # a failed solve has zero posterior density
    lw = np.full(n, -np.inf)
    ratio = log_p[ok] - log_q[ok]
    smoothed, k_hat = array_stats.psislw(ratio - np.max(ratio))
    lw[ok] = np.asarray(smoothed)
    lw = lw - np.logaddexp.reduce(lw[ok])
    ess = float(1.0 / np.sum(np.exp(2 * lw[ok])))
    return LaplacePSIS(
        sites,
        draws,
        lw,
        float(k_hat),
        ess,
        time.perf_counter() - start,
        int(np.sum(~ok)),
        log_p - log_q,
    )


# --------------------------------------------------------------------------------------------
# The convenience: every stage, checkpointed in one folder
# --------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class PipelineConfig:
    """Settings for :func:`calibrate`. The defaults are the NUTS arm ``nuts_td8``.

    ``kernel`` is ``"nuts"``, ``"sa"`` or ``"ess"`` (:func:`make_mcmc`); ``num_chains`` is
    chains (walkers for ``"ess"``); ``thinning`` keeps every n-th draw; ``cpus`` is what the
    job was given, for CPU-hour accounting.
    """

    kernel: str = "nuts"
    design: int = 256
    starts: int = 24
    opt_steps: int = 300
    workers: int = 8
    num_chains: int = 8
    chain_method: str = "parallel"
    warmup_rounds: tuple[int, ...] = (100, 100, 200, 200, 400)
    chunk: int = 200
    thinning: int = 1
    jitter: float = 0.05
    criteria: StopCriteria = field(default_factory=StopCriteria)
    target_accept_prob: float = 0.8
    max_tree_depth: int = 8
    shear: bool = True
    progress_bar: bool = True
    cpus: int = 8
    tight_metric: bool = True
    ess_max_steps: int = 20
    ess_max_iter: int = 50


@dataclass
class Calibration:
    """What :func:`calibrate` leaves behind: every stage's object, and where it was saved."""

    folder: Path
    optima: wf.Candidates
    modes: list[Optimum]
    warmup: pd.DataFrame
    checkpoint: Checkpoint
    seconds: dict[str, float]

    @property
    def idata(self) -> Any:
        return self.checkpoint.idata

    @property
    def converged(self) -> bool:
        rows = self.checkpoint.rows
        return bool(rows) and rows[-1].get("decision") == "diagnostics"

    def summary(self) -> dict[str, Any]:
        """Diagnostics for ``details.yaml``."""
        diag = diagnostics(self.checkpoint.samples)
        last = self.checkpoint.rows[-1]
        return {
            "converged": self.converged,
            "decision": last.get("decision"),
            "draws_per_chain": int(last["draws_per_chain"]),
            "rhat_max": float(diag["rhat"].max()),
            "ess_bulk_min": float(diag["ess_bulk"].min()),
            "ess_tail_min": float(diag["ess_tail"].min()),
            "divergence_frac": last.get("divergence_frac"),
            "modes": [
                {"log_density": m.log_density, "members": len(m.members)} for m in self.modes
            ],
            "warmup_rounds": int(len(self.warmup)),
            "num_warmup": int(self.warmup["num_warmup"].iloc[-1]),
            "seconds": {k: round(v) for k, v in self.seconds.items()},
            "per_parameter": {
                site: {k: round(float(v), 4) for k, v in row.items()}
                for site, row in diag.iterrows()
            },
        }


def initial_positions(
    optima: wf.Candidates, config: PipelineConfig, metric: np.ndarray | None, *, seed: int = 0
) -> dict[str, np.ndarray]:
    """Unconstrained starting points, one per chain (or walker).

    NUTS and SA chains start at the optima within :data:`SEED_WINDOW` of the best, jittered by
    ``config.jitter``. Ensemble walkers must span the posterior, so they are drawn from the
    Laplace approximation (``metric``, sites sorted) around the best optimum.
    """
    if config.kernel != "ess" or metric is None:
        return seeds(optima).init_params(config.num_chains, jitter=config.jitter, seed=seed)
    best = optima.best(1)
    sites = sorted(best.sites)
    mode = np.asarray([float(np.asarray(best.z[s])[0]) for s in sites])
    draws = np.random.default_rng(seed).multivariate_normal(mode, metric, size=config.num_chains)
    return {s: draws[:, i] for i, s in enumerate(sites)}


def calibrate(
    bm: Any,
    folder: Path,
    config: PipelineConfig = PipelineConfig(),
    *,
    fallback: Any = None,
    metric_bm: Any = None,
    seed: int = 0,
    deadline: float | None = None,
) -> Calibration:
    """Design → multi-start L-BFGS → mode check → Laplace metric → checkpointed MCMC.

    The MCMC stage is ``config.kernel`` (NUTS by default; SA and the ensemble slice sampler
    for the gradient-free reference arms), warmed up by :class:`StagedWarmup` and sampled by
    :func:`sample_chunks` with a :class:`Checkpoint` as the stop rule.

    Every stage writes to ``folder`` and is skipped on a rerun when its output exists, so a
    killed job restarts where it stopped. ``deadline`` (``time.time()``) is when the job must
    stop: warmup raises :class:`DeadlineReached` rather than start a round it cannot finish,
    and sampling stops (decision ``"deadline"``) rather than start a chunk it cannot finish.
    ``fallback`` is ``bm`` solved with diffrax's
    forward-mode adjoint, used by :func:`lbfgs` where a reverse-mode gradient is NaN;
    ``metric_bm`` is ``bm`` with a tight solver, for the Laplace metric NUTS starts from.
    Without the checkpointing it stands for::

        optima = lbfgs(bm, design(bm, 256).best(24), fallback=bm_forward)
        modes = distinct_optima(optima)
        _, cov = laplace_covariance(metric_bm, mode)  # mode: optima.best(1) as a dict
        shear = ridge_shear(bm, mode, cov)
        build = lambda n, metric, step: make_mcmc(
            bm, PipelineConfig(), n, shear=shear, inverse_mass_matrix=metric, step_size=step)
        init = shear.forward(seeds(optima).init_params(8, jitter=0.05))
        mcmc = StagedWarmup(build, folder / "mcmc").run(
            init, sheared_covariance(shear, mode, cov))
        stop = Checkpoint(mcmc, folder / "mcmc", StopCriteria(),
                          postprocess=lambda w: bm.constrain(shear.inverse(w)))
        sample_chunks(mcmc, stop, extra_fields=KERNEL_FIELDS["nuts"])
    """
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    seconds: dict[str, float] = {}
    times_path = folder / "stage_seconds.json"
    if times_path.exists():
        seconds = json.loads(times_path.read_text())

    def record(stage: str, start: float) -> None:
        seconds[stage] = seconds.get(stage, 0.0) + time.perf_counter() - start
        times_path.write_text(json.dumps(seconds))
        print(f"[calibrate] {stage} done: {json.dumps(seconds)}", flush=True)

    optima_path = folder / "optima.npz"
    if optima_path.exists():
        optima = wf.Candidates.load(optima_path)
    else:
        start = time.perf_counter()
        points = design(bm, config.design, seed=seed)
        record("design", start)
        start = time.perf_counter()
        optima = lbfgs(
            bm,
            points.best(config.starts),
            fallback=fallback,
            workers=min(config.workers, config.cpus),
            maxiter=config.opt_steps,
        )
        optima.save(optima_path)
        record("lbfgs", start)
    modes = distinct_optima(optima)

    best = optima.best(1)
    mode = {k: float(np.asarray(v)[0]) for k, v in best.z.items()}
    metric_path = folder / "metric.npy"
    metric: np.ndarray | None = None
    if metric_path.exists():
        metric = np.load(metric_path)
    elif metric_bm is not None:
        start = time.perf_counter()
        _, metric = laplace_covariance(metric_bm, mode)
        np.save(metric_path, metric)
        record("laplace", start)
    if metric is None and config.kernel == "ess":
        raise ValueError(
            "The ensemble arm draws its walkers from the Laplace metric: pass metric_bm."
        )

    raw_metric = metric
    shear: RidgeShear | None = None
    postprocess: Callable[[dict[str, Any]], dict[str, Any]] | None = None
    if config.kernel == "nuts" and config.shear and metric is not None:
        shear = ridge_shear(bm, mode, metric)
        metric = sheared_covariance(shear, mode, metric)
        (folder / "shear.json").write_text(json.dumps(asdict(shear)))

        def postprocess(chunk: dict[str, Any]) -> dict[str, Any]:
            assert shear is not None
            params = bm.constrain(shear.inverse({k: jnp.asarray(v) for k, v in chunk.items()}))
            return {k: np.asarray(v) for k, v in params.items()}

    chain_dir = folder / "mcmc"
    use_shear = shear if config.kernel == "nuts" else None

    def build(num_warmup: int, inverse_mass_matrix: Any, step_size: float) -> Any:
        return make_mcmc(
            bm,
            config,
            num_warmup,
            inverse_mass_matrix=inverse_mass_matrix if config.kernel == "nuts" else None,
            step_size=step_size,
            shear=use_shear,
        )

    staged = StagedWarmup(
        build,
        chain_dir,
        rounds=config.warmup_rounds,
        max_tree_depth=config.max_tree_depth,
        extra_fields=KERNEL_FIELDS[config.kernel],
    )
    probe = build(config.warmup_rounds[0], metric, 1.0)
    checkpoint = Checkpoint.resume(chain_dir, probe, config.criteria, postprocess)
    if checkpoint is not None:
        mcmc = probe
        staged._load()
        checkpoint.cpus = config.cpus
    else:
        start = time.perf_counter()
        init = initial_positions(optima, config, raw_metric, seed=seed)
        if use_shear is not None:
            init = {k: np.asarray(v) for k, v in use_shear.forward(init).items()}
        mcmc = staged.run(init, metric, seed=seed, deadline=deadline)
        record("warmup", start)
        checkpoint = Checkpoint(
            mcmc, chain_dir, config.criteria, postprocess=postprocess, cpus=config.cpus
        )
    checkpoint.deadline = deadline
    start = time.perf_counter()
    last = checkpoint.rows[-1].get("decision") if checkpoint.rows else None
    if last != "diagnostics":
        if last is not None:  # a spent budget: continue only if the new criteria allow it
            checkpoint.rows[-1]["decision"] = None
        sample_chunks(
            mcmc,
            checkpoint,
            seed=seed + 1 + len(checkpoint.rows),
            extra_fields=KERNEL_FIELDS[config.kernel],
        )
    record("sample", start)
    return Calibration(folder, optima, modes, staged.progress, checkpoint, seconds)


# --------------------------------------------------------------------------------------------
# Straightening the transmission ridge
# --------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class RidgeShear:
    """A unit-Jacobian change of NUTS coordinates that straightens the transmission ridge.

    The force of infection scales with ``raw_transmission_rate * (pop_2020 / N) **
    infection_pop_scale``, so the data pin ``log(raw) + s * log(K)`` (``s`` the scale, ``K`` of
    order the ratio of 2020 to age-group population) and leave a thin ridge along which
    ``raw`` falls as ``s`` rises. In the logit coordinates NUTS uses that ridge is curved
    (width about 0.02 against a length of about 1), which forces tiny steps. This replaces
    ``z_raw`` by ``w = z_raw - T(raw_c * K ** -s)`` with ``T`` the logit map onto
    ``raw``'s prior range: ``w`` is 0 on the ridge ``raw = raw_c K^-s``. It is a shear
    (``z_raw`` shifted by a function of ``z_s`` only), so its Jacobian determinant is 1 and
    the density in the new coordinates is the same function: the posterior is unchanged.
    """

    raw: str
    scale: str
    lo: float
    hi: float
    raw_c: float
    log_k: float

    def _t(self, s: Any) -> Any:
        ridge = self.raw_c * jnp.exp(-self.log_k * s)
        u = jnp.clip((ridge - self.lo) / (self.hi - self.lo), 1e-12, 1 - 1e-12)
        return jnp.log(u) - jnp.log1p(-u)

    def forward(self, z: Mapping[str, Any]) -> dict[str, Any]:
        """NUTS coordinates from ``bm`` unconstrained coordinates."""
        s = jax.nn.sigmoid(z[self.scale])
        return {**z, self.raw: z[self.raw] - self._t(s)}

    def inverse(self, w: Mapping[str, Any]) -> dict[str, Any]:
        """``bm`` unconstrained coordinates from NUTS coordinates."""
        s = jax.nn.sigmoid(w[self.scale])
        return {**w, self.raw: w[self.raw] + self._t(s)}

    def potential(self, bm: Any) -> Callable[[Mapping[str, Any]], Any]:
        """``bm``'s potential in the sheared coordinates."""
        return lambda w: bm.potential_fn(self.inverse(w))


def ridge_shear(
    bm: Any,
    mode: Mapping[str, Any],
    covariance: np.ndarray,
    *,
    raw: str = "raw_transmission_rate",
    scale: str = "infection_pop_scale",
) -> RidgeShear:
    """Fit the ridge through ``mode`` from the Laplace ``covariance`` (sites sorted).

    The local slope ``d log(raw) / d s`` is the regression slope of ``z_raw`` on ``z_s``
    under the Laplace approximation, carried to natural scale through the logit maps.
    """
    sites = sorted(bm.prior_names())
    i, j = sites.index(raw), sites.index(scale)
    slope_z = covariance[i, j] / covariance[j, j]  # d z_raw / d z_s
    prior = {p.name: p for p in bm.site_priors()}[raw]
    lo, hi = float(prior.lo), float(prior.hi)
    params = bm.constrain({k: jnp.asarray(v) for k, v in mode.items()})
    r, s = float(params[raw]), float(params[scale])
    dz_raw_dlog_raw = r * (hi - lo) / ((r - lo) * (hi - r))
    dz_s_ds = 1.0 / (s * (1.0 - s))
    dlog_raw_ds = slope_z * dz_s_ds / dz_raw_dlog_raw
    log_k = -dlog_raw_ds
    return RidgeShear(raw, scale, lo, hi, raw_c=r * np.exp(log_k * s), log_k=log_k)


def sheared_covariance(shear: RidgeShear, mode: Mapping[str, Any], cov: np.ndarray) -> np.ndarray:
    """``cov`` (sites sorted) carried into the sheared coordinates by the local Jacobian."""
    sites = sorted(mode)

    def fwd(vec: Any) -> Any:
        w = shear.forward({s: vec[k] for k, s in enumerate(sites)})
        return jnp.stack([w[s] for s in sites])

    jac = np.asarray(jax.jacfwd(fwd)(jnp.asarray([float(mode[s]) for s in sites])))
    return jac @ cov @ jac.T
