"""NeuTra: NUTS in coordinates warped by a normalising flow fitted to the posterior with SVI.

Neural transport (Hoffman et al. 2019, *NeuTra-lizing bad geometry in Hamiltonian Monte Carlo
using neural transport*) fits a flow ``T`` so that ``T(u)``, ``u ~ N(0, I)``, approximates the
posterior, then runs NUTS on ``u``. Where the flow is good the warped posterior is close to a
standard normal and NUTS needs short trajectories; where it is poor NUTS still samples the exact
posterior (the change of variables carries its Jacobian), only more slowly. So correctness rests
on NUTS diagnostics, not on the flow.

The pieces, each usable alone:

1. :class:`WhitenedCoordinates`: an affine (optionally sheared, see
   :class:`kiribati_tb.pipeline.RidgeShear`) map from a whitened vector ``x`` to ``bm``'s
   unconstrained sites, fitted from a Laplace approximation. Its Jacobian is constant, so the
   posterior on ``x`` is the same up to a constant, and a flow initialised near the identity
   already starts at the Laplace approximation.
2. :func:`flat_model`: a numpyro model with one vector site whose log density is any potential;
   numpyro's autoguides (``AutoIAFNormal``, ``AutoBNAFNormal``) and ``NeuTraReparam`` take it.
3. :class:`FlowFit`: the caller's numpyro ``SVI`` run in checkpointed chunks, with the ELBO and
   an importance-sampling check of the flow (:func:`flow_check`) after each chunk.
4. :func:`neutra_nuts_factory`: ``NUTS(NeuTraReparam(guide, params).reparam(model))`` in the
   shape :class:`kiribati_tb.pipeline.StagedWarmup` builds kernels from, and
   :class:`StepCounter`, which records leapfrog steps per draw so cost can be counted in
   gradients.

:func:`run_neutra` strings them together; its docstring lists the lines it stands for.
"""

from __future__ import annotations

import json
import pickle
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd

from kiribati_tb.pipeline import RidgeShear, _atomic, sheared_covariance

LATENT = "x"  # the flat model's one site


# --------------------------------------------------------------------------------------------
# Coordinates
# --------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class WhitenedCoordinates:
    """``x -> z``: ``w = loc + scale_tril @ x``, then ``z = shear.inverse(w)`` (if a shear).

    ``z`` are ``bm``'s unconstrained sites (``sites``, sorted). Both steps have a constant
    Jacobian determinant (the shear's is 1), so ``-bm.potential_fn(z(x))`` is the posterior's
    log density on ``x`` up to a constant. Every method takes a leading batch axis.
    """

    sites: tuple[str, ...]
    loc: np.ndarray
    scale_tril: np.ndarray
    shear: RidgeShear | None = None

    @property
    def dim(self) -> int:
        return len(self.sites)

    def to_unconstrained(self, x: Any) -> dict[str, Any]:
        """Site dict of unconstrained ``z`` from ``x`` (last axis)."""
        w = jnp.asarray(self.loc) + jnp.asarray(x) @ jnp.asarray(self.scale_tril).T
        z = {s: w[..., i] for i, s in enumerate(self.sites)}
        return z if self.shear is None else self.shear.inverse(z)

    def from_unconstrained(self, z: Mapping[str, Any]) -> Any:
        """``x`` (last axis) from a site dict of unconstrained ``z``."""
        if self.shear is not None:
            z = self.shear.forward({s: jnp.asarray(z[s]) for s in self.sites})
        w = jnp.stack([jnp.asarray(z[s]) for s in self.sites], axis=-1) - jnp.asarray(self.loc)
        flat = w.reshape(-1, self.dim).T
        x = jax.scipy.linalg.solve_triangular(jnp.asarray(self.scale_tril), flat, lower=True)
        return x.T.reshape(w.shape)

    def potential(self, bm: Any) -> Callable[[Any], Any]:
        """``bm``'s potential as a function of one (unbatched) ``x``."""
        return self.pull_back(bm.potential_fn)

    def pull_back(self, potential_fn: Callable[[Any], Any]) -> Callable[[Any], Any]:
        """Any potential of an unconstrained site dict, as a function of ``x``."""
        return lambda x: potential_fn(self.to_unconstrained(x))

    @classmethod
    def from_laplace(
        cls,
        mode: Mapping[str, float],
        covariance: np.ndarray,
        *,
        shear: RidgeShear | None = None,
        inflate: float = 1.0,
    ) -> WhitenedCoordinates:
        """Centred at ``mode`` (unconstrained site dict) and whitened by ``covariance``.

        ``covariance`` is over the sorted sites in ``bm``'s unconstrained coordinates (as
        :func:`kiribati_tb.pipeline.laplace_covariance` returns it); with ``shear`` it is
        carried into the sheared coordinates by the local Jacobian.
        """
        sites = tuple(sorted(mode))
        cov = np.asarray(covariance, dtype=float)
        if shear is not None:
            cov = sheared_covariance(shear, mode, cov)
            centre = shear.forward({s: jnp.asarray(float(mode[s])) for s in sites})
        else:
            centre = {s: float(mode[s]) for s in sites}
        loc = np.asarray([float(centre[s]) for s in sites])
        chol = np.linalg.cholesky(0.5 * (cov + cov.T) * inflate**2)
        return cls(sites, loc, chol, shear)

    def to_json(self) -> dict[str, Any]:
        return {
            "sites": list(self.sites),
            "loc": self.loc.tolist(),
            "scale_tril": np.asarray(self.scale_tril).tolist(),
            "shear": None if self.shear is None else self.shear.__dict__,
        }

    @classmethod
    def from_json(cls, blob: Mapping[str, Any]) -> WhitenedCoordinates:
        shear = None if blob["shear"] is None else RidgeShear(**blob["shear"])
        return cls(
            tuple(blob["sites"]),
            np.asarray(blob["loc"], dtype=float),
            np.asarray(blob["scale_tril"], dtype=float),
            shear,
        )


# WORKAROUND(summer4): the flow and warped NUTS run on this one-site model rather than on
# ``bm.numpyro_model()``, which has no hook for a joint change of coordinates (the Laplace
# whitening the flow starts from). See docs/summer4-workarounds.md, W11.
def flat_model(potential: Callable[[Any], Any], dim: int, *, site: str = LATENT) -> Callable:
    """A numpyro model of one real vector site ``site`` with log density ``-potential``.

    The site has an improper flat prior, so the model's density is exactly ``-potential``.
    Autoguides and ``NeuTraReparam`` work on it like on any model. ``potential`` may itself be
    built from a numpyro model (as ``BayesianModel.potential_fn`` is): its sites are blocked.
    """
    import numpyro
    import numpyro.distributions as dist
    from numpyro.distributions import constraints

    def model() -> None:
        x = numpyro.sample(site, dist.ImproperUniform(constraints.real_vector, (), (dim,)))
        # A potential built from another numpyro model (``BayesianModel.potential_fn``) runs
        # that model's sample sites; block them from this model's trace.
        with numpyro.handlers.block():
            log_density = -potential(x)
        numpyro.factor("log_density", log_density)

    return model


# --------------------------------------------------------------------------------------------
# Fitting the flow
# --------------------------------------------------------------------------------------------


def psis(log_ratio: np.ndarray) -> tuple[np.ndarray, float]:
    """Pareto-smoothed, normalised log importance weights and ``k_hat`` for ``log p - log q``.

    ``arviz_stats``' ``psislw`` takes the *negative* log weights (it is built for PSIS-LOO,
    whose weights are ``1 / p(y_i | theta)``, and negates its input), so the ratio is negated
    here.
    """
    from arviz_stats.base import array_stats

    r = np.asarray(log_ratio, dtype=float)
    smoothed, k_hat = array_stats.psislw(-(r - np.max(r)))
    smoothed = np.asarray(smoothed)
    return smoothed - np.logaddexp.reduce(smoothed), float(k_hat)


def flow_check(
    guide: Any,
    params: Any,
    potential: Callable[[Any], Any],
    *,
    n: int = 128,
    seed: int = 0,
    batch_size: int = 8,
) -> dict[str, float]:
    """How good the fitted flow ``q`` is as an approximation of the posterior ``p``.

    ``n`` draws from ``q`` are scored under ``p``; returns the ELBO estimate (mean of ``log p -
    log q`` over solves that succeeded, up to ``p``'s normalising constant), the Pareto
    ``k_hat`` of the importance ratios (below 0.7 the flow is a usable importance proposal;
    below 0.5 a good one), the normalised importance-sampling ESS, and the number of failed
    solves. Forward evaluations only: about ``n`` x one likelihood evaluation.
    """
    q = guide.get_posterior(params)
    key = jax.random.PRNGKey(seed)
    u = q.sample(key, (n,))
    log_q = np.asarray(q.log_prob(u))
    x = jax.vmap(lambda v: guide._unpack_latent(v)[LATENT])(u)
    score = jax.jit(lambda xs: jax.lax.map(potential, xs, batch_size=batch_size))
    log_p = -np.asarray(score(x))
    ok = np.isfinite(log_p) & (log_p > -1e29)
    ratio = log_p[ok] - log_q[ok]
    if ok.sum() < 10:
        return {"elbo": float("nan"), "k_hat": float("nan"), "is_ess_frac": 0.0, "failed": n}
    smoothed, k_hat = psis(ratio)
    w = np.exp(smoothed)
    return {
        "elbo": float(np.mean(ratio)),
        "k_hat": float(k_hat),
        "is_ess_frac": float(1.0 / np.sum(w**2) / n),
        "failed": int(np.sum(~ok)),
    }


def _notfinite_total(svi_state: Any) -> int:
    """Updates skipped by ``optax.apply_if_finite`` so far (0 without it)."""
    for leaf in jax.tree.leaves(
        svi_state.optim_state, is_leaf=lambda s: hasattr(s, "total_notfinite")
    ):
        if hasattr(leaf, "total_notfinite"):
            return int(leaf.total_notfinite)
    return 0


@dataclass
class FlowFit:
    """The caller's numpyro ``SVI`` run in checkpointed chunks, with a flow check per chunk.

    After every ``chunk`` steps it appends the losses, runs ``check`` (default
    :func:`flow_check` with ``potential``) every ``check_every`` chunks, prints one progress
    line and writes ``svi_state.pkl``, ``losses.npy`` and ``progress.csv`` to ``folder``.
    :meth:`run` resumes from those files, so a killed job restarts where it stopped.

    Use ``optax.apply_if_finite`` in the optimiser: a reverse-mode gradient of the adaptive
    solve is occasionally NaN (docs/summer4-workarounds.md, S5), and that update must be
    skipped rather than poison the optimiser state. The count of skipped updates is reported.
    """

    svi: Any
    folder: Path
    potential: Callable[[Any], Any]
    chunk: int = 50
    check_every: int = 2
    check_draws: int = 128
    losses: list[float] = field(default_factory=list)
    rows: list[dict[str, Any]] = field(default_factory=list)
    state: Any = None
    seconds: float = 0.0
    cpu_seconds: float = 0.0

    def __post_init__(self) -> None:
        self.folder = Path(self.folder)
        self.folder.mkdir(parents=True, exist_ok=True)

    def run(
        self,
        num_steps: int,
        *,
        seed: int = 0,
        init_params: Mapping[str, Any] | None = None,
        stop: Callable[[Mapping[str, Any]], bool] | None = None,
    ) -> FlowFit:
        """Train to ``num_steps`` steps in total (resuming), or until ``stop(row)`` is true.

        ``init_params`` (guide parameters, e.g. a flow fitted to another configuration) seed
        a fresh run; they are ignored when resuming.
        """
        self._load()
        # SVI.init also sets up the SVI object (its constrain_fn), so it runs on resume too.
        fresh = self.svi.init(jax.random.PRNGKey(seed), init_params=init_params)
        if self.state is None:
            self.state = fresh
        if not self.rows and self.check_draws > 0:  # where training starts from
            check = flow_check(
                self.svi.guide, self.params, self.potential, n=self.check_draws, seed=seed
            )
            self.rows.append({"step": 0, **check})
            print(f"[flow] {json.dumps(self.rows[-1])}", flush=True)
        update = jax.jit(self.svi.update)
        while len(self.losses) < num_steps:
            start, cpu = time.perf_counter(), time.process_time()
            n = min(self.chunk, num_steps - len(self.losses))
            chunk_losses = []
            for _ in range(n):
                self.state, loss = update(self.state)
                chunk_losses.append(float(loss))
            seconds, cpu_s = time.perf_counter() - start, time.process_time() - cpu
            self.seconds += seconds
            self.cpu_seconds += cpu_s
            self.losses.extend(chunk_losses)
            arr = np.asarray(chunk_losses)
            good = arr[np.isfinite(arr) & (arr < 1e20)]
            row: dict[str, Any] = {
                "step": len(self.losses),
                "loss_median": float(np.median(good)) if len(good) else float("nan"),
                "loss_mean": float(np.mean(good)) if len(good) else float("nan"),
                "failed_steps": int(len(arr) - len(good)),
                "skipped_updates": _notfinite_total(self.state),
                "seconds_per_step": seconds / n,
                "seconds": self.seconds,
                "cpu_seconds": self.cpu_seconds,
            }
            if (row["step"] // self.chunk) % self.check_every == 0 or len(self.losses) >= num_steps:
                check_start = time.perf_counter()
                row |= flow_check(
                    self.svi.guide,
                    self.svi.get_params(self.state),
                    self.potential,
                    n=self.check_draws,
                    seed=len(self.rows),
                )
                row["check_seconds"] = time.perf_counter() - check_start
            self.rows.append(row)
            self._save()
            print(f"[flow] {json.dumps(row)}", flush=True)
            if stop is not None and stop(row):
                break
        return self

    @property
    def params(self) -> Any:
        return self.svi.get_params(self.state)

    @property
    def progress(self) -> pd.DataFrame:
        return pd.DataFrame(self.rows).set_index("step")

    def _save(self) -> None:
        state = jax.device_get(self.state)
        blob = {
            "state": state,
            "rows": self.rows,
            "seconds": self.seconds,
            "cpu_seconds": self.cpu_seconds,
        }
        _atomic(self.folder / "svi_state.pkl", lambda p: p.write_bytes(pickle.dumps(blob)))
        _atomic(self.folder / "losses.npy", lambda p: _save_npy(p, np.asarray(self.losses)))
        _atomic(self.folder / "progress.csv", lambda p: self.progress.to_csv(p))
        params = jax.device_get(self.params)
        _atomic(self.folder / "params.pkl", lambda p: p.write_bytes(pickle.dumps(params)))

    def _load(self) -> None:
        path = self.folder / "svi_state.pkl"
        if self.state is not None or not path.exists():
            return
        blob = pickle.loads(path.read_bytes())
        self.state = jax.tree.map(jnp.asarray, blob["state"])
        self.rows = list(blob["rows"])
        self.seconds = float(blob["seconds"])
        self.cpu_seconds = float(blob.get("cpu_seconds", 0.0))
        self.losses = list(np.load(self.folder / "losses.npy"))


def _save_npy(path: Path, array: Any) -> None:
    with Path(path).open("wb") as f:  # np.save on a path would append ".npy"
        np.save(f, np.asarray(array))


def load_params(folder: Path) -> Any:
    """Guide parameters a :class:`FlowFit` wrote to ``folder``."""
    params = pickle.loads((Path(folder) / "params.pkl").read_bytes())
    return jax.tree.map(jnp.asarray, params)


# --------------------------------------------------------------------------------------------
# NUTS in the warped coordinates
# --------------------------------------------------------------------------------------------


def neutra_nuts_factory(
    model: Callable,
    guide: Any = None,
    params: Any = None,
    *,
    num_chains: int,
    chunk: int,
    chain_method: str = "parallel",
    dense_mass: bool = False,
    target_accept_prob: float = 0.8,
    max_tree_depth: int = 8,
    progress_bar: bool = False,
    extra_fields: tuple[str, ...] = ("num_steps", "diverging"),
) -> Callable[[int, np.ndarray | None, float], Any]:
    """``(num_warmup, inverse_mass_matrix, step_size) -> MCMC`` with NUTS on the warped model.

    The kernel is ``NUTS(NeuTraReparam(guide, params).reparam(model))``: it samples the
    flow's base variable (site ``f"{guide.prefix}_shared_latent"``), and every draw also
    carries ``model``'s sites as deterministic values. The returned callable has the signature
    :class:`kiribati_tb.pipeline.StagedWarmup` builds kernels with. The ``MCMC`` always records
    ``extra_fields`` (see :class:`FieldsMCMC`).

    With ``guide=None`` the kernel is plain ``NUTS(model)``: the same sampler without the flow,
    which on :func:`flat_model` over :class:`WhitenedCoordinates` with ``dense_mass=True`` is
    the pipeline's Laplace-started dense-metric NUTS (HMC is affine invariant when its metric
    moves with the coordinates), so it is the matched baseline.
    """
    from numpyro.infer import NUTS
    from numpyro.infer.reparam import NeuTraReparam

    warped = model if guide is None else NeuTraReparam(guide, params).reparam(model)

    def make(num_warmup: int, inverse_mass_matrix: Any = None, step_size: float = 1.0) -> Any:
        if inverse_mass_matrix is not None:
            inverse_mass_matrix = np.asarray(inverse_mass_matrix)
            if inverse_mass_matrix.ndim == (2 if not dense_mass else 3):
                # Per-chain adapted metrics (StagedWarmup pools only dense ones): average them.
                inverse_mass_matrix = inverse_mass_matrix.mean(axis=0)
        kernel = NUTS(
            warped,
            dense_mass=dense_mass,
            target_accept_prob=target_accept_prob,
            max_tree_depth=max_tree_depth,
            inverse_mass_matrix=(
                None if inverse_mass_matrix is None else jnp.asarray(inverse_mass_matrix)
            ),
            step_size=step_size,
        )
        return FieldsMCMC(
            kernel,
            num_warmup=int(num_warmup),
            num_samples=int(chunk),
            num_chains=int(num_chains),
            chain_method=chain_method,
            progress_bar=progress_bar,
            always_collect=extra_fields,
        )

    return make


def _fields_mcmc_class() -> type:
    from numpyro.infer import MCMC

    # WORKAROUND(summer4): wf.sample_until records only "diverging"; see W10.
    class _FieldsMCMC(MCMC):  # type: ignore[misc]
        """numpyro ``MCMC`` whose ``run`` always records ``always_collect`` extra fields.

        WORKAROUND(summer4): ``wf.sample_until`` calls ``mcmc.run(extra_fields=("diverging",))``
        and nothing else, so tree sizes (``num_steps``), needed to count gradient evaluations,
        are lost. See docs/summer4-workarounds.md, W10.
        """

        def __init__(self, *args: Any, always_collect: tuple[str, ...] = (), **kwargs: Any):
            super().__init__(*args, **kwargs)
            self.always_collect = tuple(always_collect)

        def run(self, *args: Any, extra_fields: tuple[str, ...] = (), **kwargs: Any) -> None:
            fields = tuple(dict.fromkeys(tuple(extra_fields) + self.always_collect))
            super().run(*args, extra_fields=fields, **kwargs)

    return _FieldsMCMC


def FieldsMCMC(*args: Any, always_collect: tuple[str, ...] = (), **kwargs: Any) -> Any:
    """A numpyro ``MCMC`` that records ``always_collect`` extra fields on every ``run``."""
    return _fields_mcmc_class()(*args, always_collect=always_collect, **kwargs)


def warped_postprocess(
    bm: Any, coords: WhitenedCoordinates, *, site: str = LATENT
) -> Callable[[dict[str, Any]], dict[str, Any]]:
    """Map a chunk of warped NUTS draws (with the deterministic ``site``) to parameters."""

    def post(chunk: dict[str, Any]) -> dict[str, Any]:
        params = bm.constrain(coords.to_unconstrained(jnp.asarray(chunk[site])))
        return {k: np.asarray(v) for k, v in params.items()}

    return post


@dataclass
class StepCounter:
    """A stop callable around another (e.g. a ``Checkpoint``) that records tree sizes.

    After every chunk it appends the chunk's ``num_steps`` (leapfrog steps, i.e. gradient
    evaluations, per draw and chain) to ``folder/num_steps.npy`` and passes the row on.
    """

    inner: Callable[[Mapping[str, Any]], Any]
    mcmc: Any
    folder: Path
    num_steps: np.ndarray | None = None

    def __post_init__(self) -> None:
        self.folder = Path(self.folder)
        path = self.folder / "num_steps.npy"
        if self.num_steps is None and path.exists():
            self.num_steps = np.load(path)

    def __call__(self, row: Mapping[str, Any]) -> Any:
        steps = np.asarray(self.mcmc.get_extra_fields(group_by_chain=True)["num_steps"])
        self.num_steps = (
            steps if self.num_steps is None else np.concatenate([self.num_steps, steps], axis=1)
        )
        _atomic(self.folder / "num_steps.npy", lambda p: _save_npy(p, self.num_steps))
        return self.inner(row)


def efficiency(
    samples: Mapping[str, np.ndarray], num_steps: np.ndarray, cpu_seconds: float | None = None
) -> dict[str, float]:
    """Gradient evaluations per effective draw (worst parameter), and ESS per CPU-hour."""
    from kiribati_tb.pipeline import diagnostics

    diag = diagnostics(samples)
    ess = float(min(diag["ess_bulk"].min(), diag["ess_tail"].min()))
    grads = float(np.sum(num_steps))
    out = {
        "ess_min": ess,
        "gradients": grads,
        "gradients_per_ess": grads / ess,
        "leapfrog_mean": float(np.mean(num_steps)),
        "tree_depth_mean": float(np.mean(np.log2(np.asarray(num_steps) + 1))),
        "rhat_max": float(diag["rhat"].max()),
    }
    if cpu_seconds:
        out["ess_per_cpu_hour"] = ess / (cpu_seconds / 3600.0)
    return out


# --------------------------------------------------------------------------------------------
# The convenience: flow, then warped NUTS, checkpointed in one folder
# --------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class NeutraConfig:
    """Settings for :func:`run_neutra` (see docs/neutra.md for how they were chosen)."""

    flow: str = "iaf"  # "iaf" (AutoIAFNormal), "bnaf" (AutoBNAFNormal), "none" (no flow)
    num_flows: int = 2
    hidden: tuple[int, ...] = (38, 38)
    learning_rate: float = 1e-3
    particles: int = 4
    svi_steps: int = 2000
    svi_chunk: int = 50
    check_every: int = 2
    check_draws: int = 128
    num_chains: int = 4
    chain_method: str = "parallel"
    warmup_rounds: tuple[int, ...] = (100, 100, 200)
    chunk: int = 100
    target_accept_prob: float = 0.8
    max_tree_depth: int = 8
    dense_mass: bool = False
    progress_bar: bool = True


def make_guide(model: Callable, config: NeutraConfig, *, dim: int | None = None) -> Any:
    """The autoguide ``config`` names, over ``model``.

    With ``dim``, the guide's set-up trace starts at ``x = 0`` (the Laplace centre in
    :class:`WhitenedCoordinates`) rather than numpyro's random ``init_to_uniform`` point, where
    a configuration far from the base case can fail its finite-gradient check.
    """
    from numpyro.infer import init_to_uniform, init_to_value
    from numpyro.infer.autoguide import AutoBNAFNormal, AutoIAFNormal

    init = init_to_uniform if dim is None else init_to_value(values={LATENT: jnp.zeros(dim)})
    if config.flow == "iaf":
        return AutoIAFNormal(
            model, num_flows=config.num_flows, hidden_dims=list(config.hidden), init_loc_fn=init
        )
    if config.flow == "bnaf":
        return AutoBNAFNormal(
            model,
            num_flows=config.num_flows,
            hidden_factors=list(config.hidden),
            init_loc_fn=init,
        )
    raise ValueError(f"Unknown flow {config.flow!r}; expected 'iaf' or 'bnaf'.")


def make_svi(model: Callable, guide: Any, config: NeutraConfig) -> Any:
    """``SVI(model, guide, apply_if_finite(clip + adam), Trace_ELBO(particles))``."""
    import optax
    from numpyro.infer import SVI, Trace_ELBO

    optimiser = optax.apply_if_finite(
        optax.chain(optax.clip_by_global_norm(10.0), optax.adam(config.learning_rate)),
        max_consecutive_errors=50,
    )
    return SVI(model, guide, optimiser, Trace_ELBO(num_particles=config.particles))


@dataclass
class NeutraRun:
    """What :func:`run_neutra` leaves behind."""

    folder: Path
    coords: WhitenedCoordinates
    flow: FlowFit | None
    warmup: pd.DataFrame
    checkpoint: Any
    counter: StepCounter
    seconds: dict[str, float]

    @property
    def idata(self) -> Any:
        return self.checkpoint.idata

    def summary(self) -> dict[str, Any]:
        rows = self.checkpoint.rows
        last = rows[-1] if rows else {}
        eff = (
            efficiency(self.checkpoint.samples, self.counter.num_steps)
            if self.counter.num_steps is not None
            else {}
        )
        flow_rows = [] if self.flow is None else [r for r in self.flow.rows if "k_hat" in r]
        return {
            "decision": last.get("decision"),
            "draws_per_chain": last.get("draws_per_chain"),
            "divergence_frac": last.get("divergence_frac"),
            "flow": (
                None
                if self.flow is None
                else {
                    "steps": len(self.flow.losses),
                    "seconds": self.flow.seconds,
                    "cpu_seconds": self.flow.cpu_seconds,
                    "last_check": flow_rows[-1] if flow_rows else None,
                }
            ),
            "warmup_rounds": int(len(self.warmup)),
            "sampling": eff,
            "seconds": self.seconds,
        }


def fit_flow(
    bm: Any,
    coords: WhitenedCoordinates,
    folder: Path,
    config: NeutraConfig = NeutraConfig(),
    *,
    init_flow: Any = None,
    seed: int = 0,
) -> tuple[Callable, Any, FlowFit]:
    """The flat model on ``coords``, its guide, and the fitted (or resumed) :class:`FlowFit`.

    Writes to ``folder / "flow"``. With ``config.svi_steps = 0``, ``init_flow`` is used as
    is and checked once with :func:`flow_check`.
    """
    potential = coords.potential(bm)
    model = flat_model(potential, coords.dim)
    guide = make_guide(model, config, dim=coords.dim)
    svi = make_svi(model, guide, config)
    flow = FlowFit(
        svi,
        Path(folder) / "flow",
        potential,
        chunk=config.svi_chunk,
        check_every=config.check_every,
        check_draws=config.check_draws,
    )
    if config.svi_steps > 0:
        flow.run(config.svi_steps, seed=seed, init_params=init_flow)
        return model, guide, flow
    flow._load()
    fresh = svi.init(jax.random.PRNGKey(seed), init_params=init_flow)
    if flow.state is None:
        flow.state = fresh
        check = flow_check(guide, flow.params, potential, n=config.check_draws, seed=seed)
        flow.rows.append({"step": 0, **check})
        flow._save()
        print(f"[flow] {json.dumps(flow.rows[-1])}", flush=True)
    return model, guide, flow


def run_neutra(
    bm: Any,
    coords: WhitenedCoordinates,
    folder: Path,
    config: NeutraConfig = NeutraConfig(),
    *,
    init_flow: Any = None,
    criteria: Any = None,
    seed: int = 0,
) -> NeutraRun:
    """Fit a flow by SVI, then run checkpointed NUTS in its warped coordinates.

    Every stage writes to ``folder`` and resumes on a rerun. ``init_flow`` (guide parameters,
    e.g. :func:`load_params` of another run) warm-starts SVI; with ``config.svi_steps = 0`` it
    is used as is. Without the checkpointing and timing it stands for::

        model = flat_model(coords.potential(bm), coords.dim)
        guide = AutoIAFNormal(model, num_flows=2, hidden_dims=[38, 38])
        svi = SVI(model, guide, optax.apply_if_finite(optax.adam(1e-3), 50),
                  Trace_ELBO(num_particles=4))
        flow = FlowFit(svi, folder / "flow", coords.potential(bm)).run(2000)
        build = neutra_nuts_factory(model, guide, flow.params, num_chains=4, chunk=100)
        init = {f"{guide.prefix}_shared_latent": normal(size=(4, coords.dim))}
        mcmc = StagedWarmup(build, folder / "nuts", rounds=(100, 100, 200)).run(init, None)
        stop = Checkpoint(mcmc, folder / "nuts", StopCriteria(),
                          postprocess=warped_postprocess(bm, coords))
        wf.sample_until(mcmc, StepCounter(stop, mcmc, folder / "nuts"))
    """
    from summer4.epi.calibration import workflow as wf

    from kiribati_tb.pipeline import Checkpoint, StagedWarmup, StopCriteria

    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "coords.json").write_text(json.dumps(coords.to_json()))
    seconds: dict[str, float] = {}
    times_path = folder / "stage_seconds.json"
    if times_path.exists():
        seconds = json.loads(times_path.read_text())

    def record(stage: str, start: float) -> None:
        seconds[stage] = seconds.get(stage, 0.0) + time.perf_counter() - start
        times_path.write_text(json.dumps(seconds))
        print(f"[neutra] {stage} done: {json.dumps(seconds)}", flush=True)

    flow: FlowFit | None = None
    guide: Any = None
    params: Any = None
    if config.flow == "none":  # the baseline: NUTS on the whitened coordinates, no flow
        model = flat_model(coords.potential(bm), coords.dim)
    else:
        start = time.perf_counter()
        model, guide, flow = fit_flow(bm, coords, folder, config, init_flow=init_flow, seed=seed)
        params = flow.params
        record("flow", start)

    build = neutra_nuts_factory(
        model,
        guide,
        params,
        num_chains=config.num_chains,
        chunk=config.chunk,
        chain_method=config.chain_method,
        dense_mass=config.dense_mass,
        target_accept_prob=config.target_accept_prob,
        max_tree_depth=config.max_tree_depth,
        progress_bar=config.progress_bar,
    )
    nuts_dir = folder / "nuts"
    post = warped_postprocess(bm, coords)
    criteria = StopCriteria() if criteria is None else criteria
    staged = StagedWarmup(
        build, nuts_dir, rounds=config.warmup_rounds, max_tree_depth=config.max_tree_depth
    )
    probe = build(config.warmup_rounds[0], None, 1.0)
    checkpoint = Checkpoint.resume(nuts_dir, probe, criteria, post)
    if checkpoint is not None:
        mcmc = probe
        staged._load()
    else:
        start = time.perf_counter()
        rng = np.random.default_rng(seed)
        site = LATENT if guide is None else f"{guide.prefix}_shared_latent"
        init = {site: rng.normal(size=(config.num_chains, coords.dim))}
        mcmc = staged.run(init, None, seed=seed)
        record("warmup", start)
        checkpoint = Checkpoint(mcmc, nuts_dir, criteria, postprocess=post)
    counter = StepCounter(checkpoint, mcmc, nuts_dir)
    start = time.perf_counter()
    last = checkpoint.rows[-1].get("decision") if checkpoint.rows else None
    if last != "diagnostics":
        if last is not None:
            checkpoint.rows[-1]["decision"] = None
        wf.sample_until(mcmc, counter, seed=seed + 1 + len(checkpoint.rows))
    record("sample", start)
    run = NeutraRun(folder, coords, flow, staged.progress, checkpoint, counter, seconds)
    (folder / "summary.json").write_text(json.dumps(run.summary(), indent=2, default=float))
    return run


__all__ = [
    "FieldsMCMC",
    "FlowFit",
    "LATENT",
    "NeutraConfig",
    "NeutraRun",
    "StepCounter",
    "WhitenedCoordinates",
    "efficiency",
    "fit_flow",
    "flat_model",
    "flow_check",
    "load_params",
    "make_guide",
    "psis",
    "make_svi",
    "neutra_nuts_factory",
    "run_neutra",
    "warped_postprocess",
]
