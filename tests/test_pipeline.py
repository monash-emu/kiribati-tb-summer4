"""The fast calibration pipeline's pieces, on toy problems (the TB model is too slow here)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist
import pytest
from numpyro.infer import MCMC, NUTS

from summer4.epi.calibration import workflow as wf

from kiribati_tb.pipeline import (
    Checkpoint,
    RidgeShear,
    StopCriteria,
    diagnostics,
    distinct_optima,
    laplace_covariance,
    lbfgs,
    seeds,
)


def _candidates(points: np.ndarray, log_density: np.ndarray) -> wf.Candidates:
    z = {"a": points[:, 0], "b": points[:, 1]}
    return wf.Candidates(
        sites=("a", "b"),
        z=z,
        params=z,
        log_density=log_density,
        ok=np.ones(len(points), dtype=bool),
    )


def test_distinct_optima_groups_nearby_points_best_first() -> None:
    points = np.asarray([[0.0, 0.0], [0.1, 0.0], [3.0, 3.0], [0.0, 0.2], [3.1, 3.0]])
    ld = np.asarray([-10.0, -10.1, -40.0, -10.2, -40.5])
    optima = distinct_optima(_candidates(points, ld), tol=0.5)
    assert [o.members for o in optima] == [(0, 1, 3), (2, 4)]
    assert optima[0].log_density == -10.0
    assert optima[1].z == {"a": 3.0, "b": 3.0}


def test_seeds_keep_the_window_below_the_best() -> None:
    points = np.arange(8.0).reshape(4, 2)
    ld = np.asarray([-100.0, -10.0, -30.0, -34.0])
    kept = seeds(_candidates(points, ld), window=20.0)
    np.testing.assert_array_equal(kept.log_density, [-10.0, -30.0])


def test_diagnostics_on_independent_chains() -> None:
    rng = np.random.default_rng(0)
    frame = diagnostics({"x": rng.normal(size=(4, 1000))})
    assert frame.loc["x", "rhat"] < 1.01
    assert frame.loc["x", "ess_bulk"] > 2000
    assert frame.loc["x", "ess_tail"] > 1000


def _toy_model() -> None:
    numpyro.sample("x", dist.Normal(jnp.zeros(2), jnp.asarray([1.0, 3.0])).to_event(1))


def _make_mcmc(num_warmup: int) -> MCMC:
    return MCMC(
        NUTS(_toy_model),
        num_warmup=num_warmup,
        num_samples=50,
        num_chains=2,
        chain_method="vectorized",
        progress_bar=False,
    )


def test_checkpoint_saves_every_chunk_and_resumes(tmp_path: Path) -> None:
    folder = tmp_path / "run"
    mcmc = _make_mcmc(100)
    first = Checkpoint(mcmc, folder, StopCriteria(ess=1e9, max_samples=100))
    with pytest.warns(RuntimeWarning, match="max_samples"):
        wf.sample_until(mcmc, first, seed=0)
    assert first.progress["draws_per_chain"].tolist() == [50, 100]
    assert first.progress["decision"].tolist()[-1] == "max_samples"
    for name in ("idata.nc", "mcmc_state.pkl", "progress.csv", "checkpoint.json"):
        assert (folder / name).exists()
    saved_step = np.asarray(mcmc.last_state.adapt_state.step_size)

    # A new process: fresh MCMC from the factory, state and draws reloaded from disk.
    fresh = _make_mcmc(100)
    resumed = Checkpoint.resume(folder, fresh, StopCriteria(ess=1e9, max_samples=200))
    assert resumed is not None
    np.testing.assert_array_equal(fresh.post_warmup_state.adapt_state.step_size, saved_step)
    with pytest.warns(RuntimeWarning, match="max_samples"):
        wf.sample_until(fresh, resumed, seed=1)
    assert resumed.progress["draws_per_chain"].tolist() == [50, 100, 150, 200]
    np.testing.assert_array_equal(resumed.samples["x"][:, :100], first.samples["x"])
    # The resumed chains did not re-warm: the step size is the one adapted before the restart.
    np.testing.assert_array_equal(fresh.last_state.adapt_state.step_size, saved_step)
    assert resumed.seconds_before > 0.0


def test_resume_without_checkpoint_is_none(tmp_path: Path) -> None:
    assert Checkpoint.resume(tmp_path, _make_mcmc(10)) is None


class GaussianModel:
    """A stand-in for ``BayesianModel`` with a Gaussian log density in unconstrained space."""

    def __init__(self, mean: dict[str, float], precision: np.ndarray) -> None:
        self.sites = tuple(sorted(mean))
        self.mean = jnp.asarray([mean[s] for s in self.sites])
        self.precision = jnp.asarray(precision)

    def prior_names(self) -> tuple[str, ...]:
        return self.sites

    def log_density(self, z: dict[str, Any]) -> Any:
        x = jnp.stack([jnp.asarray(z[s]) for s in self.sites]) - self.mean
        return -0.5 * x @ self.precision @ x

    def potential_fn(self, z: dict[str, Any]) -> Any:
        return -self.log_density(z)

    def constrain(self, z: dict[str, Any]) -> dict[str, Any]:
        return dict(z)


def test_lbfgs_finds_the_mode_from_every_start() -> None:
    model = GaussianModel({"a": 1.0, "b": -2.0}, np.asarray([[2.0, 1.8], [1.8, 2.0]]))
    starts = wf.Candidates.from_z(
        model, {"a": np.asarray([5.0, -3.0]), "b": np.asarray([5.0, 0.0])}
    )
    optima = lbfgs(model, starts, workers=2, gtol=1e-8)
    np.testing.assert_allclose(optima.z["a"], [1.0, 1.0], atol=1e-4)
    np.testing.assert_allclose(optima.z["b"], [-2.0, -2.0], atol=1e-4)
    assert optima.history[-2].stage == "lbfgs"
    assert np.all(optima.ok)


def test_laplace_covariance_inverts_the_precision() -> None:
    precision = np.asarray([[4.0, 1.0], [1.0, 2.0]])
    model = GaussianModel({"b": 0.5, "a": -1.0}, precision)
    sites, cov = laplace_covariance(model, {"a": -1.0, "b": 0.5}, h=1e-3)
    assert sites == ("a", "b")
    np.testing.assert_allclose(cov, np.linalg.inv(precision), rtol=1e-6)


def test_ridge_shear_is_a_unit_jacobian_bijection() -> None:
    shear = RidgeShear("raw", "s", lo=1e-4, hi=3e-3, raw_c=1.5e-3, log_k=1.7)
    z = {"raw": jnp.asarray(-1.3), "s": jnp.asarray(0.9), "other": jnp.asarray(2.0)}
    back = shear.inverse(shear.forward(z))
    for k in z:
        np.testing.assert_allclose(back[k], z[k], rtol=1e-12)

    def fwd(v: Any) -> Any:
        w = shear.forward({"raw": v[0], "s": v[1], "other": v[2]})
        return jnp.stack([w["raw"], w["s"], w["other"]])

    jac = jax.jacfwd(fwd)(jnp.asarray([-1.3, 0.9, 2.0]))
    assert float(jnp.linalg.det(jac)) == pytest.approx(1.0)
    # On the ridge raw = raw_c K^-s the sheared coordinate is zero.
    s = 0.7
    ridge = 1.5e-3 * np.exp(-1.7 * s)
    u = (ridge - 1e-4) / (3e-3 - 1e-4)
    w = shear.forward(
        {"raw": jnp.asarray(np.log(u / (1 - u))), "s": jnp.asarray(np.log(s / (1 - s)))}
    )
    assert float(w["raw"]) == pytest.approx(0.0, abs=1e-9)
