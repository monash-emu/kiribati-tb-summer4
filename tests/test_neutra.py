"""NeuTra pieces on toy posteriors (the TB model is too slow here)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from summer4.epi.calibration import workflow as wf

from kiribati_tb.neutra import (
    FieldsMCMC,
    FlowFit,
    NeutraConfig,
    StepCounter,
    WhitenedCoordinates,
    efficiency,
    flat_model,
    flow_check,
    load_params,
    make_guide,
    make_svi,
    neutra_nuts_factory,
    run_neutra,
    warped_postprocess,
)
from kiribati_tb.pipeline import Checkpoint, RidgeShear, StagedWarmup, StopCriteria

COV = np.asarray([[1.0, 0.9, 0.0], [0.9, 1.0, 0.0], [0.0, 0.0, 4.0]])
MEAN = np.asarray([0.5, -1.0, 2.0])


class ToyModel:
    """The two methods of ``BayesianModel`` NeuTra uses: a correlated Gaussian on 3 sites."""

    sites = ("a", "b", "c")

    def potential_fn(self, z: dict[str, Any]) -> Any:
        v = jnp.stack([z[s] for s in self.sites]) - MEAN
        return 0.5 * v @ jnp.linalg.solve(jnp.asarray(COV), v)

    def constrain(self, z: dict[str, Any]) -> dict[str, Any]:
        return {s: jnp.asarray(z[s]) for s in self.sites}


def _shear() -> RidgeShear:
    return RidgeShear("a", "b", lo=0.0, hi=1.0, raw_c=0.4, log_k=1.5)


@pytest.mark.parametrize("shear", [None, _shear()])
def test_whitened_coordinates_round_trip(shear: RidgeShear | None) -> None:
    mode = {"a": 0.3, "b": -0.2, "c": 1.0}
    coords = WhitenedCoordinates.from_laplace(mode, COV, shear=shear)
    z0 = coords.to_unconstrained(jnp.zeros(3))
    np.testing.assert_allclose([float(z0[s]) for s in "abc"], [0.3, -0.2, 1.0], atol=1e-12)
    x = jnp.asarray(np.random.default_rng(0).normal(size=(5, 3)))
    back = coords.from_unconstrained(coords.to_unconstrained(x))
    np.testing.assert_allclose(np.asarray(back), np.asarray(x), atol=1e-10)
    again = WhitenedCoordinates.from_json(json.loads(json.dumps(coords.to_json())))
    np.testing.assert_allclose(again.scale_tril, coords.scale_tril)
    assert again.shear == shear


@pytest.mark.parametrize("shear", [None, _shear()])
def test_the_map_has_a_constant_jacobian(shear: RidgeShear | None) -> None:
    """Differences of the potential in ``x`` equal differences in ``z``: no Jacobian term."""
    coords = WhitenedCoordinates.from_laplace({"a": 0.3, "b": -0.2, "c": 1.0}, COV, shear=shear)

    def flat(x: Any) -> Any:
        z = coords.to_unconstrained(x)
        return jnp.stack([z[s] for s in "abc"])

    xs = np.random.default_rng(1).normal(size=(4, 3))
    dets = [float(jnp.linalg.det(jax.jacfwd(flat)(jnp.asarray(x)))) for x in xs]
    np.testing.assert_allclose(dets, dets[0], rtol=1e-10)


def _fit(tmp_path: Path, steps: int) -> tuple[Any, Any, FlowFit, WhitenedCoordinates]:
    coords = WhitenedCoordinates.from_laplace({"a": 0.0, "b": 0.0, "c": 0.0}, np.eye(3))
    potential = coords.potential(ToyModel())
    model = flat_model(potential, coords.dim)
    config = NeutraConfig(hidden=(6, 6), learning_rate=1e-2, particles=4)
    guide = make_guide(model, config)
    fit = FlowFit(
        make_svi(model, guide, config), tmp_path / "flow", potential, chunk=50, check_draws=64
    )
    return model, guide, fit.run(steps), coords


def test_flow_fit_learns_and_resumes(tmp_path: Path) -> None:
    _, guide, fit, _ = _fit(tmp_path, 300)
    assert len(fit.losses) == 300
    rows = fit.progress
    losses = rows["loss_median"].dropna()
    assert losses.iloc[-1] < losses.iloc[0]
    checked = rows.dropna(subset=["k_hat"])
    assert checked.index.tolist() == [0, 100, 200, 300]
    # The ELBO is log Z - KL(q || p); for this Gaussian log Z is known, so KL is small.
    log_z = 0.5 * np.log(np.linalg.det(2 * np.pi * COV))
    assert log_z - 0.3 < checked["elbo"].iloc[-1] <= log_z + 0.1
    # a fresh FlowFit in the same folder resumes rather than restarting
    _, _, again, _ = _fit(tmp_path, 350)
    assert len(again.losses) == 350 and len(again.rows) == 8
    params = load_params(tmp_path / "flow")
    assert set(params) == set(again.params)


def test_flow_check_finds_a_poor_flow() -> None:
    """A near-standard-normal flow on a far wider, offset posterior has a large ``k_hat``."""
    coords = WhitenedCoordinates.from_laplace({"a": 3.0, "b": 3.0, "c": 3.0}, np.eye(3) * 0.01)
    potential = coords.potential(ToyModel())
    model = flat_model(potential, coords.dim)
    config = NeutraConfig(hidden=(6, 6))
    guide = make_guide(model, config)
    svi = make_svi(model, guide, config)
    params = svi.get_params(svi.init(jax.random.PRNGKey(0)))
    assert flow_check(guide, params, potential, n=128)["k_hat"] > 0.7


def test_warped_nuts_samples_the_posterior(tmp_path: Path) -> None:
    """Warmup rounds, checkpointed chunks and the step counter, in the warped space."""
    bm = ToyModel()
    model, guide, fit, coords = _fit(tmp_path, 300)
    build = neutra_nuts_factory(
        model, guide, fit.params, num_chains=2, chunk=200, chain_method="vectorized"
    )
    staged = StagedWarmup(build, tmp_path / "nuts", rounds=(150,))
    init = {"auto_shared_latent": np.random.default_rng(0).normal(size=(2, 3))}
    mcmc = staged.run(init, None)
    stop = Checkpoint(
        mcmc,
        tmp_path / "nuts",
        StopCriteria(rhat=1.05, ess=200.0, max_samples=600),
        postprocess=warped_postprocess(bm, coords),
    )
    counter = StepCounter(stop, mcmc, tmp_path / "nuts")
    wf.sample_until(mcmc, counter, warn=False)
    draws = np.stack([stop.samples[s].reshape(-1) for s in "abc"], axis=1)
    np.testing.assert_allclose(draws.mean(axis=0), MEAN, atol=0.35)
    np.testing.assert_allclose(np.cov(draws.T), COV, atol=0.6)
    steps = np.load(tmp_path / "nuts" / "num_steps.npy")
    assert steps.shape == stop.samples["a"].shape
    eff = efficiency(stop.samples, steps)
    assert eff["gradients"] == steps.sum() and eff["gradients_per_ess"] > 0


def test_fields_mcmc_always_records_its_fields() -> None:
    import numpyro
    import numpyro.distributions as dist
    from numpyro.infer import NUTS

    def model() -> None:
        numpyro.sample("x", dist.Normal(0.0, 1.0))

    mcmc = FieldsMCMC(
        NUTS(model),
        num_warmup=20,
        num_samples=10,
        progress_bar=False,
        always_collect=("num_steps",),
    )
    mcmc.run(jax.random.PRNGKey(0), extra_fields=("diverging",))
    assert set(mcmc.get_extra_fields()) == {"num_steps", "diverging"}


def test_run_neutra_end_to_end_and_reuse(tmp_path: Path) -> None:
    bm = ToyModel()
    coords = WhitenedCoordinates.from_laplace({"a": 0.0, "b": 0.0, "c": 0.0}, np.eye(3))
    config = NeutraConfig(
        hidden=(6, 6),
        learning_rate=1e-2,
        svi_steps=200,
        check_draws=64,
        num_chains=2,
        chain_method="vectorized",
        warmup_rounds=(150,),
        chunk=200,
        progress_bar=False,
    )
    criteria = StopCriteria(rhat=1.05, ess=200.0, max_samples=600)
    run = run_neutra(bm, coords, tmp_path / "first", config, criteria=criteria)
    summary = json.loads((tmp_path / "first" / "summary.json").read_text())
    assert summary["flow"]["steps"] == 200
    assert summary["sampling"]["gradients"] > 0
    assert run.idata.posterior["a"].shape[0] == 2
    # a flow reused unchanged (svi_steps=0) still drives a correct sampler
    reuse = NeutraConfig(**{**config.__dict__, "svi_steps": 0})
    again = run_neutra(
        bm,
        coords,
        tmp_path / "reuse",
        reuse,
        init_flow=load_params(tmp_path / "first" / "flow"),
        criteria=criteria,
    )
    assert again.flow.rows[0]["step"] == 0 and "k_hat" in again.flow.rows[0]
    draws = np.asarray(again.idata.posterior["c"]).ravel()
    assert abs(draws.mean() - MEAN[2]) < 0.4


def test_psis_k_hat_reads_the_right_tail() -> None:
    """Heavy right tail of the importance ratio: large k_hat; light tail: small."""
    from kiribati_tb.neutra import psis

    rng = np.random.default_rng(0)
    x = rng.normal(size=(4000, 3))
    heavy = 0.5 * np.sum(x**2, axis=1)  # weights exp(chi2_3 / 2): infinite variance
    lw, k_heavy = psis(heavy)
    assert k_heavy > 0.7
    np.testing.assert_allclose(np.exp(lw).sum(), 1.0)
    assert psis(-heavy)[1] < 0.5
    assert psis(0.1 * x[:, 0])[1] < 0.5


def test_run_neutra_without_a_flow_is_the_baseline(tmp_path: Path) -> None:
    coords = WhitenedCoordinates.from_laplace({"a": 0.0, "b": 0.0, "c": 0.0}, COV)
    config = NeutraConfig(
        flow="none",
        dense_mass=True,
        num_chains=2,
        chain_method="vectorized",
        warmup_rounds=(150,),
        chunk=200,
        progress_bar=False,
    )
    run = run_neutra(
        ToyModel(),
        coords,
        tmp_path,
        config,
        criteria=StopCriteria(rhat=1.05, ess=200.0, max_samples=600),
    )
    assert run.flow is None and not (tmp_path / "flow").exists()
    assert (tmp_path / "nuts" / "num_steps.npy").exists()
    draws = np.asarray(run.idata.posterior["b"]).ravel()
    assert abs(draws.mean() - MEAN[1]) < 0.4
