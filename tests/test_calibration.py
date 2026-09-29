"""K4: priors, targets and likelihood reproduce the original estival calibration."""

from __future__ import annotations

import jax
import numpy as np
import pytest
import yaml
from scipy import stats

from summer4.epi.calibration import Uniform
from kiribati_tb.calibration import (
    SENSITIVITY_ANALYSES,
    bayesian_model,
    calibration_setup,
    read_notifications,
)
from kiribati_tb.params import read_parameter_sheet
from kiribati_tb.paths import GOLDEN
from parity import read_golden


def test_priors_are_the_parameter_sheet() -> None:
    sheet = read_parameter_sheet().priors
    setup = calibration_setup()
    got = {p.name: (p.lo, p.hi) for p in setup.priors if isinstance(p, Uniform)}
    expected = {
        r.parameter: (r.distri_param1, r.distri_param2) for r in sheet.itertuples(index=False)
    }
    assert got == expected
    assert len(setup.priors) == len(expected) == 18


def test_targets_follow_the_tolerance_rule() -> None:
    setup = calibration_setup()
    by_key = {t.key: t for t in setup.targets}
    assert len(by_key) == 10
    notif = by_key["notifications"]
    years = sorted(read_notifications())
    np.testing.assert_array_equal(notif.times, years)
    assert notif.likelihood.sd == pytest.approx(0.40 / 1.96 * np.mean(notif.values))
    pearl = by_key["pearl_posXreach_reachable_per100k"]
    assert pearl.likelihood.sd == pytest.approx(0.20 / 1.96 * 850.2)
    assert by_key["mixing_matrix_distance"].likelihood.sd.name == "mixing_dist_sd"


@pytest.mark.parametrize("analysis", SENSITIVITY_ANALYSES)
def test_sensitivity_analyses(analysis: str) -> None:
    setup = calibration_setup(analysis)
    names = {p.name for p in setup.priors}
    keys = {t.key: t for t in setup.targets}
    if analysis == "tpt_60":
        assert setup.params["tpt_completion_perc"] == 60.0
    elif analysis == "subclinical_50":
        assert keys["perc_prev_subclinicalXreach_reachable"].values.tolist() == [50.0]
    else:
        assert not setup.config.heterogeneous_mixing
        assert not names & {"bg_mixing", "a_spread", "pc_strength"}
        assert "mixing_matrix_distance" not in keys


def estival_log_likelihood(setup: object, frame: object, mixing_sd: float) -> float:
    """estival 0.6's NormalTarget: mean over each target's times of the normal logpdf."""
    total = 0.0
    for target in setup.targets:  # type: ignore[attr-defined]
        modelled = frame.loc[target.times, target.key].to_numpy()  # type: ignore[attr-defined]
        sd = target.likelihood.sd
        sd = mixing_sd if not isinstance(sd, float) else sd
        total += float(np.mean(stats.norm.logpdf(target.values, loc=modelled, scale=sd)))
    return total


@pytest.mark.slow
def test_log_likelihood_matches_estival_rule_on_golden_outputs() -> None:
    """The port's likelihood at the golden parameters equals estival's rule on the golden."""
    params = yaml.safe_load((GOLDEN / "hetero_baseline" / "params.yaml").read_text())["params"]
    setup = calibration_setup()
    bm = bayesian_model(setup, solver={"solver": "dopri5", "rtol": 1e-10, "atol": 1e-10})
    sites = {p.name: params.get(p.name, 12.5) for p in bm._sites}
    z = bm.unconstrain(sites)
    log_density = float(jax.jit(bm.log_density)(z))
    log_prior = 0.0
    for prior in bm._sites:
        dist = prior.to_numpyro()
        # numpyro's log_density is in unconstrained space: add the log-Jacobian back out.
        transform = __import__("numpyro").distributions.transforms.biject_to(dist.support)
        log_prior += float(dist.log_prob(sites[prior.name]))
        log_prior += float(transform.log_abs_det_jacobian(z[prior.name], sites[prior.name]))
    expected = estival_log_likelihood(setup, read_golden("hetero_baseline", "derived"), 12.5)
    assert log_density - log_prior == pytest.approx(expected, rel=1e-6)


@pytest.mark.slow
def test_gradient_is_finite_at_the_golden_parameters() -> None:
    params = yaml.safe_load((GOLDEN / "hetero_baseline" / "params.yaml").read_text())["params"]
    bm = bayesian_model(calibration_setup())
    z = bm.unconstrain({p.name: params.get(p.name, 12.5) for p in bm._sites})
    grad = jax.jit(jax.grad(bm.log_density))(z)
    assert all(np.isfinite(float(v)) for v in grad.values())


def test_fast_infectiousness_loss_narrows_the_gain_prior() -> None:
    """The original narrows ``infectiousness_gain_rate`` to U(2, 10) when loss rate > 3."""
    base = {p.name: p for p in calibration_setup().priors}
    narrowed = {
        p.name: p
        for p in calibration_setup(param_overrides={"infectiousness_loss_rate": 3.5}).priors
    }
    assert (base["infectiousness_gain_rate"].lo, base["infectiousness_gain_rate"].hi) == (0.5, 10.0)
    assert (narrowed["infectiousness_gain_rate"].lo, narrowed["infectiousness_gain_rate"].hi) == (
        2.0,
        10.0,
    )
