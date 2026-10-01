"""Bayesian calibration on numpyro through summer4's ``BayesianModel``.

Ports the calibration half of ``tbh/runner_tools.py`` (estival + pymc ``DEMetropolisZ``) and the
nevergrad MAP notebook. Priors come from the ``parameters.xlsx`` constant sheet; targets are the
original's ten, with the same tolerance rule; the mixing-matrix distance target has a
prior-distributed standard deviation.
"""

from __future__ import annotations

import json

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from functools import cache
from typing import Any

import numpy as np
import pandas as pd

from summer4 import SavePlan, Target, TargetSet
from summer4.epi.calibration import BayesianModel, NormalLikelihood, Uniform, priors_from_frame
from summer4.epi.calibration.priors import Prior

from kiribati_tb.model import END_TIME, SOLVER_KWARGS, START_TIME, ModelConfig, compile_model
from kiribati_tb.outputs import build_outputs, restrict_outputs
from kiribati_tb.params import read_parameter_sheet
from kiribati_tb.paths import DATA

MIXING_PRIORS: tuple[str, ...] = ("bg_mixing", "a_spread", "pc_strength")
SENSITIVITY_ANALYSES: tuple[str, ...] = ("tpt_60", "subclinical_50", "homogeneous_mixing")


@dataclass(frozen=True)
class TargetSpec:
    """One calibration target: an output name, observations by year, a tolerance."""

    output: str
    observations: Mapping[float, float]
    tol_pct: float = 20.0
    aggregate: str = "mean"

    def target(self) -> Target:
        years = np.asarray(sorted(self.observations), dtype=float)
        values = np.asarray([self.observations[y] for y in years], dtype=float)
        likelihood = NormalLikelihood.from_tolerance(values, self.tol_pct, aggregate=self.aggregate)
        return Target(key=self.output, times=years, values=values, likelihood=likelihood)


@cache
def read_notifications() -> dict[float, float]:
    """Historical notifications for the modelled population, before 2023."""
    frame = pd.read_excel(DATA / "notifications.xlsx")
    frame = frame[frame["year"] < 2023]
    return {float(y): float(v) for y, v in zip(frame["year"], frame["inputed_all_other"])}


def target_specs() -> list[TargetSpec]:
    """The original's Normal targets (``runner_tools.targets``, less the mixing distance)."""
    return [
        TargetSpec("pearl_posXreach_reachable_per100k", {2024.0: 850.2}),
        TargetSpec("cxr_posXreach_reachable_per100k", {2024.0: 595.9}),
        TargetSpec("tst_posXage_3_9Xreach_reachable_perc", {2024.0: 3.3}),
        TargetSpec("tst_posXage_10Xreach_reachable_perc", {2024.0: 9.3}),
        TargetSpec("tst_posXage_15+Xreach_reachable_perc", {2024.0: 28.23}),
        # Kerri's paper
        TargetSpec("tst_posXage_18+Xreach_reachable_perc", {2011.0: 38.0}),
        TargetSpec("perc_prev_subclinicalXreach_reachable", {2024.0: 81.6}),  # 115 / (115 + 26)
        TargetSpec("perc_prev_infectiousXreach_reachable", {2024.0: 69.9}),  # 100 / 143
        TargetSpec("notifications", read_notifications(), tol_pct=40.0),
    ]


def mixing_distance_target() -> Target:
    """Distance to the ``conmat`` matrix in 2025 ~ Normal(0, sd), ``sd ~ Uniform(5, 20)``."""
    return Target(
        key="mixing_matrix_distance",
        times=np.asarray([2025.0]),
        values=np.asarray([0.0]),
        likelihood=NormalLikelihood(sd=Uniform("mixing_dist_sd", 5.0, 20.0)),
    )


def calibration_priors(params: Mapping[str, float]) -> list[Prior]:
    """Priors from the constant sheet, with the original's ``infectiousness_gain_rate`` rule.

    When ``infectiousness_loss_rate > 3`` the original narrows ``infectiousness_gain_rate`` to
    Uniform(2, 10), since early runs showed no posterior mass below 2.5.
    """
    sheet = read_parameter_sheet()
    priors = list(
        priors_from_frame(
            sheet.priors, "parameter", "distribution", "distri_param1", "distri_param2"
        )
    )
    if params["infectiousness_loss_rate"] > 3.0:
        priors = [p for p in priors if p.name != "infectiousness_gain_rate"]
        priors.append(Uniform("infectiousness_gain_rate", 2.0, 10.0))
    return priors


# WORKAROUND(summer4): BayesianModel folds targets.plan() into the save plan before
# outputs.plan(), and Target.contribute requires every key to be a save key or carry a leaf
# quantity. The Kiribati targets are derived OutputSet names (ratios such as ``_per100k``), which
# are never save keys, so the stock TargetSet raises. The outputs already plan every leaf the
# targets read, and log_likelihood reads the targets from the evaluated outputs, so planning
# is skipped here. See docs/summer4-workarounds.md, W3.
@dataclass(frozen=True, slots=True)
class OutputTargets(TargetSet):
    """Targets scored on named ``OutputSet`` outputs rather than on save keys."""

    def plan(self, base: SavePlan) -> SavePlan:
        return base


@dataclass(frozen=True)
class CalibrationSetup:
    """Everything that defines one calibration: model structure, fixed values, priors, targets."""

    config: ModelConfig
    params: dict[str, float]
    priors: list[Prior]
    targets: list[Target] = field(default_factory=list)

    @property
    def target_set(self) -> TargetSet:
        return OutputTargets(targets=tuple(self.targets))


def calibration_setup(
    sensitivity_analysis: str | None = None,
    param_overrides: Mapping[str, float] | None = None,
    *,
    aggregate: str = "mean",
) -> CalibrationSetup:
    """The base-case calibration, or one of the original's sensitivity analyses.

    ``tpt_60`` lowers TPT completion to 60%; ``subclinical_50`` replaces the subclinical
    prevalence target with 50%; ``homogeneous_mixing`` drops the mixing priors and the distance
    target and uses one mixing pool. ``param_overrides`` replaces fixed values first, as the
    original's cluster grid does (``clinical_regression_rate``, ``infectiousness_loss_rate``,
    ``rel_sus_unreachable``); it can trigger the ``infectiousness_gain_rate`` narrowing rule.

    ``aggregate`` is how a target's log-densities over its years combine. The code that
    produced the published posterior (estival 0.6) takes the ``"mean"``; the paper's methods
    appendix (§9.2) writes a sum. Only notifications has more than one year, so ``"sum"``
    weights the notification term by its number of years.
    """
    params = dict(read_parameter_sheet().constants) | dict(param_overrides or {})
    config = ModelConfig()
    specs = target_specs()
    with_distance = True
    if sensitivity_analysis == "tpt_60":
        params["tpt_completion_perc"] = 60.0
    elif sensitivity_analysis == "subclinical_50":
        specs = [s for s in specs if s.output != "perc_prev_subclinicalXreach_reachable"]
        specs.append(TargetSpec("perc_prev_subclinicalXreach_reachable", {2024.0: 50.0}))
    elif sensitivity_analysis == "homogeneous_mixing":
        config = ModelConfig(heterogeneous_mixing=False)
        with_distance = False
    elif sensitivity_analysis is not None:
        raise ValueError(f"Unknown sensitivity analysis: {sensitivity_analysis}")
    priors = calibration_priors(params)
    if not config.heterogeneous_mixing:
        priors = [p for p in priors if p.name not in MIXING_PRIORS]
    targets = [replace(s, aggregate=aggregate).target() for s in specs]
    if with_distance:
        targets.append(mixing_distance_target())
    return CalibrationSetup(config, params, priors, targets)


# The yearly mixing matrix (``Lookup`` on ``floor(Time() - START_TIME)``) changes value at every
# integer year inside the window. Given to the step-size controller as ``jump_ts``, the solver
# ends a step on each one instead of stepping across it.
MIXING_JUMPS: np.ndarray = np.arange(START_TIME + 1.0, END_TIME)
# Hairer & Wanner's PI step-size controller for step sizes limited by stability (the
# ``diffrax.PIDController`` documentation's suggestion for stiff problems).
PI_COEFFICIENTS: dict[str, float] = {"pcoeff": 0.4, "icoeff": 0.3}


def calibration_solver(
    solver: Any = None,
    *,
    rtol: float = 1.4e-4,
    adjoint: Any = None,
    max_steps: int = 4096,
) -> dict[str, Any]:
    """``run`` keyword arguments for calibration: the caller's diffrax objects, tuned for speed.

    Defaults (see ``docs/gradient-performance.md``): ``diffrax.Bosh3()``, a PI controller at
    ``rtol = atol = 1.4e-4`` (summer2gen's default tolerance, which the original calibrated
    with) told about the yearly mixing jumps, and ``RecursiveCheckpointAdjoint(checkpoints=
    1024)``. The step size here is limited by stability, not tolerance (the fastest Jacobian
    eigenvalue is about -10 per year in the posterior), and Bosh3 covers a year with fewer
    vector-field evaluations than Dopri5 at that limit. Against a 1e-10 reference gradient at
    the MAP and 24 published-posterior draws this is more accurate than the original's Dopri5
    setting (median relative error 0.05% against 0.33%, worst 0.5% against 8.6%), and its
    reverse-mode gradient was finite at all 256 prior design points and all 25 draws, where
    Dopri5's was NaN at 43 and 1. 1,024 checkpoints cover every step of a
    plausible solve, so the reverse pass does not recompute the forward one. ``max_steps``
    caps the cost of a pathological proposal (summer4's default, 64 per year, is 11,840) and
    scores it as a failed solve.
    """
    import diffrax

    from summer4.solvers import Diffrax

    controller = diffrax.PIDController(
        rtol=rtol, atol=rtol, jump_ts=MIXING_JUMPS, **PI_COEFFICIENTS
    )
    backend = Diffrax(
        diffrax.Bosh3() if solver is None else solver,
        stepsize_controller=controller,
        adjoint=(
            diffrax.RecursiveCheckpointAdjoint(checkpoints=1024) if adjoint is None else adjoint
        ),
    )
    return {"solver": backend, "max_steps": max_steps}


CALIBRATION_SOLVER: dict[str, Any] = calibration_solver()

# The original's calibration solver: summer2gen's default Dopri5 at ``rtol = atol = 1.4e-4``,
# stepping across the mixing jumps. Its log density is within 0.011 nats of a 1e-10 solve; its
# gradient within 0.33% (median over 25 posterior draws; 8.6% at worst). Kept for comparison.
ORIGINAL_CALIBRATION_SOLVER: dict[str, Any] = {
    "solver": "dopri5",
    "rtol": 1.4e-4,
    "atol": 1.4e-4,
    "max_steps": 4096,
}


def bayesian_model(
    setup: CalibrationSetup,
    *,
    solver: Mapping[str, Any] | None = None,
    extra_outputs: Sequence[str] = (),
    t1: float = END_TIME,
) -> BayesianModel:
    """A summer4 ``BayesianModel`` for ``setup``, scoring only the target outputs.

    ``t1`` is where the solve stops. Calibration can stop at :data:`CALIBRATION_END`, the last
    target year (5% fewer solver steps); projections (``posterior_runs``) need the default.
    """
    outputs = build_outputs(setup.config)
    keys = [t.key for t in setup.targets] + list(extra_outputs)
    if any(float(np.max(t.times)) > t1 for t in setup.targets):
        raise ValueError(f"t1={t1} is before a target year.")
    run_kwargs = {"t0": START_TIME, "t1": t1, "dt": 1.0}
    run_kwargs |= dict(CALIBRATION_SOLVER if solver is None else solver)
    return BayesianModel(
        compile_model(setup.config),
        setup.params,
        priors=setup.priors,
        targets=setup.target_set,
        outputs=restrict_outputs(outputs, keys),
        run_kwargs=run_kwargs,
    )


# The last year any target is observed (the mixing-matrix distance, 2025).
CALIBRATION_END = 2025.0

# For the Laplace metric's finite-difference Hessian, where solver noise must be small.
TIGHT_SOLVER: dict[str, Any] = {"solver": "dopri5", "rtol": 1e-8, "atol": 1e-8, "max_steps": 16384}


def forward_mode_solver(solver: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """``solver`` with diffrax's forward-mode adjoint (default: :func:`calibration_solver`'s).

    ``solver`` is a named spec (``{"solver": "dopri5", "rtol": ..., "atol": ...}``); without one
    this is ``calibration_solver(adjoint=diffrax.ForwardMode())``.

    With the original's Dopri5 setting, reverse-mode gradients of the adaptive solve are NaN at
    17% of 256 prior design points (none of 96 published posterior draws, but 1 of another 25),
    apparently where a *rejected* trial step evaluated the vector field at an invalid state
    (``0 * nan`` in the backward pass). Forward mode discards the rejected step's tangent
    instead, so it stays finite; it costs about 1.5x a reverse-mode gradient here. The Bosh3
    calibration solver's reverse-mode gradient was finite at all of those points. See
    docs/summer4-workarounds.md, S5.
    """
    import diffrax

    from summer4.solvers import Diffrax

    if solver is None:
        return calibration_solver(adjoint=diffrax.ForwardMode())
    spec = dict(solver)
    controller = diffrax.PIDController(rtol=spec.pop("rtol"), atol=spec.pop("atol"))
    name = spec.pop("solver")
    solvers = {"dopri5": diffrax.Dopri5, "tsit5": diffrax.Tsit5, "heun": diffrax.Heun}
    backend = Diffrax(
        solvers[name](), stepsize_controller=controller, adjoint=diffrax.ForwardMode()
    )
    return {"solver": backend, **spec}


def prior_midpoints(setup: CalibrationSetup) -> dict[str, float]:
    """The centre of every calibrated site's uniform prior (``mixing_dist_sd`` included)."""
    sites = list(setup.priors)
    for target in setup.targets:
        sd = getattr(target.likelihood, "sd", None)
        if isinstance(sd, Prior):
            sites.append(sd)
    return {p.name: 0.5 * (p.lo + p.hi) for p in sites if isinstance(p, Uniform)}


def map_point(setup: CalibrationSetup) -> dict[str, float]:
    """The optax MAP from ``outputs/find_map/base/map.json`` if written, else prior midpoints."""
    from kiribati_tb.paths import REPO_ROOT

    path = REPO_ROOT / "outputs" / "find_map" / "base" / "map.json"
    mid = prior_midpoints(setup)
    if not path.exists():
        return mid
    fitted = json.loads(path.read_text())["params"]
    return {k: float(fitted.get(k, v)) for k, v in mid.items()}


def with_params(setup: CalibrationSetup, overrides: Mapping[str, float]) -> CalibrationSetup:
    """``setup`` with some fixed values replaced."""
    return replace(setup, params={**setup.params, **dict(overrides)})


__all__ = [
    "CALIBRATION_SOLVER",
    "MIXING_JUMPS",
    "ORIGINAL_CALIBRATION_SOLVER",
    "CalibrationSetup",
    "SENSITIVITY_ANALYSES",
    "SOLVER_KWARGS",
    "TargetSpec",
    "bayesian_model",
    "calibration_solver",
    "calibration_setup",
    "target_specs",
    "with_params",
]
