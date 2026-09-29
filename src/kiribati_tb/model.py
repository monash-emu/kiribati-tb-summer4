"""The Kiribati TB model in summer4.

Ports ``get_tb_model`` from ``tbh/model.py``. The original builds an unstratified summer2 model
and stratifies it by age and then reachability, attaching per-stratum flow adjustments. Here
the three properties form one map up front and every flow carries its adjustments as
``Multiply(value, where=selector)``.

Flow names follow the original's, so outputs and parity tests can refer to them. The one
exception is deaths: the original declares one flow per (cause, compartment, age); the port
declares one flow per cause and reads the age from the per-age death-rate table.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from summer4 import (
    REMAINDER,
    CompiledModel,
    Compartments,
    EntryFlow,
    FlowModel,
    InitialPopulation,
    Multiply,
    Param,
    Property,
    PropertyMap,
    Result,
    SavePlan,
    SaveRequest,
    Split,
    Time,
    TraitChain,
    TransitionFlow,
    exp,
    maximum,
    tanh,
)
from summer4.data import Data

from kiribati_tb.demography import AGE_GROUPS, Demography, load_demography
from kiribati_tb.params import read_parameter_sheet

COMPARTMENTS: tuple[str, ...] = (
    "mtb_naive",
    "incipient",
    "contained",
    "cleared",
    "subclin_lowinf",
    "clin_lowinf",
    "subclin_inf",
    "clin_inf",
    "treatment",
    "recovered",
)
LATENT_COMPS: tuple[str, ...] = ("incipient", "contained", "cleared")
ACTIVE_COMPS: tuple[str, ...] = ("subclin_lowinf", "clin_lowinf", "subclin_inf", "clin_inf")
REACH_STRATA: tuple[str, ...] = ("reachable", "unreachable")

STATE = Property("state", COMPARTMENTS)
AGE = Property("age", AGE_GROUPS)
REACH = Property("reach", REACH_STRATA)

SEED = 100.0
# summer2's ``get_sigmoidal_interpolation_function`` default curvature. summer4's ``sharpness``
# is the same parameter but defaults to 1 (near-linear), so every sigmoidal table here passes it.
SUMMER2_SIGMOID_CURVATURE = 16.0


def latency_age_bands() -> dict[str, tuple[str, ...]]:
    """Age bands sharing each latency parameter suffix (``age0``: under 5, ``age5``: 5–14)."""
    bands: dict[str, list[str]] = {"age0": [], "age5": [], "age15": []}
    for age in AGE_GROUPS:
        key = "age0" if int(age) < 5 else "age5" if int(age) < 15 else "age15"
        bands[key].append(age)
    return {key: tuple(value) for key, value in bands.items()}


def by_latency_band(stem: str) -> tuple[Multiply, ...]:
    """Age adjustments ``Param(f"{stem}_rate_{band}")`` for the three latency age bands."""
    return tuple(
        Multiply(Param(f"{stem}_rate_{band}"), where=AGE.isin(ages))
        for band, ages in latency_age_bands().items()
    )


def pmap() -> PropertyMap:
    """State × age × reachability: 160 compartments."""
    return PropertyMap.from_property(STATE).stratify(AGE).stratify(REACH)


def compartment_labels(model_map: PropertyMap) -> tuple[str, ...]:
    """summer2 compartment names (``{state}Xage_{age}Xreachability_{reach}``) in map order."""
    rows = model_map.to_dicts()
    return tuple(f"{r['state']}Xage_{r['age']}Xreachability_{r['reach']}" for r in rows)


def passive_detection_rate() -> object:
    """``recent_detection_rate`` times the tanh scale-up from ``past_frac`` to 1."""
    past = Param("passive_detection_past_frac")
    scale = (
        tanh(Param("passive_detection_shape") * (Time() - Param("passive_detection_inflection")))
        / 2.0
        + 0.5
    )
    return Param("recent_detection_rate") * (scale * (1.0 - past) + past)


def death_rate(demog: Demography) -> object:
    """Per-age background death rate: one sigmoidal table over ``AGE``."""
    table = demog.death_rate_table(AGE)
    return table.interp("sigmoidal", sharpness=SUMMER2_SIGMOID_CURVATURE)


def births_entry_rate(demog: Demography) -> object:
    """Yearly population increments, sigmoidally interpolated (people per year)."""
    return Data(demog.entry_times, demog.entry_values).interp(
        "sigmoidal", sharpness=SUMMER2_SIGMOID_CURVATURE
    )


def treatment_success() -> object:
    """Treatment success proportion: linear in time through the ``tx_success_pct`` knots."""
    series = read_parameter_sheet().time_variant["tx_success_pct"]
    return Data(series.index.to_numpy(dtype=float), series.to_numpy(dtype=float) / 100.0).interp(
        "linear"
    )


@dataclass(frozen=True)
class TreatmentOutcomes:
    """Per-age negative treatment outcome rates (``get_neg_tx_outcome_funcs``).

    Natural death on treatment is ``1 - exp(-tx_duration * mu_age)``. Deaths caused by TB on
    treatment take the requested share of unsuccessful outcomes left after natural deaths,
    floored at zero; relapse takes the rest.
    """

    relapse: object
    death: object


def treatment_outcomes(background: object, success: object) -> TreatmentOutcomes:
    """Relapse and on-treatment TB death rates from background mortality and success."""
    duration = Param("tx_duration")
    natural = 1.0 - exp(-duration * background)
    requested = (1.0 - success) * Param("pct_neg_tx_death") / 100.0
    tb_death = maximum(requested - natural, 0.0)
    relapse = 1.0 - success - tb_death - natural
    return TreatmentOutcomes(relapse=relapse / duration, death=tb_death / duration)


def add_natural_history(model: FlowModel) -> None:
    """Latency, progression, clinical and infectiousness transitions, self-recovery."""
    s = STATE
    model.add_flow(
        TransitionFlow(
            "containment",
            s["incipient"],
            s["contained"],
            1.0,
            adjust=by_latency_band("containment"),
        )
    )
    model.add_flow(
        TransitionFlow("clearance", s["contained"], s["cleared"], Param("clearance_rate"))
    )
    model.add_flow(
        TransitionFlow("breakdown", s["contained"], s["incipient"], Param("breakdown_rate"))
    )
    prop_inf = Param("progression_prop_infectious")
    for name, dest, share in (
        ("progression_lowinf", "subclin_lowinf", 1.0 - prop_inf),
        ("progression_inf", "subclin_inf", prop_inf),
    ):
        model.add_flow(
            TransitionFlow(
                name, s["incipient"], s[dest], share, adjust=by_latency_band("progression")
            )
        )
    for inf in ("inf", "lowinf"):
        model.add_flow(
            TransitionFlow(
                f"clinical_progression_{inf}",
                s[f"subclin_{inf}"],
                s[f"clin_{inf}"],
                Param("clinical_progression_rate"),
            )
        )
        model.add_flow(
            TransitionFlow(
                f"clinical_regression_{inf}",
                s[f"clin_{inf}"],
                s[f"subclin_{inf}"],
                Param("clinical_regression_rate"),
            )
        )
    # The original's flow names carry the typo "infectiousnnes"; kept so names match.
    for clin in ("clin", "subclin"):
        model.add_flow(
            TransitionFlow(
                f"infectiousnnes_gain_{clin}",
                s[f"{clin}_lowinf"],
                s[f"{clin}_inf"],
                Param("infectiousness_gain_rate"),
            )
        )
        model.add_flow(
            TransitionFlow(
                f"infectiousnnes_loss_{clin}",
                s[f"{clin}_inf"],
                s[f"{clin}_lowinf"],
                Param("infectiousness_loss_rate"),
            )
        )
    for inf in ("inf", "lowinf"):
        model.add_flow(
            TransitionFlow(
                f"self_recovery_{inf}",
                s[f"subclin_{inf}"],
                s["recovered"],
                Param("self_recovery_rate"),
            )
        )


def add_detection_and_treatment(
    model: FlowModel, outcomes: TreatmentOutcomes, success: object
) -> None:
    """Passive detection from each active state, then treatment recovery and relapse."""
    detection = passive_detection_rate()
    unreachable = Multiply(Param("rel_detection_unreachable"), where=REACH["unreachable"])
    for comp in ACTIVE_COMPS:
        rate = (
            Param("rel_detection_subclin") * detection if comp.startswith("subclin_") else detection
        )
        model.add_flow(
            TransitionFlow(
                f"tb_detection_{comp}",
                STATE[comp],
                STATE["treatment"],
                rate,
                adjust=(unreachable,),
            )
        )
    model.add_flow(
        TransitionFlow(
            "tx_recovery", STATE["treatment"], STATE["recovered"], success / Param("tx_duration")
        )
    )
    model.add_flow(
        TransitionFlow("tx_relapse", STATE["treatment"], STATE["subclin_lowinf"], outcomes.relapse)
    )


def add_ageing(model: FlowModel) -> None:
    """Ageing through every band at one over its width (summer2 ``AgeStratification``)."""
    model.add_flow(
        TransitionFlow(
            "ageing", AGE.present(), AGE.present(), 1.0, pairing=TraitChain.from_breakpoints(AGE)
        )
    )


def add_births_and_deaths(
    model: FlowModel, demog: Demography, background: object, outcomes: TreatmentOutcomes
) -> None:
    """Deaths recycled to (``mtb_naive``, age 0) in the same reachability; births as entry.

    Natural deaths apply to every compartment except (``mtb_naive``, age 0), as in the original.
    """
    newborn = STATE["mtb_naive"] & AGE[AGE_GROUPS[0]]
    model.add_flow(TransitionFlow("all_cause_mortality", ~newborn, newborn, background))
    for inf in ("inf", "lowinf"):
        model.add_flow(
            TransitionFlow(
                f"tb_mortality_{inf}",
                STATE[f"clin_{inf}"],
                newborn,
                Param(f"tb_mortality_rate_{inf}"),
            )
        )
    model.add_flow(TransitionFlow("tx_death", STATE["treatment"], newborn, outcomes.death))

    entry = births_entry_rate(demog)
    frac = Param("reachable_pop_frac")
    for reach, share in (("reachable", frac), ("unreachable", 1.0 - frac)):
        model.add_flow(EntryFlow(f"births_{reach}", newborn & REACH[reach], entry * share))


def initial_population(demog: Demography) -> InitialPopulation:
    """The first UN year's population, less a seed of 100 in ``clin_inf``, evenly over age."""
    return InitialPopulation(
        {
            STATE["mtb_naive"]: demog.initial_population - SEED,
            STATE["clin_inf"]: SEED,
        },
        splits=(
            Split(REACH, {"reachable": Param("reachable_pop_frac"), "unreachable": REMAINDER}),
        ),
    )


def build_model() -> FlowModel:
    """The model without transmission (phase K1): structure, demography, care cascade."""
    demog = load_demography()
    model = FlowModel(pmap())
    background = death_rate(demog)
    success = treatment_success()
    outcomes = treatment_outcomes(background, success)
    add_natural_history(model)
    add_detection_and_treatment(model, outcomes, success)
    add_ageing(model)
    add_births_and_deaths(model, demog, background, outcomes)
    model.set_initial_population(initial_population(demog))
    return model


def compile_model() -> CompiledModel:
    """Compile :func:`build_model`."""
    return build_model().compile()


START_TIME = 1850.0
END_TIME = 2035.0
SAVE_TIMES: np.ndarray = np.arange(START_TIME, END_TIME + 1.0)
# The goldens use summer2gen's Dormand-Prince odeint at rtol = atol = 1e-8; this is the same
# method and tolerance in diffrax. summer2gen also caps the step at one year; diffrax's
# PIDController takes no step cap through ``CompiledModel.run`` in summer4 v0.2.0a5, and the
# parity suite shows it is not needed at these tolerances.
SOLVER_KWARGS: dict[str, Any] = {"solver": "dopri5", "rtol": 1e-8, "atol": 1e-8}


def solve(compiled: CompiledModel, params: Mapping[str, Any], save: SavePlan) -> Result:
    """Run 1850–2035 with the parity solver settings."""
    return compiled.run(params, t0=START_TIME, t1=END_TIME, dt=1.0, save=save, **SOLVER_KWARGS)


def run_compartments(compiled: CompiledModel, params: Mapping[str, Any]) -> pd.DataFrame:
    """Yearly compartment sizes, one column per summer2 compartment name."""
    plan = SavePlan(requests={"compartments": SaveRequest(Compartments())}, ts=SAVE_TIMES)
    result = solve(compiled, params, plan)
    values = np.asarray(result["compartments"].values.data)
    frame = pd.DataFrame(values, columns=compartment_labels(compiled.pmap), index=SAVE_TIMES)
    frame.index.name = "time"
    return frame
