"""Derived outputs: one ``OutputSet`` with every name the original requested.

Ports ``request_model_outputs`` from ``tbh/outputs.py``, including the ``save_results=False``
intermediates, so the names and values match the original's ``derived_outputs`` frame. Every
compartment output reads one shared ``Compartments()`` save and every flow output one
``FlowMass`` save per flow; the selections and sums are post-ops.

Flow outputs use summer2's midpoint convention (``FlowMass(...).midpoint()``). Cumulative
outputs start in 2026.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

import jax.numpy as jnp
import numpy as np
import pandas as pd

from summer4 import (
    CompiledModel,
    Compartments,
    FlowMass,
    OutputSet,
    Param,
    Result,
    SaveFn,
    SavePlan,
    Selector,
    Source,
)
from summer4.results.outputset import OutputExpr

from kiribati_tb.demography import AGE_GROUPS
from kiribati_tb.mixing import START_YEAR, canberra_distance, read_conmat_matrix
from kiribati_tb.model import (
    ACTIVE_COMPS,
    AGE,
    COMPARTMENTS,
    LATENT_COMPS,
    MIXING_STACK,
    REACH,
    REACH_STRATA,
    SAVE_TIMES,
    STATE,
    ModelConfig,
    solve,
)

REACHABLE = "reachable"
CUMULATIVE_START = 2026.0
AGE_AGGREGATES: dict[str, tuple[str, ...]] = {
    "3_9": ("3", "5"),
    "15+": tuple(a for a in AGE_GROUPS if int(a) >= 15),
    "18+": tuple(a for a in AGE_GROUPS if int(a) >= 18),
}


def _total(outputs: OutputSet, names: Iterable[str]) -> OutputExpr:
    refs = [outputs.ref(name) for name in names]
    expr = refs[0]
    for ref in refs[1:]:
        expr = expr + ref
    return expr


def _compartments(where: Selector | None = None) -> OutputExpr:
    """Total of the compartments matching ``where``, from the one shared compartment save."""
    leaf = Compartments()
    return leaf.total() if where is None else leaf.select(where).total()


def _flow(flow: str, where: Selector | None = None) -> OutputExpr:
    """Midpoint flow output of ``flow`` over the edges whose source matches ``where``."""
    leaf = FlowMass(flow)
    selected = leaf.total() if where is None else leaf.select(Source(where)).total()
    return selected.midpoint()


def _per_capita(outputs: OutputSet, name: str, per: float, denominator: str = "population") -> None:
    """``request_per_capita_output``: ``{name}_{perc|per100k}``."""
    suffix = {100.0: "perc", 100000.0: "per100k"}.get(per, f"per{per}")
    outputs[f"{name}_{suffix}"] = per * outputs.ref(name) / outputs.ref(denominator)


def passive_detection_clin(ctx: Any) -> Any:
    """``recent_detection_rate`` × tanh scale-up at the save time (the flow's own rate)."""
    p = ctx.derived
    scale = jnp.tanh(p["passive_detection_shape"] * (ctx.t - p["passive_detection_inflection"]))
    past = p["passive_detection_past_frac"]
    return p["recent_detection_rate"] * ((scale / 2.0 + 0.5) * (1.0 - past) + past)


def passive_detection_subclin(ctx: Any) -> Any:
    """Subclinical passive detection rate: ``rel_detection_subclin`` × the clinical rate."""
    return ctx.derived["rel_detection_subclin"] * passive_detection_clin(ctx)


_CONMAT = read_conmat_matrix()


def mixing_matrix_distance(ctx: Any) -> Any:
    """Canberra distance from this year's mixing matrix to the ``conmat`` matrix."""
    stack = ctx.derived[MIXING_STACK]
    year = jnp.clip(jnp.floor(ctx.t - START_YEAR).astype(jnp.int32), 0, stack.shape[0] - 1)
    return canberra_distance(stack[year], jnp.asarray(_CONMAT))


def add_population(outputs: OutputSet) -> None:
    outputs["population"] = _compartments()
    for age in AGE_GROUPS:
        outputs[f"populationXage_{age}"] = _compartments(AGE[age])
        outputs[f"populationXage_{age}Xreach_{REACHABLE}"] = _compartments(
            AGE[age] & REACH[REACHABLE]
        )
    for label, ages in AGE_AGGREGATES.items():
        outputs[f"populationXage_{label}"] = _total(outputs, [f"populationXage_{a}" for a in ages])
        outputs[f"populationXage_{label}Xreach_{REACHABLE}"] = _total(
            outputs, [f"populationXage_{a}Xreach_{REACHABLE}" for a in ages]
        )
    for reach in REACH_STRATA:
        outputs[f"populationXreach_{reach}"] = _compartments(REACH[reach])


def add_births_and_incidence(outputs: OutputSet) -> None:
    for reach in REACH_STRATA:
        outputs[f"births_{reach}"] = _flow(f"births_{reach}")
    outputs["births"] = _total(outputs, [f"births_{r}" for r in REACH_STRATA])

    for inf in ("lowinf", "inf"):
        outputs[f"tb_incidence_{inf}"] = _flow(f"progression_{inf}")
        for reach in REACH_STRATA:
            outputs[f"tb_incidence_{inf}Xreach_{reach}"] = _flow(f"progression_{inf}", REACH[reach])
    for reach in REACH_STRATA:
        outputs[f"tb_incidenceXreach_{reach}"] = _total(
            outputs, [f"tb_incidence_{inf}Xreach_{reach}" for inf in ("lowinf", "inf")]
        )
    outputs["tb_incidence"] = _total(outputs, [f"tb_incidenceXreach_{r}" for r in REACH_STRATA])
    _per_capita(outputs, "tb_incidence", 100000.0)
    outputs["prop_tb_incidenceXreach_unreachable"] = outputs.ref(
        "tb_incidenceXreach_unreachable"
    ) / outputs.ref("tb_incidence")
    outputs["cum_tb_incidence"] = outputs.ref("tb_incidence").cumulative(start=CUMULATIVE_START)


def add_prevalence(outputs: OutputSet) -> None:
    for comp in COMPARTMENTS:
        for age in AGE_GROUPS:
            outputs[f"prev_{comp}Xage_{age}"] = _compartments(STATE[comp] & AGE[age])
            for reach in REACH_STRATA:
                outputs[f"prev_{comp}Xage_{age}Xreach_{reach}"] = _compartments(
                    STATE[comp] & AGE[age] & REACH[reach]
                )
        outputs[f"prev_{comp}"] = _total(outputs, [f"prev_{comp}Xage_{a}" for a in AGE_GROUPS])
        for reach in REACH_STRATA:
            outputs[f"prev_{comp}Xreach_{reach}"] = _total(
                outputs, [f"prev_{comp}Xage_{a}Xreach_{reach}" for a in AGE_GROUPS]
            )
    for state, comps, per in (("tbi", LATENT_COMPS, 100.0), ("tb", ACTIVE_COMPS, 100000.0)):
        outputs[f"{state}_prevalence"] = _total(outputs, [f"prev_{c}" for c in comps])
        _per_capita(outputs, f"{state}_prevalence", per)


def _tst_sensitivity(comp: str) -> Any:
    if comp in LATENT_COMPS:
        return Param(f"prev_se_{comp}_tst")
    if comp in ACTIVE_COMPS or comp == "treatment":
        return 1.0
    if comp == "recovered":
        return Param("prev_se_cleared_tst")
    return 0.0


def add_screening_positivity(outputs: OutputSet) -> None:
    """Measured TST, PEARL and CXR positivity among the reachable (the calibration targets)."""
    ages_and_aggregates = (*AGE_GROUPS, *AGE_AGGREGATES)
    for comp in COMPARTMENTS:
        for age in AGE_GROUPS:
            outputs[f"tst_pos_{comp}Xage_{age}Xreach_{REACHABLE}"] = outputs.ref(
                f"prev_{comp}Xage_{age}Xreach_{REACHABLE}"
            ) * _tst_sensitivity(comp)
        for label, ages in AGE_AGGREGATES.items():
            outputs[f"tst_pos_{comp}Xage_{label}Xreach_{REACHABLE}"] = _total(
                outputs, [f"tst_pos_{comp}Xage_{a}Xreach_{REACHABLE}" for a in ages]
            )
    for age in ages_and_aggregates:
        name = f"tst_posXage_{age}Xreach_{REACHABLE}"
        outputs[name] = _total(
            outputs, [f"tst_pos_{c}Xage_{age}Xreach_{REACHABLE}" for c in COMPARTMENTS]
        )
        _per_capita(outputs, name, 100.0, f"populationXage_{age}Xreach_{REACHABLE}")
    name = f"tst_posXreach_{REACHABLE}"
    outputs[name] = _total(outputs, [f"tst_posXage_{a}Xreach_{REACHABLE}" for a in AGE_GROUPS])
    _per_capita(outputs, name, 100.0, f"populationXreach_{REACHABLE}")

    for comp in ACTIVE_COMPS:
        for age in AGE_GROUPS:
            for test in ("pearl", "cxr"):
                outputs[f"{test}_prev_{comp}Xage_{age}Xreach_{REACHABLE}"] = outputs.ref(
                    f"prev_{comp}Xage_{age}Xreach_{REACHABLE}"
                ) * Param(f"prev_se_{comp}_{test}")
    for test in ("pearl", "cxr"):
        for age in AGE_GROUPS:
            name = f"{test}_posXage_{age}Xreach_{REACHABLE}"
            outputs[name] = _total(
                outputs, [f"{test}_prev_{c}Xage_{age}Xreach_{REACHABLE}" for c in ACTIVE_COMPS]
            )
            _per_capita(outputs, name, 100000.0, f"populationXage_{age}Xreach_{REACHABLE}")
        name = f"{test}_posXreach_{REACHABLE}"
        outputs[name] = _total(
            outputs, [f"{test}_posXage_{a}Xreach_{REACHABLE}" for a in AGE_GROUPS]
        )
        _per_capita(outputs, name, 100000.0, f"populationXreach_{REACHABLE}")


def add_disease_profile(outputs: OutputSet) -> None:
    """Viable infection, % subclinical, % infectious."""
    outputs["viable_tbi_prevalence"] = _total(outputs, ["prev_incipient", "prev_contained"])
    _per_capita(outputs, "viable_tbi_prevalence", 100.0)

    subclin = STATE.isin([c for c in ACTIVE_COMPS if c.startswith("subclin_")])
    outputs[f"subclin_tb_prevalenceXreach_{REACHABLE}"] = _compartments(subclin & REACH[REACHABLE])
    for reach in REACH_STRATA:
        name = f"tb_prevalenceXreach_{reach}"
        outputs[name] = _total(outputs, [f"prev_{c}Xreach_{reach}" for c in ACTIVE_COMPS])
        _per_capita(outputs, name, 100000.0, f"populationXreach_{reach}")
    outputs[f"perc_prev_subclinicalXreach_{REACHABLE}"] = (
        100.0
        * outputs.ref(f"subclin_tb_prevalenceXreach_{REACHABLE}")
        / outputs.ref(f"tb_prevalenceXreach_{REACHABLE}")
    )
    infectious = STATE.isin([c for c in ACTIVE_COMPS if c.endswith("_inf")])
    outputs[f"infectious_tb_prevalenceXreach_{REACHABLE}"] = _compartments(
        infectious & REACH[REACHABLE]
    )
    outputs[f"perc_prev_infectiousXreach_{REACHABLE}"] = (
        100.0
        * outputs.ref(f"infectious_tb_prevalenceXreach_{REACHABLE}")
        / outputs.ref(f"tb_prevalenceXreach_{REACHABLE}")
    )


def add_notifications_and_screening(outputs: OutputSet, screening_flows: Sequence[str]) -> None:
    for comp in ACTIVE_COMPS:
        outputs[f"notifications_{comp}"] = _flow(f"tb_detection_{comp}")
    outputs["notifications"] = _total(outputs, [f"notifications_{c}" for c in ACTIVE_COMPS])
    outputs["perc_notifications_clin"] = (
        100.0
        * (outputs.ref("notifications_clin_lowinf") + outputs.ref("notifications_clin_inf"))
        / outputs.ref("notifications")
    )
    for flow in screening_flows:
        outputs[flow] = _flow(flow)
    if screening_flows:
        outputs["screening"] = _total(outputs, screening_flows)
    else:
        # summer2 aggregates an empty source list to zeros.
        outputs["screening"] = 0.0 * outputs.ref("births")


def add_mortality(outputs: OutputSet) -> None:
    natural = []
    for comp in COMPARTMENTS:
        for i, age in enumerate(AGE_GROUPS):
            if i > 0 or comp != "mtb_naive":
                name = f"all_cause_mortality_from_{comp}_age_{age}"
                outputs[name] = _flow("all_cause_mortality", STATE[comp] & AGE[age])
                natural.append(name)
    outputs["nat_mortality"] = _total(outputs, natural)

    tb_deaths = []
    for inf in ("inf", "lowinf"):
        for age in AGE_GROUPS:
            name = f"tb_mortality_{inf}_age_{age}"
            outputs[name] = _flow(f"tb_mortality_{inf}", AGE[age])
            tb_deaths.append(name)
    for age in AGE_GROUPS:
        name = f"tx_death_age_{age}"
        outputs[name] = _flow("tx_death", AGE[age])
        tb_deaths.append(name)
    outputs["tb_mortality"] = _total(outputs, tb_deaths)
    _per_capita(outputs, "tb_mortality", 100000.0)
    outputs["cum_tb_mortality"] = outputs.ref("tb_mortality").cumulative(start=CUMULATIVE_START)


def add_computed_values(outputs: OutputSet, config: ModelConfig) -> None:
    """Passive detection rates and, with age mixing, the distance to the ``conmat`` matrix.

    WORKAROUND(summer4): these are ``SaveFn`` leaves that recompute the values from the save time
    and the prepared parameters. summer4 cannot save the value of a rate expression that is not
    grouped (``Capture``/``GroupedOutput`` need a ``GroupedRate``), and ``ComputedValue`` needs
    a ``derived_fn`` that would run in every vector-field call to produce a number only the
    outputs read. See docs/summer4-workarounds.md, W2.
    """
    outputs["passive_detection_rate_clin"] = SaveFn(passive_detection_clin)
    outputs["passive_detection_rate_subclin"] = SaveFn(passive_detection_subclin)
    if config.heterogeneous_mixing:
        outputs["mixing_matrix_distance"] = SaveFn(mixing_matrix_distance)


def build_outputs(
    config: ModelConfig = ModelConfig(), screening_flows: Sequence[str] = ()
) -> OutputSet:
    """Every derived output of the original model, by the original's names."""
    outputs = OutputSet()
    add_population(outputs)
    add_births_and_incidence(outputs)
    add_prevalence(outputs)
    add_screening_positivity(outputs)
    add_disease_profile(outputs)
    add_notifications_and_screening(outputs, screening_flows)
    add_mortality(outputs)
    add_computed_values(outputs, config)
    return outputs


def derived_outputs_frame(named: Result) -> pd.DataFrame:
    """The original's ``derived_outputs`` frame: year index, one column per output."""
    frame = named.to_frame(shape="wide", backend="pandas")
    frame.index = np.asarray(frame.index, dtype=float)
    frame.index.name = "time"
    return frame


def run_outputs(
    compiled: CompiledModel, outputs: OutputSet, params: Mapping[str, Any]
) -> pd.DataFrame:
    """Solve 1850–2035 and return the ``derived_outputs`` frame for ``outputs``."""
    result = solve(compiled, params, outputs.plan(SavePlan(ts=SAVE_TIMES)))
    return derived_outputs_frame(outputs.evaluate(result, params))


def _inline(expr: OutputExpr, outputs: OutputSet, keep: set[str]) -> OutputExpr:
    """``expr`` with every reference to a name outside ``keep`` replaced by its expression."""
    if expr.kind == "ref" and expr.name is not None and expr.name not in keep:
        return _inline(outputs[expr.name], outputs, keep)
    if not expr.kids:
        return expr
    return dataclasses.replace(expr, kids=tuple(_inline(k, outputs, keep) for k in expr.kids))


def restrict_outputs(outputs: OutputSet, names: Iterable[str]) -> OutputSet:
    """An ``OutputSet`` with only ``names`` as keys; other outputs they read are inlined.

    Leaves are shared saves either way, so nothing extra is solved for. Used where summer4 acts
    on every key of a set: a calibration scores ten targets, and ``posterior_runs`` records
    every key for every draw (see docs/summer4-workarounds.md, W5).
    """
    keep = set(names)
    restricted = OutputSet()
    for name in outputs.keys():
        if name in keep:
            restricted[name] = _inline(outputs[name], outputs, keep)
    missing = keep - set(restricted.keys())
    if missing:
        raise KeyError(f"Unknown outputs: {sorted(missing)}")
    return restricted
