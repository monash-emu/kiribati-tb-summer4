"""Screening tools and programs (``tbh/interventions.py``).

A screening program screens the reachable population at a constant rate over its campaign,
chosen so that the requested share of the whole population is covered:
``rate = -log(1 - coverage / reachable_pop_frac / 100) / duration``, switched on and off by a
0.01-year linear ramp at each end. Each state a tool detects gets one flow, at the program rate
times the tool's sensitivity for that state times the share successfully started on treatment
(or TPT).
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field

from summer4 import Multiply, Param, Time, log
from summer4.timevarying import linear

TB_STATES: tuple[str, ...] = ("subclin_lowinf", "clin_lowinf", "subclin_inf", "clin_inf")


@dataclass(frozen=True)
class ScreeningTool:
    """What a screen detects and where the people it detects go.

    Attributes:
        name: Tool name (``TST``, ``SSX``, ``CXR``, ``PLTS``, ``PEARL``).
        sensitivities: Detection probability by state, as rate expressions.
        dest: Destination state of the detected.
        success_prop: Share of positives who start treatment (or complete TPT).
    """

    name: str
    sensitivities: tuple[tuple[str, object], ...]
    dest: str
    success_prop: object = 1.0


def _by_state(test: str) -> tuple[tuple[str, object], ...]:
    return tuple((s, Param(f"prev_se_{s}_{test}")) for s in TB_STATES)


TST = ScreeningTool(
    "TST",
    (("incipient", Param("prev_se_incipient_tst")), ("contained", Param("prev_se_contained_tst"))),
    dest="cleared",
    success_prop=Param("tpt_completion_perc") / 100.0,
)
SSX = ScreeningTool("SSX", _by_state("ssx"), dest="treatment")
CXR = ScreeningTool("CXR", _by_state("cxr"), dest="treatment")
PLTS = ScreeningTool("PLTS", _by_state("plts"), dest="treatment")
# PEARL: CXR-Xpert for 35% of those screened, CXR alone for the other 65%.
PEARL = ScreeningTool(
    "PEARL",
    tuple(
        (s, 0.35 * Param(f"prev_se_{s}_pearl") + 0.65 * Param(f"prev_se_{s}_cxr"))
        for s in TB_STATES
    ),
    dest="treatment",
)


@dataclass(frozen=True)
class ScreeningProgram:
    """One screening campaign.

    Attributes:
        name: Program name; flows are named ``{name}_{state}``.
        start_time: Campaign start (year).
        end_time: Campaign end (year).
        total_coverage_perc: Share of the whole population screened, in percent.
        tool: The screening tool.
        age_multipliers: Coverage multiplier by age band (0 excludes a band); unlisted bands 1.
    """

    name: str
    start_time: float
    end_time: float
    total_coverage_perc: float
    tool: ScreeningTool
    age_multipliers: tuple[tuple[str, float], ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        if self.end_time <= self.start_time:
            raise ValueError("End time must be after start time")

    def raw_rate(self) -> object:
        """Per-person screening rate among the reachable, as a function of time."""
        among_reachable = self.total_coverage_perc / Param("reachable_pop_frac")
        duration = self.end_time - self.start_time
        rate = -log(1.0 - among_reachable / 100.0) / duration
        s, e = self.start_time, self.end_time
        return linear(Time(), (s - 0.01, s, e - 0.01, e), (0.0, rate, rate, 0.0))

    def flow_names(self) -> tuple[str, ...]:
        return tuple(f"{self.name}_{state}" for state, _ in self.tool.sensitivities)

    def age_adjustments(self) -> tuple[Multiply, ...]:
        """``Multiply`` by each listed band's coverage multiplier."""
        from kiribati_tb.model import AGE

        return tuple(Multiply(m, where=AGE[age]) for age, m in self.age_multipliers)


def screening_program(
    tool: ScreeningTool,
    name: str,
    coverage: float,
    start_time: float,
    end_time: float,
    ages_excluded: Mapping[str, float] | tuple[str, ...] = (),
) -> ScreeningProgram:
    """``make_scr_program`` from ``data/scenarios.py``: exclude some age bands entirely."""
    return ScreeningProgram(
        name=name,
        start_time=start_time,
        end_time=end_time,
        total_coverage_perc=coverage,
        tool=tool,
        age_multipliers=tuple((age, 0.0) for age in ages_excluded),
    )
