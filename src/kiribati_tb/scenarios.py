"""The screening scenarios of ``data/scenarios.py``, unchanged in ids, names and programs.

Scenarios 1–3: the PEARL algorithm (CXR-Xpert for 10+, symptom screening for 3–9, TST for 3+).
Scenarios 6–8: dropping Xpert (CXR for 10+, symptom screening 3–9, TST 3+). Scenarios 16–18:
dropping Xpert and TST, CXR for 10+ only. Each at 65%, 75% and 84.5% coverage in 2026.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from kiribati_tb.interventions import CXR, PEARL, SSX, TST, ScreeningProgram, screening_program

COVERAGE: dict[str, float] = {"med": 0.65, "high": 0.75, "vhigh": 0.845}
EFF_ENROLMENT_PERC = 100.0
START_TIME, END_TIME = 2026.0, 2027.0


@dataclass(frozen=True)
class Scenario:
    """A named set of screening programs, with optional parameter overrides."""

    sc_id: str
    sc_name: str
    programs: tuple[ScreeningProgram, ...]
    sc_name_2: str = ""
    description: str = ""
    params_ow: dict[str, float] = field(default_factory=dict)


def _program(
    tool: object, name: str, coverage: float, excluded: tuple[str, ...]
) -> ScreeningProgram:
    return screening_program(
        tool,  # type: ignore[arg-type]
        name,
        coverage * EFF_ENROLMENT_PERC,
        START_TIME,
        END_TIME,
        excluded,
    )


def _pearl(coverage: float) -> tuple[ScreeningProgram, ...]:
    return (
        _program(PEARL, "pearl_10+", coverage, ("0", "3", "5")),
        _program(SSX, "ssx_3_9", coverage, ("0", "10", "15", "18", "40", "65")),
        _program(TST, "tst_3+", coverage, ("0",)),
    )


def _drop_xpert(coverage: float) -> tuple[ScreeningProgram, ...]:
    return (
        _program(CXR, "cxr_10+", coverage, ("0", "3", "5")),
        _program(SSX, "ssx_3_9", coverage, ("0", "10", "15", "18", "40", "65")),
        _program(TST, "tst_3+", coverage, ("0",)),
    )


def _cxr_only(coverage: float) -> tuple[ScreeningProgram, ...]:
    return (_program(CXR, "cxr_10+", coverage, ("0", "3", "5")),)


def build_scenarios() -> list[Scenario]:
    """Scenarios 1–3, 6–8 and 16–18 of the original, in the original's order."""
    scenarios: list[Scenario] = []
    keys = list(COVERAGE)
    desc = (
        "PEARL algorithm: CXR-Xpert for 35% of 10+yr olds (CXR only for other 65%), SSX and "
        "TST for 3+yr olds"
    )
    for num, key in enumerate(keys, start=1):
        cov = COVERAGE[key]
        scenarios.append(
            Scenario(
                f"scenario_{num}",
                f"{num}. PEARL / {key.capitalize()}",
                _pearl(cov),
                f"PEARL | {int(100 * cov)}%",
                desc,
            )
        )
    desc = "Dropping Xpert: CXR for 10+yr olds, SSx and TST for 3+yr olds"
    for num, key in enumerate(keys, start=6):
        cov = COVERAGE[key]
        scenarios.append(
            Scenario(
                f"scenario_{num}",
                f"{num}. CXR-TST / {round(int(100 * cov))}%",
                _drop_xpert(cov),
                f"Drop Xpert | {int(100 * cov)}%",
                desc,
            )
        )
    desc = "Dropping Xpert and TST, stop screening <10yrs: CXR and SSx for 10+yr olds only"
    for num, key in enumerate(keys, start=16):
        cov = COVERAGE[key]
        scenarios.append(
            Scenario(
                f"scenario_{num}",
                f"{num}. CXR 10+yrs / {key.capitalize()}",
                _cxr_only(cov),
                f"Drop Xpert & TST | {int(100 * cov)}%",
                desc,
            )
        )
    return scenarios


SCENARIOS: list[Scenario] = build_scenarios()
SCENARIOS_BY_ID: dict[str, Scenario] = {s.sc_id: s for s in SCENARIOS}
