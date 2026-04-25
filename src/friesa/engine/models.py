# Copyright © 2026 Stable Cyber LLC. All Rights Reserved.
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class OperationalState(str, Enum):
    safe = "S0"
    compromised = "S1"
    detected = "S2"
    contained = "S3"
    recovered = "S4"
    catastrophic = "SF"


class Observability(str, Enum):
    observable = "Obs"
    unobservable = "NoObs"


State = tuple[OperationalState, Observability]


ALL_STATES: list[State] = [
    (operation, observability)
    for operation in OperationalState
    for observability in Observability
]


@dataclass(frozen=True)
class RiskResult:
    k: float
    catastrophe_probability: float
    governance_risk: float
    residual_risk: float
