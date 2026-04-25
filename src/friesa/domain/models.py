# Copyright © 2026 Stable Cyber LLC. All Rights Reserved.
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class Layer(str, Enum):
    physical = "P"
    sensor = "S"
    data = "D"
    compute = "C"
    actuator = "A"
    environment = "E"
    time = "T"


class FailureMode(str, Enum):
    corruption = "corruption"
    omission = "omission"
    delay = "delay"


@dataclass(frozen=True)
class AttackStep:
    layer: Layer
    mode: FailureMode


@dataclass(frozen=True)
class AttackPath:
    steps: list[AttackStep]
