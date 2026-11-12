"""Immutable, request-local sampling configuration."""

import math
from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class SamplingOptions:
    temperature: float = 0.0

    def __post_init__(self):
        if (
            isinstance(self.temperature, bool)
            or not isinstance(self.temperature, (int, float))
            or not math.isfinite(self.temperature)
            or self.temperature < 0
        ):
            raise ValueError("temperature must be finite and nonnegative")

    @property
    def strategy(self) -> Literal["greedy", "temperature"]:
        return "greedy" if self.temperature == 0 else "temperature"
