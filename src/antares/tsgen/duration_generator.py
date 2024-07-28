# Copyright (c) 2024, RTE (https://www.rte-france.com)
#
# See AUTHORS.txt
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
#
# SPDX-License-Identifier: MPL-2.0
#
# This file is part of the Antares project.
from abc import ABC, abstractmethod
from enum import Enum
from math import log, sqrt

import numpy as np
import numpy.typing as npt

from .random_generator import RNG


class ProbabilityLaw(Enum):
    UNIFORM = "UNIFORM"
    GEOMETRIC = "GEOMETRIC"


class DurationGenerator(ABC):
    """
    Logic to generate unavailability duration.
    """

    @abstractmethod
    def generate_duration(self, day: int) -> int:
        ...


class GeneratorWrapper(DurationGenerator):
    """
    Wraps another generator to skip unnecessary random number generations.
    Used to keep backward compat with cpp implementation.
    """

    def __init__(
        self,
        delegate: DurationGenerator,
        volatility: float,
        expecs: npt.NDArray[np.int_],
    ) -> None:
        self.volatility = volatility
        self.expectations = expecs
        self.delegate = delegate

    def generate_duration(self, day: int) -> int:
        """
        generation of random outage duration
        """
        expectation = self.expectations[day]
        # Logic copied from cpp implementation for results preservation
        if self.volatility == 0 or expectation == 1:
            return expectation
        return self.delegate.generate_duration(day)


class UniformDurationGenerator(DurationGenerator):
    def __init__(
        self, rng: RNG, volatility: float, expecs: npt.NDArray[np.int_]
    ) -> None:
        self.rng = rng
        self.a = np.empty(len(expecs), dtype=float)
        self.b = np.empty(len(expecs), dtype=float)
        for day, expec in enumerate(expecs):
            xtemp = volatility * (expec - 1)
            self.a[day] = expec - xtemp
            self.b[day] = 2 * xtemp + 1

    def generate_duration(self, day: int) -> int:
        """
        generation of random outage duration
        """
        rnd_nb = self.rng.next()
        return int(self.a[day] + rnd_nb * self.b[day])


class GeometricDurationGenerator(DurationGenerator):
    def __init__(
        self, rng: RNG, volatility: float, expecs: npt.NDArray[np.int_]
    ) -> None:
        self.rng = rng
        self.a = np.empty(len(expecs), dtype=float)
        self.b = np.empty(len(expecs), dtype=float)
        for day, expec in enumerate(expecs):
            xtemp = volatility * volatility * expec * (expec - 1)
            if xtemp != 0:
                ytemp = (sqrt(4 * xtemp + 1) - 1) / (2 * xtemp)
                self.a[day] = expec - 1 / ytemp
                self.b[day] = 1 / log(1 - ytemp)
            else:
                self.a[day] = expec - 1
                self.b[day] = 0

    def generate_duration(self, day: int) -> int:
        """
        generation of random outage duration
        """
        rnd_nb = self.rng.next()
        return min(int(1 + self.a[day] + self.b[day] * log(rnd_nb)), 1999)


def make_duration_generator(
    rng: RNG, law: ProbabilityLaw, volatility: float, expectations: npt.NDArray[np.int_]
) -> DurationGenerator:
    """
    return a DurationGenerator for the given law
    """
    base_rng: DurationGenerator
    if law == ProbabilityLaw.UNIFORM:
        base_rng = UniformDurationGenerator(rng, volatility, expectations)
    elif law == ProbabilityLaw.GEOMETRIC:
        base_rng = GeometricDurationGenerator(rng, volatility, expectations)
    else:
        raise ValueError(f"Unknown law type: {law}")
    return GeneratorWrapper(base_rng, volatility, expectations)
