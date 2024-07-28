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
import random
from abc import ABC, abstractmethod
from typing import Optional

from .mersenne_twister import MersenneTwister


class RNG(ABC):
    """
    Random number generator interface
    """

    @abstractmethod
    def next(self) -> float:
        ...


class PythonRNG(ABC):
    """
    Native python RNG.
    """

    def next(self) -> float:
        return random.random()


class MersenneTwisterRNG(RNG):
    """
    Our own RNG based on Mersenne-Twister algorithm.
    """

    def __init__(self, seed: int = 5489):
        self._rng = MersenneTwister()
        self._rng.seed(seed)

    def next(self) -> float:
        return self._rng.next()
