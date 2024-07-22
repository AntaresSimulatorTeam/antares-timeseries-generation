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

from pathlib import Path

import pytest

from random_generator import RNG, MersenneTwisterRNG


@pytest.fixture
def data_directory() -> Path:
    return Path(__file__).parent / "data"


@pytest.fixture
def output_directory() -> Path:
    return Path(__file__).parent / "output"

@pytest.fixture
def rng() -> RNG:
    return MersenneTwisterRNG()