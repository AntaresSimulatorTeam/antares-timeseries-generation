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

from antares.tsgen.random_generator import MersenneTwisterRNG, PythonRNG


def test_python_rng():
    rng = PythonRNG()
    random.seed(10)
    expected = [random.random() for _ in range(10)]
    random.seed(10)
    assert [rng.next() for _ in range(10)] == expected


def test_custom_rng():
    rng = MersenneTwisterRNG()
    assert [rng.next() for _ in range(10)] == [
        0.8147236920927473,
        0.13547700413863104,
        0.9057919343248456,
        0.835008589978099,
        0.12698681189841285,
        0.968867771320247,
        0.9133758558690026,
        0.22103404282150652,
        0.6323592501302154,
        0.30816705043152137,
    ]
