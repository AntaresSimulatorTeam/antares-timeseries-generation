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

import matplotlib.pyplot as plt
import pytest

from antares.tsgen.duration_generator import (
    GeometricDurationGenerator,
    ProbabilityLaw,
    UniformDurationGenerator,
    make_duration_generator,
)
from antares.tsgen.random_generator import MersenneTwisterRNG


@pytest.mark.parametrize(
    "volatility,expectations,expected",
    [
        (0, [4, 2, 3, 1], [4, 2, 3, 1]),  # Expecting output = input
        (1, [4, 2, 3, 1], [6, 1, 5, 1]),
        (2, [4, 2, 3, 1], [8, 0, 7, 1]),  # Expecting twice the deviation as above
    ],
)
def test_uniform_law_generator(rng, volatility, expectations, expected):
    generator = UniformDurationGenerator(rng, volatility, expectations)
    assert [generator.generate_duration(d) for d in range(len(expected))] == expected


# TODO: results are suprisingly low, check this. Negative results can be obtained...
@pytest.mark.parametrize(
    "volatility,expectations,expected",
    [
        (0, [4, 2, 3, 1], [4, 2, 3, 1]),  # Expecting output = input
        (1, [4, 2, 3, 1], [1, 3, 1, 1]),
        (2, [4, 2, 3, 1], [-1, 5, 0, 1]),  # Expecting twice the deviation as above
    ],
)
def test_geometric_law_generator(rng, volatility, expectations, expected):
    generator = GeometricDurationGenerator(rng, volatility, expectations)
    assert [generator.generate_duration(d) for d in range(len(expected))] == expected


def test_legacy_generator_skips_rng_when_zero_vol():
    rng = MersenneTwisterRNG()
    generator = make_duration_generator(
        rng, ProbabilityLaw.UNIFORM, volatility=0, expectations=[10, 10]
    )
    generator.generate_duration(0)
    # Check we still have a random number identical to the first one that should be generated
    assert rng.next() == MersenneTwisterRNG().next()


def test_geometric_law(rng, output_directory):
    volatility = 1
    generator = GeometricDurationGenerator(rng, volatility, [10])

    expec = 0
    nb_values = 45
    values = [0] * nb_values
    N = 1000000
    N_inv = 1 / N
    for _ in range(N):
        value = generator.generate_duration(0)
        assert value >= 1
        expec += value

        if value < nb_values:
            values[value] += N_inv

    expec /= N
    assert expec == pytest.approx(10, abs=0.1)

    plt.plot(range(nb_values), values)
    plt.savefig(output_directory / "geometric_law_distrib.png")
    plt.clf()


def test_uniform_law(rng, output_directory):
    volatility = 1
    generator = UniformDurationGenerator(rng, volatility, [10])

    expec = 0
    nb_values = 45
    values = [0] * nb_values
    N = 1000000
    N_inv = 1 / N
    for _ in range(N):
        value = generator.generate_duration(0)
        assert value >= 1
        expec += value

        if value < nb_values:
            values[value] += N_inv

    expec /= N
    assert expec == pytest.approx(10, abs=0.1)

    plt.plot(range(nb_values), values)
    plt.savefig(output_directory / "uniform_law_distrib.png")
    plt.clf()
