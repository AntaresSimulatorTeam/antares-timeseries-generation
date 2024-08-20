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

import numpy as np
import numpy.testing as npt
import pytest

from antares.tsgen.random_generator import MersenneTwisterRNG
from antares.tsgen.ts_generator import (
    ProbabilityLaw,
    ThermalCluster,
    ThermalDataGenerator,
    _categorize_outages,
    _check_cluster,
    _column_powers,
    _daily_to_hourly,
)


def test_daily_to_hourly():
    daily = np.array([[1, 2]])
    hourly = _daily_to_hourly(daily)
    expected = [[1, 2]] * 24
    npt.assert_equal(hourly, expected)


def test_elevate_to_power():
    input = np.array([1, 0.5, 0.1])
    powers = _column_powers(input, 3)
    expected = np.array([[1, 1, 1], [1, 0.5, 0.25], [1, 0.1, 0.01]])
    npt.assert_almost_equal(powers, expected, decimal=3)


@pytest.fixture()
def base_cluster_365_days():
    days = 365
    return ThermalCluster(
        unit_count=10,
        nominal_power=100,
        modulation=np.ones(dtype=float, shape=8760),
        fo_law=ProbabilityLaw.UNIFORM,
        fo_volatility=0,
        po_law=ProbabilityLaw.UNIFORM,
        po_volatility=0,
        fo_duration=10 * np.ones(dtype=int, shape=days),
        fo_rate=0.2 * np.ones(dtype=float, shape=days),
        po_duration=10 * np.ones(dtype=int, shape=days),
        po_rate=np.zeros(dtype=float, shape=days),
        npo_min=np.zeros(dtype=int, shape=days),
        npo_max=10 * np.ones(dtype=int, shape=days),
    )


def test_invalid_fo_rates(rng, base_cluster_365_days):
    days = 365
    cluster = base_cluster_365_days
    cluster.fo_rate[12] = -0.2
    cluster.fo_rate[10] = -0.1

    with pytest.raises(
        ValueError,
        match="Forced failure rate is negative on following days: \[10, 12\]",
    ):
        generator = ThermalDataGenerator(rng=rng, days=days)
        generator.generate_time_series(cluster, 1)


def test_invalid_po_rates(rng, base_cluster_365_days):
    days = 365
    cluster = base_cluster_365_days
    cluster.po_rate[12] = -0.2
    cluster.po_rate[10] = -0.1

    with pytest.raises(
        ValueError,
        match="Planned failure rate is negative on following days: \[10, 12\]",
    ):
        generator = ThermalDataGenerator(rng=rng, days=days)
        generator.generate_time_series(cluster, 1)


def valid_cluster() -> ThermalCluster:
    days = 365
    return ThermalCluster(
        unit_count=10,
        nominal_power=100,
        modulation=np.ones(dtype=float, shape=8760),
        fo_law=ProbabilityLaw.UNIFORM,
        fo_volatility=0,
        po_law=ProbabilityLaw.UNIFORM,
        po_volatility=0,
        fo_duration=10 * np.ones(dtype=int, shape=days),
        fo_rate=0.2 * np.ones(dtype=float, shape=days),
        po_duration=10 * np.ones(dtype=int, shape=days),
        po_rate=np.zeros(dtype=float, shape=days),
        npo_min=np.zeros(dtype=int, shape=days),
        npo_max=10 * np.ones(dtype=int, shape=days),
    )


def test_invalid_cluster():
    cluster = valid_cluster()
    _check_cluster(cluster)

    cluster = valid_cluster()
    with pytest.raises(ValueError):
        cluster.nominal_power = -1
        _check_cluster(cluster)

    cluster = valid_cluster()
    with pytest.raises(ValueError):
        cluster.unit_count = -1
        _check_cluster(cluster)

    cluster = valid_cluster()
    with pytest.raises(ValueError):
        cluster.fo_duration[10] = -1
        _check_cluster(cluster)

    cluster = valid_cluster()
    with pytest.raises(ValueError):
        cluster.po_duration[10] = -1
        _check_cluster(cluster)

    cluster = valid_cluster()
    with pytest.raises(ValueError):
        cluster.modulation[10] = -1
        _check_cluster(cluster)

    cluster = valid_cluster()
    with pytest.raises(ValueError):
        cluster.modulation = np.ones(30)
        _check_cluster(cluster)

    cluster = valid_cluster()
    with pytest.raises(ValueError):
        cluster.fo_rate = cluster.fo_rate[:-2]
        _check_cluster(cluster)


@pytest.mark.parametrize(
    "available_units,po_candidates,fo_candidates,expected",
    [
        (0, 0, 0, (0, 0, 0)),
        (1, 1, 1, (1, 0, 0)),
        (1, 1, 0, (0, 1, 0)),
        (1, 0, 1, (0, 0, 1)),
        (10, 1, 1, (0, 1, 1)),
        (10, 5, 5, (2, 3, 3)),
        (10, 8, 3, (2, 6, 1)),
    ],
)
def test_distribute_outages(available_units, po_candidates, fo_candidates, expected):
    outages = _categorize_outages(available_units, po_candidates, fo_candidates)
    assert outages == expected


def test_forced_outages(rng):
    days = 365
    cluster = ThermalCluster(
        unit_count=10,
        nominal_power=100,
        modulation=np.ones(dtype=float, shape=8760),
        fo_law=ProbabilityLaw.UNIFORM,
        fo_volatility=0,
        po_law=ProbabilityLaw.UNIFORM,
        po_volatility=0,
        fo_duration=10 * np.ones(dtype=int, shape=days),
        fo_rate=0.2 * np.ones(dtype=float, shape=days),
        po_duration=10 * np.ones(dtype=int, shape=days),
        po_rate=np.zeros(dtype=float, shape=days),
        npo_min=np.zeros(dtype=int, shape=days),
        npo_max=10 * np.ones(dtype=int, shape=days),
    )
    cluster.modulation[12] = 0.5

    generator = ThermalDataGenerator(rng=rng, days=days)
    results = generator.generate_time_series(cluster, 1)
    # 2 forced outages occur on day 5, with duration 10
    npt.assert_equal(results.forced_outages.T[0][:6], [0, 0, 0, 0, 2, 0])
    npt.assert_equal(results.forced_outage_durations.T[0][:6], [0, 0, 0, 0, 10, 0])
    # No planned outage
    npt.assert_equal(results.planned_outages.T[0], np.zeros(365))
    npt.assert_equal(results.planned_outage_durations.T[0], np.zeros(365))

    npt.assert_equal(results.available_units.T[0][:5], [9, 9, 9, 9, 8])
    # Check available power consistency with available units and modulation
    available_power = results.available_power.T
    assert available_power[0][0] == 900
    assert available_power[0][12] == 450  # Modulation is 0.5 for hour 12
    assert available_power[0][4 * 24] == 800


def test_planned_outages(rng):
    days = 365
    cluster = ThermalCluster(
        unit_count=10,
        nominal_power=100,
        modulation=np.ones(dtype=float, shape=8760),
        fo_law=ProbabilityLaw.UNIFORM,
        fo_volatility=0,
        po_law=ProbabilityLaw.UNIFORM,
        po_volatility=0,
        fo_duration=10 * np.ones(dtype=int, shape=days),
        fo_rate=np.zeros(dtype=float, shape=days),
        po_duration=10 * np.ones(dtype=int, shape=days),
        po_rate=0.2 * np.ones(dtype=float, shape=days),
        npo_min=np.zeros(dtype=int, shape=days),
        npo_max=10 * np.ones(dtype=int, shape=days),
    )
    cluster.modulation[12] = 0.5

    generator = ThermalDataGenerator(rng=rng, days=days)
    results = generator.generate_time_series(cluster, 1)
    # 0 forced outage
    npt.assert_equal(results.forced_outages.T[0], np.zeros(365))
    npt.assert_equal(results.forced_outage_durations.T[0], np.zeros(365))
    # No planned outage
    npt.assert_equal(results.planned_outages.T[0][:6], [0, 0, 0, 0, 2, 0])
    npt.assert_equal(results.available_units.T[0][:5], [9, 9, 9, 9, 8])
    # Check available power consistency with available units and modulation
    available_power = results.available_power.T
    assert available_power[0][0] == 900
    assert available_power[0][12] == 450  # Modulation is 0.5 for hour 12
    assert available_power[0][4 * 24] == 800


def test_planned_outages_limitation(rng):
    days = 365
    # Maximum 1 planned outage at a time.
    cluster = ThermalCluster(
        unit_count=10,
        nominal_power=100,
        modulation=np.ones(dtype=float, shape=8760),
        fo_law=ProbabilityLaw.UNIFORM,
        fo_volatility=0,
        po_law=ProbabilityLaw.UNIFORM,
        po_volatility=0,
        fo_duration=10 * np.ones(dtype=int, shape=days),
        fo_rate=np.zeros(dtype=float, shape=days),
        po_duration=2 * np.ones(dtype=int, shape=days),
        po_rate=0.2 * np.ones(dtype=float, shape=days),
        npo_min=np.zeros(dtype=int, shape=days),
        npo_max=1 * np.ones(dtype=int, shape=days),
    )

    generator = ThermalDataGenerator(rng=rng, days=days)
    results = generator.generate_time_series(cluster, 1)
    # No forced outage
    npt.assert_equal(results.forced_outages.T[0], np.zeros(365))
    npt.assert_equal(results.forced_outage_durations.T[0], np.zeros(365))
    # Maxmimum one planned outage at a time
    npt.assert_equal(results.planned_outages.T[0][:6], [1, 0, 1, 0, 1, 0])
    npt.assert_equal(results.planned_outage_durations.T[0][:6], [2, 0, 2, 0, 2, 0])
    npt.assert_equal(results.available_units.T[0][:5], [9, 9, 9, 9, 9])
    # Check available power consistency with available units and modulation
    available_power = results.available_power.T
    assert available_power[0][0] == 900
    assert available_power[0][4 * 24] == 900


def test_planned_outages_min_limitation(rng):
    days = 365
    # Minimum 2 planned outages at a time
    cluster = ThermalCluster(
        unit_count=10,
        nominal_power=100,
        modulation=np.ones(dtype=float, shape=8760),
        fo_law=ProbabilityLaw.UNIFORM,
        fo_volatility=0,
        po_law=ProbabilityLaw.UNIFORM,
        po_volatility=0,
        fo_duration=10 * np.ones(dtype=int, shape=days),
        fo_rate=np.zeros(dtype=float, shape=days),
        po_duration=10 * np.ones(dtype=int, shape=days),
        po_rate=0.2 * np.ones(dtype=float, shape=days),
        npo_min=2 * np.ones(dtype=int, shape=days),
        npo_max=5 * np.ones(dtype=int, shape=days),
    )

    generator = ThermalDataGenerator(rng=rng, days=days)
    results = generator.generate_time_series(cluster, 1)
    # No forced outage
    npt.assert_equal(results.forced_outages.T[0], np.zeros(365))
    npt.assert_equal(results.forced_outage_durations.T[0], np.zeros(365))
    # Maxmimum one planned outage at a time
    npt.assert_equal(results.planned_outages.T[0][:6], [0, 0, 1, 0, 0, 1])
    npt.assert_equal(results.planned_outage_durations.T[0][:6], [0, 0, 10, 0, 0, 10])
    npt.assert_equal(results.available_units.T[0][:5], [8, 8, 8, 8, 8])
    # Check available power consistency with available units and modulation
    available_power = results.available_power.T
    assert available_power[0][0] == 800
    assert available_power[0][4 * 24] == 800


def test_with_long_fo_and_po_duration(data_directory):
    days = 365
    modulation_matrix = np.ones(8760, dtype=float)
    modulation_matrix[:24] = 0.5
    modulation_matrix[24:52] = 0.1
    fo_duration = np.full(days, 1, dtype=int)
    po_duration = np.full(days, 1, dtype=int)
    fo_rate = np.full(days, 0.01, dtype=float)
    po_rate = np.full(days, 0.3, dtype=float)
    fo_duration[:31] = 2
    po_duration[:31] = 3
    fo_rate[:31] = 0.1
    po_rate[:31] = 0.02
    cluster = ThermalCluster(
        unit_count=10,
        nominal_power=500,
        modulation=modulation_matrix,
        fo_law=ProbabilityLaw.UNIFORM,
        fo_volatility=0.5,
        po_law=ProbabilityLaw.GEOMETRIC,
        po_volatility=0.5,
        fo_duration=fo_duration,
        fo_rate=fo_rate,
        po_duration=po_duration,
        po_rate=po_rate,
        npo_min=0 * np.ones(dtype=int, shape=days),
        npo_max=3 * np.ones(dtype=int, shape=days),
    )
    rng = MersenneTwisterRNG(seed=3005489)
    generator = ThermalDataGenerator(rng=rng, days=days)
    results = generator.generate_time_series(cluster, 10)

    expected_matrix = np.loadtxt(
        data_directory.joinpath(f"expected_result_long_po_and_fo_duration.txt"), delimiter="\t"
    )
    assert np.array_equal(results.available_power, expected_matrix)
