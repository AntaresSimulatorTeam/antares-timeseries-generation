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

from antares.tsgen.ts_generator import (
    ProbabilityLaw,
    ThermalCluster,
    ThermalDataGenerator,
)


def test_forced_outages(rng):
    days = 365
    cluster = ThermalCluster(
        unit_count=10,
        nominal_power=100,
        modulation=np.ones(dtype=float, shape=24),
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
    npt.assert_equal(results.forced_outages[0][:6], [0, 0, 0, 0, 2, 0])
    npt.assert_equal(results.forced_outage_durations[0][:6], [0, 0, 0, 0, 10, 0])
    # No planned outage
    npt.assert_equal(results.planned_outages[0], np.zeros(365))
    npt.assert_equal(results.planned_outage_durations[0], np.zeros(365))

    npt.assert_equal(results.available_units[0][:5], [9, 9, 9, 9, 8])
    # Check available power consistency with available units and modulation
    assert results.available_power[0][0] == 900
    assert results.available_power[0][12] == 450  # Modulation is 0.5 for hour 12
    assert results.available_power[0][4 * 24] == 800


def test_planned_outages(rng):
    days = 365
    cluster = ThermalCluster(
        unit_count=10,
        nominal_power=100,
        modulation=np.ones(dtype=float, shape=24),
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
    npt.assert_equal(results.forced_outages[0], np.zeros(365))
    npt.assert_equal(results.forced_outage_durations[0], np.zeros(365))
    # No planned outage
    npt.assert_equal(results.planned_outages[0][:6], [0, 0, 0, 0, 2, 0])
    npt.assert_equal(results.available_units[0][:5], [9, 9, 9, 9, 8])
    # Check available power consistency with available units and modulation
    assert results.available_power[0][0] == 900
    assert results.available_power[0][12] == 450  # Modulation is 0.5 for hour 12
    assert results.available_power[0][4 * 24] == 800


def test_planned_outages_limitation(rng):
    days = 365
    # Maximum 1 planned outage at a time.
    cluster = ThermalCluster(
        unit_count=10,
        nominal_power=100,
        modulation=np.ones(dtype=float, shape=24),
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
    npt.assert_equal(results.forced_outages[0], np.zeros(365))
    npt.assert_equal(results.forced_outage_durations[0], np.zeros(365))
    # Maxmimum one planned outage at a time
    npt.assert_equal(results.planned_outages[0][:6], [0, 1, 0, 1, 0, 1])
    npt.assert_equal(results.planned_outage_durations[0][:6], [0, 2, 0, 2, 0, 2])
    npt.assert_equal(results.available_units[0][:5], [9, 9, 9, 9, 9])
    # Check available power consistency with available units and modulation
    assert results.available_power[0][0] == 900
    assert results.available_power[0][4 * 24] == 900


def test_planned_outages_min_limitation(rng):
    days = 365
    # Minimum 2 planned outages at a time
    cluster = ThermalCluster(
        unit_count=10,
        nominal_power=100,
        modulation=np.ones(dtype=float, shape=24),
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
    npt.assert_equal(results.forced_outages[0], np.zeros(365))
    npt.assert_equal(results.forced_outage_durations[0], np.zeros(365))
    # Maxmimum one planned outage at a time
    npt.assert_equal(results.planned_outages[0][:6], [0, 0, 1, 0, 0, 1])
    npt.assert_equal(results.planned_outage_durations[0][:6], [0, 0, 10, 0, 0, 10])
    npt.assert_equal(results.available_units[0][:5], [8, 8, 8, 8, 8])
    # Check available power consistency with available units and modulation
    assert results.available_power[0][0] == 800
    assert results.available_power[0][4 * 24] == 800
