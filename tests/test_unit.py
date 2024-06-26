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

import pytest

from ts_generator import ProbilityLaw, ThermalCluster, ThermalDataGenerator

NB_OF_DAY = 10


@pytest.fixture
def cluster() -> ThermalCluster:
    return ThermalCluster(
        unit_count=1,
        nominal_power=500,
        modulation=[1 for i in range(24)],
        fo_law=ProbilityLaw.UNIFORM,
        fo_volatility=0,
        po_law=ProbilityLaw.UNIFORM,
        po_volatility=0,
        fo_duration=[2 for i in range(NB_OF_DAY)],
        fo_rate=[0.2 for i in range(NB_OF_DAY)],
        po_duration=[1 for i in range(NB_OF_DAY)],
        po_rate=[0.1 for i in range(NB_OF_DAY)],
        npo_min=[0 for i in range(NB_OF_DAY)],
        npo_max=[1 for i in range(NB_OF_DAY)],
    )


def test_ts_value(cluster):
    ts_nb = 4

    generator = ThermalDataGenerator(days_per_year=NB_OF_DAY)
    results = generator.generate_time_series(cluster, ts_nb)

    assert results.available_power.shape == (ts_nb, NB_OF_DAY * 24)
    assert results.nb_ppo.shape == (ts_nb, NB_OF_DAY)
    assert results.nb_pfo.shape == (ts_nb, NB_OF_DAY)
    assert results.nb_mxo.shape == (ts_nb, NB_OF_DAY)
    assert results.pod.shape == (ts_nb, NB_OF_DAY)
    assert results.fod.shape == (ts_nb, NB_OF_DAY)

    for l in range(ts_nb):
        for c in range(NB_OF_DAY * 24):
            assert results.available_power[l][c] % 500 == 0


def test_ts_value_with_modulation(cluster):
    modulation = [(i % 10) * 0.1 for i in range(24)]

    cluster.modulation = modulation

    ts_nb = 4

    generator = ThermalDataGenerator(days_per_year=NB_OF_DAY)
    results = generator.generate_time_series(cluster, ts_nb)

    assert results.available_power.shape == (ts_nb, NB_OF_DAY * 24)
    assert results.nb_ppo.shape == (ts_nb, NB_OF_DAY)
    assert results.nb_pfo.shape == (ts_nb, NB_OF_DAY)
    assert results.nb_mxo.shape == (ts_nb, NB_OF_DAY)
    assert results.pod.shape == (ts_nb, NB_OF_DAY)
    assert results.fod.shape == (ts_nb, NB_OF_DAY)

    for l in range(ts_nb):
        for d in range(NB_OF_DAY):
            if modulation[d] != 0:
                assert results.available_power[l][d * 24] % (500 * modulation[d]) == 0
            else:
                assert results.available_power[l][d * 24] == 0
