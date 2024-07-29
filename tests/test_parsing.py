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

from antares.tsgen.cluster_parsing import parse_cluster_ts, parse_yaml_cluster
from antares.tsgen.cluster_resolve import resolve_thermal_cluster
from antares.tsgen.ts_generator import ProbabilityLaw, ThermalCluster

NB_OF_DAY = 10


@pytest.fixture
def cluster() -> ThermalCluster:
    return ThermalCluster(
        unit_count=1,
        nominal_power=500,
        modulation=np.ones(dtype=float, shape=24),
        fo_law=ProbabilityLaw.UNIFORM,
        fo_volatility=0,
        po_law=ProbabilityLaw.UNIFORM,
        po_volatility=0,
        fo_duration=np.ones(dtype=int, shape=NB_OF_DAY) * 2,
        fo_rate=np.ones(dtype=float, shape=NB_OF_DAY) * 0.2,
        po_duration=np.ones(dtype=int, shape=NB_OF_DAY),
        po_rate=np.ones(dtype=float, shape=NB_OF_DAY) * 0.1,
        npo_min=np.zeros(dtype=int, shape=NB_OF_DAY),
        npo_max=np.ones(dtype=int, shape=NB_OF_DAY),
    )


def test(cluster, data_directory):
    with (data_directory / "cluster.yaml").open("r") as file:
        yaml_cluster = parse_yaml_cluster(file)
    ts_param = parse_cluster_ts(data_directory / "ts_param.csv")
    modulation = parse_cluster_ts(data_directory / "modulation.csv")
    cld = resolve_thermal_cluster(yaml_cluster, ts_param, modulation)

    assert cld.unit_count == cluster.unit_count
    assert cld.nominal_power == cluster.nominal_power
    npt.assert_equal(cld.modulation, cluster.modulation)
    assert cld.fo_law == cluster.fo_law
    assert cld.fo_volatility == cluster.fo_volatility
    assert cld.po_law == cluster.po_law
    assert cld.po_volatility == cluster.po_volatility
    npt.assert_equal(cld.fo_duration, cluster.fo_duration)
    npt.assert_equal(cld.fo_rate, cluster.fo_rate)
    npt.assert_equal(cld.po_duration, cluster.po_duration)
    npt.assert_equal(cld.po_rate, cluster.po_rate)
    npt.assert_equal(cld.npo_min, cluster.npo_min)
    npt.assert_equal(cld.npo_max, cluster.npo_max)
