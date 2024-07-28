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

import csv

import pytest

from antares.tsgen.cluster_import import import_thermal_cluster
from antares.tsgen.ts_generator import ThermalCluster, ThermalDataGenerator


@pytest.fixture
def cluster(data_directory) -> ThermalCluster:
    return import_thermal_cluster(data_directory / "cluster_40.csv")


def test_cluster(cluster, output_directory):
    ts_nb = 1

    generator = ThermalDataGenerator()
    results = generator.generate_time_series(cluster, ts_nb)

    tot_po = 0
    tot_fo = 0
    for i in range(365 * ts_nb):
        tot_po += results.planned_outages[i // 365][i % 365] * 2
        tot_fo += results.forced_outages[i // 365][i % 365] * 8
    true_por = tot_po / (365 * ts_nb)
    true_for = tot_fo / (365 * ts_nb)

    with open(output_directory / "cluster.csv", "w") as file:
        writer = csv.writer(file, delimiter=",", quotechar='"')

        writer.writerow(["timeseries :"])
        writer.writerows(
            [
                [line[i] for i in range(0, len(line), 24)]
                for line in results.available_power
            ]
        )

        writer.writerow(["total PO :", tot_po, "total FO :", tot_fo])
        writer.writerow(
            ["PO rate :", round(true_por, 4), "FO rate :", round(true_for, 4)]
        )
