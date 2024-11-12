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

import numpy as np
import pytest

from antares.tsgen.cluster_import import import_thermal_cluster
from antares.tsgen.ts_generator import ThermalCluster, TimeseriesGenerator


@pytest.fixture
def cluster_1(data_directory) -> ThermalCluster:
    return import_thermal_cluster(data_directory / "cluster_1.csv")


@pytest.fixture
def cluster_100(data_directory) -> ThermalCluster:
    return import_thermal_cluster(data_directory / "cluster_100.csv")


@pytest.fixture
def cluster_high_por(data_directory) -> ThermalCluster:
    return import_thermal_cluster(data_directory / "cluster_high_por.csv")


def test_one_unit_cluster(cluster_1, output_directory):
    ts_nb = 4

    generator = TimeseriesGenerator()
    results = generator.generate_time_series_for_clusters(cluster_1, ts_nb)

    tot_po = 0
    tot_fo = 0
    for i in range(365 * ts_nb):
        tot_po += results.outage_output.planned_outages[i % 365][i // 365] * 2
        tot_fo += results.outage_output.forced_outages[i % 365][i // 365] * 8
    true_por = tot_po / (365 * ts_nb)
    true_for = tot_fo / (365 * ts_nb)

    with open(output_directory / "test_one_unit_cluster.csv", "w") as file:
        writer = csv.writer(file, delimiter=",", quotechar='"')

        writer.writerow(["timeseries :"])
        writer.writerows([[line[i] for i in range(0, len(line), 24)] for line in results.available_power])

        writer.writerow(["total PO :", tot_po, "total FO :", tot_fo])
        writer.writerow(["PO rate :", round(true_por, 4), "FO rate :", round(true_for, 4)])


def test_generation_with_fo_rate_at_1(cluster_1):
    # Put FO_rate at 1 shouldn't raise an issue
    cluster_1.outage_gen_params.fo_rate = np.ones(365)
    TimeseriesGenerator().generate_time_series_for_clusters(cluster_1, 1)

    # Reset
    cluster_1.outage_gen_params.fo_rate = np.zeros(365)

    # Put PO_rate at 1 shouldn't raise an issue
    cluster_1.outage_gen_params.po_rate = np.ones(365)
    TimeseriesGenerator().generate_time_series_for_clusters(cluster_1, 1)


def test_hundred_unit_cluster(cluster_100, output_directory):
    ts_nb = 50

    generator = TimeseriesGenerator()
    results = generator.generate_time_series_for_clusters(cluster_100, ts_nb)

    tot_po = 0
    tot_fo = 0
    for i in range(365 * ts_nb):
        tot_po += results.outage_output.planned_outages[i % 365][i // 365] * 2
        tot_fo += results.outage_output.forced_outages[i % 365][i // 365] * 8
    true_por = tot_po / (365 * ts_nb)
    true_for = tot_fo / (365 * ts_nb)

    # check the max PO
    tots_simult_po = [[] for _ in range(ts_nb)]
    cursor = [0] * 10
    tot_simult_po = 0
    for i in range(365 * ts_nb):
        po = results.outage_output.planned_outages[i % 365][i // 365]
        mo = results.outage_output.mixed_outages[i % 365][i // 365]

        tot_simult_po += po
        tot_simult_po += mo
        tot_simult_po -= cursor.pop(0)

        cursor.append(0)
        cursor[1] += po
        cursor[9] += mo

        if i > 10:
            assert tot_simult_po <= cluster_100.outage_gen_params.npo_max[i % 365]
            assert tot_simult_po >= cluster_100.outage_gen_params.npo_min[i % 365]

        tots_simult_po[i // 365].append(tot_simult_po)

    with open(output_directory / "test_100_unit_cluster.csv", "w") as file:
        writer = csv.writer(file, delimiter=",", quotechar='"')

        writer.writerow(["timeseries :"])
        writer.writerows([[line[i] for i in range(0, len(line), 24)] for line in results.available_power])

        writer.writerow(["total PO :", tot_po, "total FO :", tot_fo])
        writer.writerow(["PO rate :", round(true_por, 4), "FO rate :", round(true_for, 4)])

        writer.writerow(["total simultaneous PO :"])
        writer.writerows(tots_simult_po)


def test_max_po(cluster_high_por, output_directory):
    ts_nb = 4

    generator = TimeseriesGenerator()
    results = generator.generate_time_series_for_clusters(cluster_high_por, ts_nb)

    # check the max PO
    tots_simult_po = [[] for _ in range(ts_nb)]
    cursor = [0] * 10
    tot_simult_po = 0
    for i in range(365 * ts_nb):
        po = results.outage_output.planned_outages[i % 365][i // 365]
        mo = results.outage_output.mixed_outages[i % 365][i // 365]

        tot_simult_po += po
        tot_simult_po += mo
        tot_simult_po -= cursor.pop(0)

        cursor.append(0)
        cursor[1] += po
        cursor[9] += mo

        if i > 10:
            assert tot_simult_po <= cluster_high_por.outage_gen_params.npo_max[i % 365]
            assert tot_simult_po >= cluster_high_por.outage_gen_params.npo_min[i % 365]

        tots_simult_po[i // 365].append(tot_simult_po)

    with open(output_directory / "test_high_por_cluster.csv", "w") as file:
        writer = csv.writer(file, delimiter=",", quotechar='"')

        writer.writerow(["timeseries :"])
        writer.writerows([[line[i] for i in range(0, len(line), 24)] for line in results.available_power])

        writer.writerow(["total simultaneous PO :"])
        writer.writerows(tots_simult_po)
