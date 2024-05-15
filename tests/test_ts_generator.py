import csv
import pytest

from ts_generator import ThermalCluster, ProbilityLaw, ThermalDataGenerator


def test_one_unit_cluster(cluster_1, output_directory):
    ts_nb = 10

    generator = ThermalDataGenerator()
    output_series = [[0 for _ in range(365 * 24)] for __ in range(ts_nb)]
    output_outages = [[0 for _ in range(365)] for __ in range(ts_nb)]

    generator.generate_time_series(cluster_1, ts_nb, output_series, output_outages)

    tot_po = 0
    tot_fo = 0
    for i in range(365 * ts_nb):
        tot_po += output_outages[i // 365][i % 365][0] * 2
        tot_fo += output_outages[i // 365][i % 365][1] * 8
    true_por = tot_po / (365 * ts_nb)
    true_for = tot_fo / (365 * ts_nb)

    with open(output_directory / "test_one_unit_cluster.csv", "w") as file:
        writer = csv.writer(file, delimiter=',', quotechar='"')
        
        writer.writerow(["timeseries :"])
        writer.writerows([[line[i] for i in range(0, len(line), 24)] for line in output_series])
        
        writer.writerow(["outage duration ([POD, FOD]) :"])
        writer.writerows(output_outages)

        writer.writerow(["total PO :", tot_po, "total FO :", tot_fo])
        writer.writerow(["PO rate :", round(true_por, 4), "FO rate :", round(true_for, 4)])


def test_hundred_unit_cluster(cluster_100, output_directory):
    ts_nb = 50

    generator = ThermalDataGenerator()
    output_series = [[0 for _ in range(365 * 24)] for __ in range(ts_nb)]
    output_outages = [[0 for _ in range(365)] for __ in range(ts_nb)]

    generator.generate_time_series(cluster_100, ts_nb, output_series, output_outages)

    tot_po = 0
    tot_fo = 0
    for i in range(365 * ts_nb):
        tot_po += output_outages[i // 365][i % 365][0] * 2
        tot_fo += output_outages[i // 365][i % 365][1] * 8
    true_por = tot_po / (365 * ts_nb * cluster_100.unit_count)
    true_for = tot_fo / (365 * ts_nb * cluster_100.unit_count)

    #check the max PO
    tots_po = [[] for _ in range(ts_nb)]
    cursor = [0] * 10
    tot_po = 0
    for i in range(365 * ts_nb):
        po = output_outages[i // 365][i % 365][0]
        mo = output_outages[i // 365][i % 365][2]

        tot_po += po
        tot_po += mo
        tot_po -= cursor.pop(0)

        cursor.append(0)
        cursor[1] += po
        cursor[9] += mo

        if i > 10:
            assert tot_po <= cluster_100.npo_max[i % 365]
            assert tot_po >= cluster_100.npo_min[i % 365]

        tots_po[i // 365].append(tot_po)

    with open(output_directory / "test_100_unit_cluster.csv", "w") as file:
        writer = csv.writer(file, delimiter=',', quotechar='"')
        
        writer.writerow(["timeseries :"])
        writer.writerows([[line[i] for i in range(0, len(line), 24)] for line in output_series])
        
        writer.writerow(["outage duration ([POD, FOD]) :"])
        writer.writerows(output_outages)

        writer.writerow(["total PO :", tot_po, "total FO :", tot_fo])
        writer.writerow(["PO rate :", round(true_por, 4), "FO rate :", round(true_for, 4)])

        writer.writerow(["total simultaneous PO :"])
        writer.writerows(tots_po)
