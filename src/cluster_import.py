import csv
from pathlib import Path

from ts_generator import ThermalCluster, ProbilityLaw

def from_csv(path: Path, days_per_year: int = 365):
    law_dict = {"UNIFORM":ProbilityLaw.UNIFORM, "GEOMETRIC":ProbilityLaw.GEOMETRIC}
    with open(path, "r") as csvfile:
        reader = list(csv.reader(csvfile, delimiter=',', quotechar='"'))
        cluster = ThermalCluster(
            unit_count = int(reader[1][1]),
            nominal_power = float(reader[2][1]),
            modulation = [float(i) for i in reader[3][1:24 + 1]],
            
            fo_law = law_dict[reader[4][1]],
            fo_volatility = float(reader[5][1]),
            po_law = law_dict[reader[6][1]],
            po_volatility = float(reader[7][1]),

            fo_duration = [float(i) for i in reader[8][1:days_per_year + 1]],
            fo_rate = [float(i) for i in reader[9][1:days_per_year + 1]],
            po_duration = [float(i) for i in reader[10][1:days_per_year + 1]],
            po_rate = [float(i) for i in reader[11][1:days_per_year + 1]],
            npo_min = [float(i) for i in reader[12][1:days_per_year + 1]],
            npo_max = [float(i) for i in reader[13][1:days_per_year + 1]],
        )
    return cluster
