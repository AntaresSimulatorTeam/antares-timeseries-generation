from pathlib import Path

import numpy as np

from ts_generator import ProbilityLaw, ThermalCluster


def import_thermal_cluster(path: Path, days_per_year: int = 365):
    law_dict = {"UNIFORM":ProbilityLaw.UNIFORM, "GEOMETRIC":ProbilityLaw.GEOMETRIC}
    array = np.genfromtxt(path, delimiter=',', dtype=str)
    return ThermalCluster(
        unit_count = int(array[1][1]),
        nominal_power = float(array[2][1]),
        modulation = [float(i) for i in array[3][1:24 + 1]],
        fo_law = law_dict[array[4][1]],
        fo_volatility = float(array[5][1]),
        po_law = law_dict[array[6][1]],
        po_volatility = float(array[7][1]),

        fo_duration = array[8][1:days_per_year + 1].astype(float),
        fo_rate = array[9][1:days_per_year + 1].astype(float),
        po_duration = array[10][1:days_per_year + 1].astype(float),
        po_rate = array[11][1:days_per_year + 1].astype(float),
        npo_min = array[12][1:days_per_year + 1].astype(float),
        npo_max = array[13][1:days_per_year + 1].astype(float),
    )