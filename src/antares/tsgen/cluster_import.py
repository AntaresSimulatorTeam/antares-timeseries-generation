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

from pathlib import Path

import numpy as np

from .ts_generator import ProbabilityLaw, ThermalCluster


def import_thermal_cluster(path: Path, days_per_year: int = 365) -> ThermalCluster:
    law_dict = {
        "UNIFORM": ProbabilityLaw.UNIFORM,
        "GEOMETRIC": ProbabilityLaw.GEOMETRIC,
    }
    array = np.genfromtxt(path, delimiter=",", dtype=str)
    return ThermalCluster(
        unit_count=int(array[1][1]),
        nominal_power=float(array[2][1]),
        modulation=array[3][1 : 24 + 1].astype(int),
        fo_law=law_dict[array[4][1]],
        fo_volatility=float(array[5][1]),
        po_law=law_dict[array[6][1]],
        po_volatility=float(array[7][1]),
        fo_duration=array[8][1 : days_per_year + 1].astype(int),
        fo_rate=array[9][1 : days_per_year + 1].astype(float),
        po_duration=array[10][1 : days_per_year + 1].astype(int),
        po_rate=array[11][1 : days_per_year + 1].astype(float),
        npo_min=array[12][1 : days_per_year + 1].astype(int),
        npo_max=array[13][1 : days_per_year + 1].astype(int),
    )
