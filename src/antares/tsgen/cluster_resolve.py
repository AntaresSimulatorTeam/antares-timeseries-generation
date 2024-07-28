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

import pandas as pd

from .cluster_parsing import InputCluster
from .ts_generator import ProbabilityLaw, ThermalCluster


def resolve_thermal_cluster(
    parsed_yaml: InputCluster,
    parameters_ts: pd.core.frame.DataFrame,
    modulation: pd.core.frame.DataFrame,
) -> ThermalCluster:
    law_dict = {
        "UNIFORM": ProbabilityLaw.UNIFORM,
        "GEOMETRIC": ProbabilityLaw.GEOMETRIC,
    }
    return ThermalCluster(
        unit_count=parsed_yaml.unit_count,
        nominal_power=parsed_yaml.nominal_power,
        modulation=modulation["modulation"].to_numpy(dtype=float),
        fo_law=law_dict[parsed_yaml.fo_law],
        fo_volatility=parsed_yaml.fo_volatility,
        po_law=law_dict[parsed_yaml.po_law],
        po_volatility=parsed_yaml.po_volatility,
        fo_duration=parameters_ts["FOD"].to_numpy(dtype=int),
        fo_rate=parameters_ts["FOR"].to_numpy(dtype=float),
        po_duration=parameters_ts["POD"].to_numpy(dtype=int),
        po_rate=parameters_ts["POR"].to_numpy(dtype=float),
        npo_min=parameters_ts["POMax"].to_numpy(dtype=int),
        npo_max=parameters_ts["POMin"].to_numpy(dtype=int),
    )
