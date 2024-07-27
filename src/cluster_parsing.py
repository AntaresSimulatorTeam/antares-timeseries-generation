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
from typing import Iterable, Optional, TextIO, Tuple

import pandas as pd
from pydantic import BaseModel, Field
from yaml import safe_load


def _to_kebab(snake: str) -> str:
    return snake.replace("_", "-")


class InputCluster(BaseModel):
    unit_count: int
    nominal_power: float
    fo_law: str
    fo_volatility: float
    po_law: str
    po_volatility: float

    class Config:
        alias_generator = _to_kebab
        extra = "forbid"


def parse_yaml_cluster(input_cluster: TextIO) -> InputCluster:
    tree = safe_load(input_cluster)
    return InputCluster.model_validate(tree["cluster"])


def parse_cluster_ts(
    file: Path, shape: Optional[Tuple[int, int]] = None
) -> pd.core.frame.DataFrame:
    ts = pd.read_csv(file)
    assert shape is None or ts.shape == shape
    return ts
