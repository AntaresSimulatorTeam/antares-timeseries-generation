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
import argparse
from pathlib import Path
from typing import List, Optional


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--study", type=Path, help="path to the root directory of the study"
    )
    parser.add_argument(
        "--models", nargs="+", type=Path, help="list of path to model file, *.yml"
    )
    parser.add_argument(
        "--component", type=Path, help="path to the component file, *.yml"
    )
    parser.add_argument(
        "--timeseries", type=Path, help="path to the timeseries directory"
    )
    parser.add_argument(
        "--duration", type=int, help="duration of the simulation", default=1
    )
    parser.add_argument(
        "--scenario", type=int, help="number of scenario of the simulation", default=1
    )

    args = parser.parse_args()


if __name__ == "__main__":
    main()
