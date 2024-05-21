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

from dataclasses import dataclass
from enum import Enum
from math import log, sqrt
from random import random
from typing import Any, List, Optional, Tuple

import numpy as np

# probabilities above FAILURE_RATE_EQ_1 are considered certain (equal to 1)
FAILURE_RATE_EQ_1 = 0.999


class rndgenerator:
    @classmethod
    def next(self) -> float:
        return random()


class ProbilityLaw(Enum):
    UNIFORM = "UNIFORM"
    GEOMETRIC = "GEOMETRIC"


@dataclass
class ThermalCluster:
    # available units of the cluster
    unit_count: int
    # nominal power
    nominal_power: float
    # modulation of the nominal power for a certain hour in the day (between 0 and 1)
    modulation: List[float]  ### maybe group nominal_power and modulation in one vaiable

    # forced and planed outage parameters
    # indexed by day of the year
    fo_duration: List[int]
    fo_rate: List[float]
    po_duration: List[int]
    po_rate: List[float]
    npo_min: List[int]  # number of planed outage min in a day
    npo_max: List[int]  # number of planed outage max in a day

    # forced and planed outage probability law and volatility
    # volatility characterizes the distance from the expect at which the value drawn can be
    fo_law: ProbilityLaw
    fo_volatility: float  ### maybe create an object to store law and volatility
    po_law: ProbilityLaw
    po_volatility: float


class ThermalDataGenerator:
    def __init__(self, days_per_year: int = 365) -> None:
        self.days_per_year = days_per_year

        ## explain that ??
        self.log_size = 4000  # >= 5 * (max(df) + max(dp))
        # the number of starting (if positive)/ stopping (if negative) units (due to FO and PO) at a given time
        self.LOG = [0] * self.log_size
        # same but only for PO; necessary to ensure maximum and minimum PO is respected
        self.LOGP = [0] * self.log_size

        # pogramed and forced outage rate
        self.lf = np.empty(days_per_year, dtype=float)
        self.lp = np.empty(days_per_year, dtype=float)

        ## ???
        self.ff = np.empty(days_per_year, dtype=float)  # ff = lf / (1 - lf)
        self.pp = np.empty(days_per_year, dtype=float)  # pp = lp / (1 - lp)

        # precalculated value to speed up generation of random outage duration
        self.af = np.empty(days_per_year, dtype=float)
        self.bf = np.empty(days_per_year, dtype=float)
        self.ap = np.empty(days_per_year, dtype=float)
        self.bp = np.empty(days_per_year, dtype=float)

    def prepare_outage_duration_constant(
        self,
        law: ProbilityLaw,
        volatility: float,
        A: np.ndarray[Any, np.dtype[Any]],
        B: np.ndarray[Any, np.dtype[Any]],
        expecs: List[int],
    ) -> None:
        """
        precalculation of constant values use in generation of outage duration
        results are stored in A and B
        """
        if law == ProbilityLaw.UNIFORM:
            for day in range(self.days_per_year):
                D = expecs[day]
                xtemp = volatility * (D - 1)
                A[day] = D - xtemp
                B[day] = 2 * xtemp + 1
        elif law == ProbilityLaw.GEOMETRIC:
            for day in range(self.days_per_year):
                D = expecs[day]
                xtemp = volatility * volatility * D * (D - 1)
                if xtemp != 0:
                    ytemp = (sqrt(4 * xtemp + 1) - 1) / (2 * xtemp)
                    A[day] = D - 1 / ytemp
                    B[day] = 1 / log(1 - ytemp)
                else:
                    A[day] = D - 1
                    B[day] = 0

    def duration_generator(
        self, law: ProbilityLaw, volatility: float, a: float, b: float, expec: int
    ) -> int:
        """
        generation of random outage duration
        """
        rnd_nb = rndgenerator.next()
        if law == ProbilityLaw.UNIFORM:
            return int(a + rnd_nb * b)
        elif law == ProbilityLaw.GEOMETRIC:
            return min(int(1 + a + b * log(rnd_nb)), int(self.log_size / 2 - 1))

    def generate_time_series(
        self,
        cluster: ThermalCluster,
        number_of_timeseries: int,
        output_series: List[List[float]],
        output_outages: Optional[List[List[Tuple[int, int, int, int, int]]]] = None,
    ) -> None:
        """
        generation of multiple timeseries for a given thermal cluster
        output_series stores available power at a given time
        output_outages stores infomation about outages begining at a given time -> (PPO, PFO, MXO, POD, FOD)
        """

        # --- precalculation ---
        # cached values for (1-lf)**k and (1-lp)**k
        self.FPOW: List[List[float]] = []
        self.PPOW: List[List[float]] = []

        for day in range(self.days_per_year):
            # lf and lp represent the forced and programed failure rate
            # failure rate means the probability to enter in outage each day
            # its value is given by: OR / [OR + OD * (1 - OR)]
            FOR = cluster.fo_rate[day]
            FOD = cluster.fo_duration[day]
            self.lf[day] = FOR / (FOR + FOD * (1 - FOR))

            POR = cluster.po_rate[day]
            POD = cluster.po_duration[day]
            self.lp[day] = POR / (POR + POD * (1 - POR))

            if self.lf[day] < 0:
                raise ValueError(f"forced failure rate is negative on day {day}")
            if self.lp[day] < 0:
                raise ValueError(f"programed failure rate is negative on day {day}")

            ## i dont understand what these calulations are for
            ## consequently reduce the lower failure rate
            if self.lf[day] < self.lp[day]:
                self.lf[day] *= (1 - self.lp[day]) / (1 - self.lf[day])
            if self.lp[day] < self.lf[day]:
                self.lp[day] *= (1 - self.lf[day]) / (1 - self.lp[day])

            a = 0
            b = 0
            if self.lf[day] <= FAILURE_RATE_EQ_1:
                a = 1 - self.lf[day]
                self.ff[day] = self.lf[day] / a
            if self.lp[day] <= FAILURE_RATE_EQ_1:
                b = 1 - self.lp[day]
                self.pp[day] = self.lp[day] / b

            # pre calculating power values
            self.FPOW.append([])
            self.PPOW.append([])
            for k in range(cluster.unit_count + 1):
                self.FPOW[-1].append(pow(a, k))
                self.PPOW[-1].append(pow(b, k))

        self.prepare_outage_duration_constant(
            cluster.fo_law, cluster.fo_volatility, self.af, self.bf, cluster.fo_duration
        )
        self.prepare_outage_duration_constant(
            cluster.po_law, cluster.po_volatility, self.ap, self.bp, cluster.po_duration
        )

        # --- calculation ---
        # the two first generated time series will be dropped, necessary to make system stable and physically coherent
        # as a consequence, N + 2 time series will be computed

        # mixed, pure planned and pure force outage
        MXO = 0
        PFO = 0
        PPO = 0

        # dates
        now = 0
        fut = 0

        # current number of PO and AU (avlaible units)
        cur_nb_PO = 0
        cur_nb_AU = cluster.unit_count
        # stock is a way to keep the number of PO pushed back due to PO max / antcipated due to PO min
        # stock > 0 number of PO pushed back, stock < 0 number of PO antcipated
        stock = 0

        for ts_index in range(number_of_timeseries + 2):
            # hour in the year
            hour = 0
            if ts_index > 1:
                if output_outages:
                    output_outage = output_outages[ts_index - 2]
                output_serie = output_series[ts_index - 2]

            for day in range(self.days_per_year):
                if cur_nb_AU > 100:  ## it was like that in c++, i'm not shure why
                    # maybe it's for the pow (FPOW, PPOW) calculation, if so it might not be the right place to do
                    raise ValueError("avalaible unit number out of bound (> 100)")

                # = return of units wich were in outage =
                cur_nb_PO -= self.LOGP[now]
                self.LOGP[
                    now
                ] = 0  # set to 0 because this cell will be use again later (in self.log_size days)
                cur_nb_AU += self.LOG[now]
                self.LOG[now] = 0

                # = determinating units that go on outage =
                # FO and PO canditate
                FOC = 0
                POC = 0

                if self.lf[day] > 0 and self.lf[day] <= FAILURE_RATE_EQ_1:
                    A = rndgenerator.next()
                    last = self.FPOW[day][cur_nb_AU]
                    if A > last:
                        cumul = last
                        for d in range(1, cur_nb_AU + 1):
                            last = last * self.ff[day] * (cur_nb_AU + 1 - d) / d
                            cumul += last
                            FOC = d
                            if A <= cumul:
                                break
                elif self.lf[day] > FAILURE_RATE_EQ_1:
                    FOC = cur_nb_AU
                else:  # self.lf[day] == 0
                    FOC = 0

                if self.lp[day] > 0 and self.lp[day] <= FAILURE_RATE_EQ_1:
                    # apparent number of available units
                    AUN_app = cur_nb_AU
                    if stock >= 0 and stock <= cur_nb_AU:
                        AUN_app -= stock
                    elif stock > cur_nb_AU:
                        AUN_app = 0

                    A = rndgenerator.next()
                    last = self.PPOW[day][cur_nb_AU]
                    if A > last:
                        cumul = last
                        for d in range(1, cur_nb_AU + 1):
                            last = last * self.pp[day] * (cur_nb_AU + 1 - d) / d
                            cumul += last
                            POC = d
                            if A <= cumul:
                                break
                elif self.lp[day] > FAILURE_RATE_EQ_1:
                    POC = cur_nb_AU
                else:  # self.lf[day] == 0
                    POC = 0

                # apparent PO is compared to cur_nb_AU, considering stock
                candidate = POC + stock
                if 0 <= candidate and candidate <= cur_nb_AU:
                    POC = candidate
                    stock = 0
                if candidate > cur_nb_AU:
                    POC = cur_nb_AU
                    stock = candidate - cur_nb_AU
                if candidate < 0:
                    POC = 0
                    stock = candidate

                # = checking min and max PO =
                if POC + cur_nb_PO > cluster.npo_max[day]:
                    # too many PO to place
                    # the excedent is placed in stock
                    POC = cluster.npo_max[day] - cur_nb_PO
                    cur_nb_PO += POC
                elif POC + cur_nb_PO < cluster.npo_min[day]:
                    if cluster.npo_min[day] - cur_nb_PO > cur_nb_AU:
                        stock -= cur_nb_AU - POC
                        POC = cur_nb_AU
                        cur_nb_PO += POC
                    else:
                        stock -= cluster.npo_min[day] - (POC + cur_nb_PO)
                        POC = cluster.npo_min[day] - cur_nb_PO
                        cur_nb_PO += POC
                else:
                    cur_nb_PO += POC

                # = distributing outage in category =
                # pure planed, pure forced, mixed
                if cluster.unit_count == 1:
                    if POC == 1 and FOC == 1:
                        MXO = 1
                        PPO = 0
                        PFO = 0
                    else:
                        MXO = 0
                        PPO = int(POC)
                        PFO = int(FOC)
                else:
                    if cur_nb_AU != 0:
                        MXO = int(POC * FOC // cur_nb_AU)
                        PPO = int(POC - MXO)
                        PFO = int(FOC - MXO)
                    else:
                        MXO = 0
                        PPO = 0
                        PFO = 0

                # = units stopping =
                cur_nb_AU -= PPO + PFO + MXO

                # = generating outage duration = (from the law)
                true_POD = 0
                true_FOD = 0

                if PPO != 0 or MXO != 0:
                    true_POD = self.duration_generator(
                        cluster.po_law,
                        cluster.po_volatility,
                        self.ap[day],
                        self.bp[day],
                        cluster.po_duration[day],
                    )
                if PFO != 0 or MXO != 0:
                    true_FOD = self.duration_generator(
                        cluster.fo_law,
                        cluster.fo_volatility,
                        self.af[day],
                        self.bf[day],
                        cluster.fo_duration[day],
                    )

                if PPO != 0:
                    fut = (now + true_POD) % self.log_size
                    self.LOG[fut] += PPO
                    self.LOGP[fut] += PPO
                if PFO != 0:
                    fut = (now + true_FOD) % self.log_size
                    self.LOG[fut] += PFO
                if MXO != 0:
                    fut = (now + true_POD + true_FOD) % self.log_size
                    self.LOG[fut] += MXO
                    self.LOGP[fut] += MXO

                # = storing output in output arrays =
                if ts_index > 1:  # drop the 2 first generated timeseries
                    if output_outages:
                        output_outage[day] = (PPO, PFO, MXO, true_POD, true_FOD)
                    for h in range(24):
                        output_serie[hour] = (
                            cur_nb_AU * cluster.nominal_power * cluster.modulation[h]
                        )
                        hour += 1

                now = (now + 1) % self.log_size
