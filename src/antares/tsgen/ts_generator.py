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
from typing import List

import numpy as np
import numpy.typing as npt

from antares.tsgen.duration_generator import ProbabilityLaw, make_duration_generator
from antares.tsgen.random_generator import RNG, MersenneTwisterRNG

# probabilities above FAILURE_RATE_EQ_1 are considered certain (equal to 1)
FAILURE_RATE_EQ_1 = 0.999


@dataclass
class ThermalCluster:
    # available units of the cluster
    unit_count: int
    # nominal power
    nominal_power: float
    # modulation of the nominal power for a certain hour in the day (between 0 and 1)
    modulation: npt.NDArray[np.int_]

    # forced and planed outage parameters
    # indexed by day of the year
    fo_duration: npt.NDArray[np.int_]
    fo_rate: npt.NDArray[np.float_]
    po_duration: npt.NDArray[np.int_]
    po_rate: npt.NDArray[np.float_]
    npo_min: npt.NDArray[np.int_]  # number of planed outage min in a day
    npo_max: npt.NDArray[np.int_]  # number of planed outage max in a day

    # forced and planed outage probability law and volatility
    # volatility characterizes the distance from the expect at which the value drawn can be
    fo_law: ProbabilityLaw
    fo_volatility: float
    po_law: ProbabilityLaw
    po_volatility: float


class OutputTimeseries:
    def __init__(self, ts_count: int, days: int) -> None:
        self.available_units = np.zeros(shape=(ts_count, days), dtype=int)
        # available power each hours
        self.available_power = np.zeros((ts_count, 24 * days), dtype=float)
        # number of pure planed, pure forced and mixed outage each day
        self.planned_outages = np.zeros((ts_count, days), dtype=int)
        self.forced_outages = np.zeros((ts_count, days), dtype=int)
        self.mixed_outages = np.zeros((ts_count, days), dtype=int)
        # number of pure planed and pure forced outage duration each day
        # (mixed outage duration = pod + fod)
        self.planned_outage_durations = np.zeros((ts_count, days), dtype=int)
        self.forced_outage_durations = np.zeros((ts_count, days), dtype=int)


def _column_powers(column: npt.NDArray[np.float_], width: int) -> npt.NDArray:
    """
    Returns a matrix of given width where column[i] is the ith power of the input column.
    """
    powers = np.arange(width)
    powers.shape = (1, len(powers))
    column.shape = (len(column), 1)
    return pow(column, powers)


class ThermalDataGenerator:
    def __init__(self, rng: RNG = MersenneTwisterRNG(), days: int = 365) -> None:
        self.rng = rng
        self.days = days

    def generate_time_series(
        self,
        cluster: ThermalCluster,
        number_of_timeseries: int,
    ) -> OutputTimeseries:
        """
        generation of multiple timeseries for a given thermal cluster
        """

        # TODO: Remove this log size limit, seems useless and error prone if very large durations
        log_size = 4000  # >= 5 * (max(df) + max(dp))
        # the number of starting (if positive)/ stopping (if negative) units (due to FO and PO) at a given time
        log = np.zeros(log_size, dtype=int)
        # same but only for PO; necessary to ensure maximum and minimum PO is respected
        logp = np.zeros(log_size, dtype=int)

        # pogramed and forced outage rate
        daily_fo_rate = np.zeros(self.days, dtype=float)
        daily_po_rate = np.zeros(self.days, dtype=float)

        ## ???
        ff = np.zeros(self.days, dtype=float)  # ff = lf / (1 - lf)
        pp = np.zeros(self.days, dtype=float)  # pp = lp / (1 - lp)

        # --- precalculation ---
        # cached values for (1-lf)**k and (1-lp)**k
        fpow = np.zeros(shape=(self.days, cluster.unit_count + 1))
        ppow = np.zeros(shape=(self.days, cluster.unit_count + 1))

        # lf and lp represent the forced and programed failure rate
        # failure rate means the probability to enter in outage each day
        # its value is given by: OR / [OR + OD * (1 - OR)]
        daily_fo_rate = cluster.fo_rate / (
            cluster.fo_rate + cluster.fo_duration * (1 - cluster.fo_rate)
        )
        daily_po_rate = cluster.po_rate / (
            cluster.po_rate + cluster.po_duration * (1 - cluster.po_rate)
        )

        invalid_days = daily_fo_rate < 0
        if invalid_days.any():
            raise ValueError(
                f"forced failure rate is negative on days {invalid_days.nonzero()[0].tolist()}"
            )
        invalid_days = daily_po_rate < 0
        if invalid_days.any():
            raise ValueError(
                f"planned failure rate is negative on days {invalid_days.nonzero()[0].tolist()}"
            )

        ## i dont understand what these calulations are for
        ## consequently reduce the lower failure rate
        mask = daily_fo_rate < daily_po_rate
        daily_fo_rate[mask] *= (1 - daily_po_rate[mask]) / (1 - daily_fo_rate[mask])
        mask = daily_po_rate < daily_fo_rate
        daily_po_rate[mask] *= (1 - daily_fo_rate[mask]) / (1 - daily_po_rate[mask])

        a = np.zeros(shape=self.days, dtype=float)
        b = np.zeros(shape=self.days, dtype=float)
        mask = daily_fo_rate <= FAILURE_RATE_EQ_1
        a[mask] = 1 - daily_fo_rate[mask]
        ff[mask] = daily_fo_rate[mask] / a

        mask = daily_po_rate <= FAILURE_RATE_EQ_1
        b[mask] = 1 - daily_po_rate[mask]
        pp[mask] = daily_po_rate[mask] / b

        fpow = _column_powers(a, cluster.unit_count + 1)
        ppow = _column_powers(b, cluster.unit_count + 1)

        fod_generator = make_duration_generator(
            self.rng, cluster.fo_law, cluster.fo_volatility, cluster.fo_duration
        )
        pod_generator = make_duration_generator(
            self.rng, cluster.po_law, cluster.po_volatility, cluster.po_duration
        )

        # --- calculation ---
        # the two first generated time series will be dropped, necessary to make system stable and physically coherent
        # as a consequence, N + 2 time series will be computed

        # output that will be returned
        output = OutputTimeseries(number_of_timeseries, self.days)

        # dates
        now = 0

        # current number of PO and AU (avlaible units)
        current_planned_outages = 0
        current_available_units = cluster.unit_count
        # stock is a way to keep the number of PO pushed back due to PO max / antcipated due to PO min
        # stock > 0 number of PO pushed back, stock < 0 number of PO antcipated
        stock = 0

        for ts_index in range(-2, number_of_timeseries):
            for day in range(self.days):
                # = return of units wich were in outage =
                current_planned_outages -= logp[now]
                logp[
                    now
                ] = 0  # set to 0 because this cell will be use again later (in self.log_size days)
                current_available_units += log[now]
                log[now] = 0

                fo_candidates = 0
                po_candidates = 0

                if daily_fo_rate[day] > 0 and daily_fo_rate[day] <= FAILURE_RATE_EQ_1:
                    draw = self.rng.next()
                    last = fpow[day, current_available_units]
                    if draw > last:
                        cumul = last
                        for d in range(1, current_available_units + 1):
                            last = (
                                last * ff[day] * (current_available_units + 1 - d) / d
                            )
                            cumul += last
                            fo_candidates = d
                            if draw <= cumul:
                                break
                elif (
                    daily_fo_rate[day] > FAILURE_RATE_EQ_1
                ):  # TODO: not same comparison as cpp ?
                    fo_candidates = current_available_units
                else:  # self.lf[day] == 0
                    fo_candidates = 0

                if daily_po_rate[day] > 0 and daily_po_rate[day] <= FAILURE_RATE_EQ_1:
                    apparent_available_units = current_available_units
                    if stock >= 0 and stock <= current_available_units:
                        apparent_available_units -= stock
                    elif stock > current_available_units:
                        apparent_available_units = 0

                    draw = self.rng.next()
                    last = ppow[day, apparent_available_units]
                    if draw > last:
                        cumul = last
                        for d in range(1, apparent_available_units + 1):
                            last = (
                                last * pp[day] * (apparent_available_units + 1 - d) / d
                            )
                            cumul += last
                            po_candidates = d
                            if draw <= cumul:
                                break
                elif daily_po_rate[day] > FAILURE_RATE_EQ_1:
                    po_candidates = current_available_units
                else:  # self.lf[day] == 0
                    po_candidates = 0

                # apparent PO is compared to cur_nb_AU, considering stock
                candidate = po_candidates + stock
                if 0 <= candidate and candidate <= current_available_units:
                    po_candidates = candidate
                    stock = 0
                if candidate > current_available_units:
                    po_candidates = current_available_units
                    stock = candidate - current_available_units
                if candidate < 0:
                    po_candidates = 0
                    stock = candidate

                # = checking min and max PO =
                if po_candidates + current_planned_outages > cluster.npo_max[day]:
                    # too many PO to place
                    # the excedent is placed in stock
                    po_candidates = cluster.npo_max[day] - current_planned_outages
                    current_planned_outages += po_candidates
                elif po_candidates + current_planned_outages < cluster.npo_min[day]:
                    if (
                        cluster.npo_min[day] - current_planned_outages
                        > current_available_units
                    ):
                        stock -= current_available_units - po_candidates
                        po_candidates = current_available_units
                        current_planned_outages += po_candidates
                    else:
                        stock -= cluster.npo_min[day] - (
                            po_candidates + current_planned_outages
                        )
                        po_candidates = cluster.npo_min[day] - current_planned_outages
                        current_planned_outages += po_candidates
                else:
                    current_planned_outages += po_candidates

                # = distributing outage in category =
                # pure planed, pure forced, mixed
                mixed_outages = 0
                pure_forced_outages = 0
                pure_planned_outages = 0
                if cluster.unit_count == 1:
                    if po_candidates == 1 and fo_candidates == 1:
                        mixed_outages = 1
                        pure_planned_outages = 0
                        pure_forced_outages = 0
                    else:
                        mixed_outages = 0
                        pure_planned_outages = int(po_candidates)
                        pure_forced_outages = int(fo_candidates)
                else:
                    if current_available_units != 0:
                        mixed_outages = int(
                            po_candidates * fo_candidates // current_available_units
                        )
                        pure_planned_outages = int(po_candidates - mixed_outages)
                        pure_forced_outages = int(fo_candidates - mixed_outages)
                    else:
                        mixed_outages = 0
                        pure_planned_outages = 0
                        pure_forced_outages = 0

                # = units stopping =
                current_available_units -= (
                    pure_planned_outages + pure_forced_outages + mixed_outages
                )

                # = generating outage duration = (from the law)
                po_duration = 0
                fo_duration = 0

                if pure_planned_outages != 0 or mixed_outages != 0:
                    po_duration = pod_generator.generate_duration(day)
                if pure_forced_outages != 0 or mixed_outages != 0:
                    fo_duration = fod_generator.generate_duration(day)

                if pure_planned_outages != 0:
                    return_timestep = (now + po_duration) % log_size
                    log[return_timestep] += pure_planned_outages
                    logp[return_timestep] += pure_planned_outages
                if pure_forced_outages != 0:
                    return_timestep = (now + fo_duration) % log_size
                    log[return_timestep] += pure_forced_outages
                if mixed_outages != 0:
                    return_timestep = (now + po_duration + fo_duration) % log_size
                    log[return_timestep] += mixed_outages
                    logp[return_timestep] += mixed_outages

                # = storing output in output arrays =
                if ts_index >= 0:  # drop the 2 first generated timeseries
                    output.planned_outages[ts_index][day] = pure_planned_outages
                    output.forced_outages[ts_index][day] = pure_forced_outages
                    output.mixed_outages[ts_index][day] = mixed_outages
                    output.planned_outage_durations[ts_index][day] = po_duration
                    output.forced_outage_durations[ts_index][day] = fo_duration
                    output.available_units[ts_index][day] = current_available_units

                now = (now + 1) % log_size

        output.available_power = (
            np.repeat(output.available_units, 24, axis=1)
            * cluster.nominal_power
            * np.tile(cluster.modulation, self.days)
        )
        return output
