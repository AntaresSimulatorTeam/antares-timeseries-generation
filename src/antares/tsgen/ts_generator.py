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
from typing import List, Tuple

import numpy as np
import numpy.typing as npt

from antares.tsgen.duration_generator import ProbabilityLaw, make_duration_generator
from antares.tsgen.random_generator import RNG, MersenneTwisterRNG

# probabilities above FAILURE_RATE_EQ_1 are considered certain (equal to 1)
FAILURE_RATE_EQ_1 = 0.999


IntArray = npt.NDArray[np.int_]
FloatArray = npt.NDArray[np.float_]


@dataclass
class ThermalCluster:
    # available units of the cluster
    unit_count: int
    # nominal power
    nominal_power: float
    # modulation of the nominal power for a certain hour in the day (between 0 and 1)
    modulation: IntArray

    # forced and planed outage parameters
    # indexed by day of the year
    fo_duration: IntArray
    fo_rate: FloatArray
    po_duration: IntArray
    po_rate: FloatArray
    npo_min: IntArray  # number of planed outage min in a day
    npo_max: IntArray  # number of planed outage max in a day

    # forced and planed outage probability law and volatility
    # volatility characterizes the distance from the expect at which the value drawn can be
    fo_law: ProbabilityLaw
    fo_volatility: float
    po_law: ProbabilityLaw
    po_volatility: float

    def __post_init__(self) -> None:
        _check_cluster(self)


def _check_1_dim(array: npt.NDArray, name: str) -> None:
    if array.ndim != 1:
        raise ValueError(f"{name} must be a 1 dimension array.")


def _check_array(condition: npt.NDArray[np.bool_], message: str) -> None:
    if condition.any():
        raise ValueError(f"{message}: {condition.nonzero()[0].tolist()}")


def _check_cluster(cluster: ThermalCluster) -> None:
    if cluster.unit_count <= 0:
        raise ValueError(
            f"Unit count must be strictly positive, got {cluster.unit_count}."
        )
    if cluster.nominal_power <= 0:
        raise ValueError(
            f"Nominal power must be strictly positive, got {cluster.nominal_power}."
        )
    if cluster.fo_volatility < 0:
        raise ValueError(
            f"Forced outage volatility must be positive, got {cluster.unit_count}."
        )
    if cluster.po_volatility < 0:
        raise ValueError(
            f"Planned outage volatility must be positive, got {cluster.unit_count}."
        )

    _check_1_dim(cluster.fo_rate, "Forced outage failure rate")
    _check_1_dim(cluster.fo_duration, "Forced outage duration")
    _check_1_dim(cluster.po_rate, "Planned failure rate")
    _check_1_dim(cluster.po_duration, "Planned outage duration")
    _check_1_dim(cluster.npo_min, "Minimum count of planned outages")
    _check_1_dim(cluster.npo_max, "Maximum count of planned outages")
    _check_1_dim(cluster.modulation, "Hourly modulation")
    if len(cluster.modulation) != 24:
        raise ValueError("hourly modulation array must have 24 values.")

    _check_array(
        cluster.fo_rate < 0, "Forced failure rate is negative on following days"
    )
    _check_array(
        cluster.fo_rate > 1, "Forced failure rate is greater than 1 on following days"
    )
    _check_array(
        cluster.fo_duration < 0, "Forced outage duration is negative on following days"
    )
    _check_array(
        cluster.po_rate < 0, "Planned failure rate is negative on following days"
    )
    _check_array(
        cluster.po_rate > 1, "Planned failure rate is greater than 1 on following days"
    )
    _check_array(
        cluster.po_duration < 0, "Planned outage duration is negative on following days"
    )
    _check_array(
        cluster.modulation < 0, "Hourly modulation is negative on following hours"
    )

    lengths = {
        len(a)
        for a in [
            cluster.fo_rate,
            cluster.fo_duration,
            cluster.po_rate,
            cluster.po_duration,
            cluster.npo_min,
            cluster.npo_max,
        ]
    }
    if len(lengths) != 1:
        raise ValueError(f"Not all daily arrays have same size, got {lengths}")


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


def _column_powers(column: FloatArray, width: int) -> npt.NDArray:
    """
    Returns a matrix of given width where column[i] is the ith power of the input vector.
    """
    powers = np.arange(width)
    powers.shape = (1, len(powers))
    column.shape = (len(column), 1)
    return pow(column, powers)


def _daily_to_hourly(daily_data: npt.NDArray) -> npt.NDArray:
    """
    Converts daily rows of a 2D array to hourly rows
    """
    if daily_data.ndim != 2:
        raise ValueError("Daily data must be a 2D-array")
    return np.repeat(daily_data, 24, axis=1)


def _categorize_outages(
    available_units: int, po_candidates: int, fo_candidates: int
) -> Tuple[int, int, int]:
    if po_candidates > available_units:
        raise ValueError(
            "Planned outages candidate cannot be greater than available units."
        )
    if fo_candidates > available_units:
        raise ValueError(
            "Forced outages candidate cannot be greater than available units."
        )

    if available_units == 0:
        return 0, 0, 0
    mixed = po_candidates * fo_candidates // available_units
    pure_planned = po_candidates - mixed
    pure_forced = fo_candidates - mixed
    return mixed, pure_planned, pure_forced


class ForcedOutagesDrawer:
    def __init__(self, rng: RNG, unit_count: int, failure_rate: FloatArray):
        days = len(failure_rate)
        self.rng = rng
        self.failure_rate = failure_rate
        self.ff = np.zeros(days, dtype=float)  # ff = lf / (1 - lf)
        a = np.zeros(shape=days, dtype=float)
        mask = failure_rate <= FAILURE_RATE_EQ_1
        a[mask] = 1 - failure_rate[mask]
        self.ff[mask] = failure_rate[mask] / a
        self.fpow = _column_powers(a, unit_count + 1)

    def draw(self, available_units: int, day: int) -> int:
        fo_candidates = 0
        rate = self.failure_rate[day]
        if rate > 0 and rate <= FAILURE_RATE_EQ_1:
            draw = self.rng.next()
            last = self.fpow[day, available_units]
            if draw > last:
                cumul = last
                for d in range(1, available_units + 1):
                    last = last * self.ff[day] * (available_units + 1 - d) / d
                    cumul += last
                    fo_candidates = d
                    if draw <= cumul:
                        break
        elif rate > FAILURE_RATE_EQ_1:  # TODO: not same comparison as cpp ?
            fo_candidates = available_units
        else:  # self.lf[day] == 0
            fo_candidates = 0
        return fo_candidates


def _compute_failure_rates(outage_rates: FloatArray, durations: IntArray) -> FloatArray:
    """
    Daily failure rates (= chance that an outage occurs on a given day), computed
    from outage rates (= share of a period during which a unit is in outage),
    and outage duration expectations.
    """
    return outage_rates / (outage_rates + durations * (1 - outage_rates))


def _combine_failure_rates(rates1: FloatArray, rates2: FloatArray) -> None:
    ## i dont understand what these calulations are for
    ## consequently reduce the lower failure rate
    mask = rates1 < rates2
    rates1[mask] *= (1 - rates2[mask]) / (1 - rates1[mask])
    mask = rates2 < rates1
    rates2[mask] *= (1 - rates1[mask]) / (1 - rates2[mask])


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
        _check_cluster(cluster)

        # TODO: Remove this log size limit, seems useless and error prone if very large durations
        log_size = 4000  # >= 5 * (max(df) + max(dp))
        # the number of starting (if positive)/ stopping (if negative) units (due to FO and PO) at a given time
        log = np.zeros(log_size, dtype=int)
        # same but only for PO; necessary to ensure maximum and minimum PO is respected
        logp = np.zeros(log_size, dtype=int)

        ## ???
        pp = np.zeros(self.days, dtype=float)  # pp = lp / (1 - lp)

        # lf and lp represent the forced and programed failure rate
        # failure rate means the probability to enter in outage each day
        # its value is given by: OR / [OR + OD * (1 - OR)]
        daily_fo_rate = _compute_failure_rates(cluster.fo_rate, cluster.fo_duration)
        daily_po_rate = _compute_failure_rates(cluster.po_rate, cluster.po_duration)
        _combine_failure_rates(daily_fo_rate, daily_po_rate)

        fo_drawer = ForcedOutagesDrawer(
            rng=self.rng, unit_count=cluster.unit_count, failure_rate=daily_fo_rate
        )
        b = np.zeros(shape=self.days, dtype=float)

        mask = daily_po_rate <= FAILURE_RATE_EQ_1
        b[mask] = 1 - daily_po_rate[mask]
        pp[mask] = daily_po_rate[mask] / b

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

                fo_candidates = fo_drawer.draw(current_available_units, day)
                po_candidates = 0

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
                mixed_outages, planned_outages, forced_outages = _categorize_outages(
                    current_available_units, po_candidates, fo_candidates
                )

                # = units stopping =
                current_available_units -= (
                    planned_outages + forced_outages + mixed_outages
                )

                # = generating outage duration = (from the law)
                po_duration = 0
                fo_duration = 0

                if planned_outages != 0 or mixed_outages != 0:
                    po_duration = pod_generator.generate_duration(day)
                if forced_outages != 0 or mixed_outages != 0:
                    fo_duration = fod_generator.generate_duration(day)

                if planned_outages != 0:
                    return_timestep = (now + po_duration) % log_size
                    log[return_timestep] += planned_outages
                    logp[return_timestep] += planned_outages
                if forced_outages != 0:
                    return_timestep = (now + fo_duration) % log_size
                    log[return_timestep] += forced_outages
                if mixed_outages != 0:
                    return_timestep = (now + po_duration + fo_duration) % log_size
                    log[return_timestep] += mixed_outages
                    logp[return_timestep] += mixed_outages

                # = storing output in output arrays =
                if ts_index >= 0:  # drop the 2 first generated timeseries
                    output.planned_outages[ts_index, day] = planned_outages
                    output.forced_outages[ts_index, day] = forced_outages
                    output.mixed_outages[ts_index, day] = mixed_outages
                    output.planned_outage_durations[ts_index, day] = po_duration
                    output.forced_outage_durations[ts_index, day] = fo_duration
                    output.available_units[ts_index, day] = current_available_units

                now = (now + 1) % log_size

        hourly_available_units = _daily_to_hourly(output.available_units)
        hourly_modulation = np.tile(cluster.modulation, self.days)
        output.available_power = (
            hourly_available_units * cluster.nominal_power * hourly_modulation
        )

        return output
