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
from typing import Any, Tuple

import numpy as np
import numpy.typing as npt
from numpy import dtype, ndarray

from antares.tsgen.duration_generator import DurationGenerator, ProbabilityLaw, make_duration_generator
from antares.tsgen.random_generator import RNG, MersenneTwisterRNG

# probabilities above FAILURE_RATE_EQ_1 are considered certain (equal to 1)
FAILURE_RATE_EQ_1 = 0.999

IntArray = npt.NDArray[np.int_]
FloatArray = npt.NDArray[np.float_]


@dataclass()
class OutageGenerationParameters:
    # available units of the cluster
    unit_count: int

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
        _check_outage_gen_params(self)


@dataclass
class ThermalCluster:
    # available units of the cluster
    # unit_count: int
    outage_gen_params: OutageGenerationParameters
    # nominal power
    nominal_power: float

    modulation: IntArray

    def __post_init__(self) -> None:
        _check_cluster(self)


@dataclass
class LinkCapacity:
    # outage generation parameters
    outage_gen_params: OutageGenerationParameters

    # nominal capacity
    nominal_capacity: float

    # direct / indirect modulation of the nominal capacity
    modulation_direct: FloatArray
    modulation_indirect: FloatArray

    def __post_init__(self) -> None:
        _check_link_capacity(self)


def _check_1_dim(array: npt.NDArray, name: str) -> None:
    if array.ndim != 1:
        raise ValueError(f"{name} must be a 1 dimension array.")


def _check_array(condition: npt.NDArray[np.bool_], message: str) -> None:
    if condition.any():
        raise ValueError(f"{message}: {condition.nonzero()[0].tolist()}")


def _check_outage_gen_params(outage_gen_params: OutageGenerationParameters) -> None:
    if outage_gen_params.unit_count < 0:
        raise ValueError(f"Unit count must be positive, got {outage_gen_params.unit_count}.")
    if outage_gen_params.fo_volatility < 0:
        raise ValueError(f"Forced outage volatility must be positive, got {outage_gen_params.unit_count}.")
    if outage_gen_params.po_volatility < 0:
        raise ValueError(f"Planned outage volatility must be positive, got {outage_gen_params.unit_count}.")

    _check_1_dim(outage_gen_params.fo_rate, "Forced outage failure rate")
    _check_1_dim(outage_gen_params.fo_duration, "Forced outage duration")
    _check_1_dim(outage_gen_params.po_rate, "Planned failure rate")
    _check_1_dim(outage_gen_params.po_duration, "Planned outage duration")
    _check_1_dim(outage_gen_params.npo_min, "Minimum count of planned outages")
    _check_1_dim(outage_gen_params.npo_max, "Maximum count of planned outages")

    _check_array(outage_gen_params.fo_rate < 0, "Forced failure rate is negative on following days")
    _check_array(outage_gen_params.fo_rate > 1, "Forced failure rate is greater than 1 on following days")
    _check_array(outage_gen_params.fo_duration <= 0, "Forced outage duration is null or negative on following days")
    _check_array(outage_gen_params.po_rate < 0, "Planned failure rate is negative on following days")
    _check_array(outage_gen_params.po_rate > 1, "Planned failure rate is greater than 1 on following days")
    _check_array(outage_gen_params.po_duration <= 0, "Planned outage duration is null or negative on following days")


def _check_cluster(cluster: ThermalCluster) -> None:
    if cluster.nominal_power < 0:
        raise ValueError(f"Nominal power must be positive, got {cluster.nominal_power}.")

    _check_outage_gen_params(cluster.outage_gen_params)

    _check_1_dim(cluster.modulation, "Hourly modulation")

    if len(cluster.modulation) != 8760:
        raise ValueError("hourly modulation array must have 8760 values.")

    _check_array(cluster.modulation < 0, "Hourly modulation is negative on following hours")

    _check_lengths(cluster.outage_gen_params)


def _check_link_capacity(link_capacity: LinkCapacity) -> None:
    if link_capacity.nominal_capacity <= 0:
        raise ValueError(f"Nominal power must be strictly positive, got {link_capacity.nominal_capacity}.")

    _check_outage_gen_params(link_capacity.outage_gen_params)

    _check_1_dim(link_capacity.modulation_indirect, "Direct modulation")
    _check_1_dim(link_capacity.modulation_indirect, "Indirect hourly modulation")

    if len(link_capacity.modulation_direct) != 8760 or len(link_capacity.modulation_indirect) != 8760:
        raise ValueError("hourly modulation array must have 8760 values.")

    _check_array(link_capacity.modulation_direct < 0, "Hourly direct modulation is negative on following hours")
    _check_array(link_capacity.modulation_indirect < 0, "Hourly indirect modulation is negative on following hours")

    _check_lengths(link_capacity.outage_gen_params)


def _check_lengths(outage_gen_params: OutageGenerationParameters) -> None:
    lengths = {
        len(a)
        for a in [
            outage_gen_params.fo_rate,
            outage_gen_params.fo_duration,
            outage_gen_params.po_rate,
            outage_gen_params.po_duration,
            outage_gen_params.npo_min,
            outage_gen_params.npo_max,
        ]
    }

    if len(lengths) != 1:
        raise ValueError(f"Not all daily arrays have same size, got {lengths}")


class OutageOutput:
    def __init__(self, ts_count: int, days: int) -> None:
        # number_of_timeseries
        self.ts_count = ts_count
        # number of days
        self.days = days
        self.available_units = np.zeros(shape=(days, ts_count), dtype=int)
        # number of pure planed, pure forced and mixed outage each day
        self.planned_outages = np.zeros((days, ts_count), dtype=int)
        self.forced_outages = np.zeros((days, ts_count), dtype=int)
        self.mixed_outages = np.zeros((days, ts_count), dtype=int)
        # number of pure planed and pure forced outage duration each day
        # (mixed outage duration = pod + fod)
        self.planned_outage_durations = np.zeros((days, ts_count), dtype=int)
        self.forced_outage_durations = np.zeros((days, ts_count), dtype=int)


class ClusterOutputTimeseries:
    def __init__(self, outage_output: OutageOutput) -> None:
        # output parameters
        self.outage_output = outage_output
        # available power each hours
        self.available_power = np.zeros((24 * outage_output.days, outage_output.ts_count), dtype=float)


class LinkOutputTimeseries:
    def __init__(self, outage_output: OutageOutput) -> None:
        # output parameters
        self.outage_output = outage_output
        # direct available power each hours
        self.direct_available_power = np.zeros((24 * outage_output.days, outage_output.ts_count), dtype=float)
        # available power each hours
        self.indirect_available_power = np.zeros((24 * outage_output.days, outage_output.ts_count), dtype=float)


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
    return np.repeat(daily_data, 24, axis=0)


def _categorize_outages(available_units: int, po_candidates: int, fo_candidates: int) -> Tuple[int, int, int]:
    if po_candidates > available_units:
        raise ValueError("Planned outages candidate cannot be greater than available units.")
    if fo_candidates > available_units:
        raise ValueError("Forced outages candidate cannot be greater than available units.")

    if available_units == 0:
        return 0, 0, 0
    mixed = po_candidates * fo_candidates // available_units
    pure_planned = po_candidates - mixed
    pure_forced = fo_candidates - mixed
    return mixed, pure_planned, pure_forced


class ForcedOutagesDrawer:
    def __init__(self, rng: RNG, unit_count: int, failure_rate: FloatArray):
        self.rng = rng
        self.failure_rate = failure_rate
        mask = failure_rate <= FAILURE_RATE_EQ_1
        a = np.where(mask, 1 - failure_rate, 0)
        self.ff = np.where(mask, failure_rate / a, 0)  # ff = lf / (1 - lf)
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
        elif rate > 0:
            fo_candidates = available_units
        else:
            fo_candidates = 0
        return fo_candidates


class PlannedOutagesDrawer:
    def __init__(self, rng: RNG, unit_count: int, failure_rate: FloatArray):
        self.rng = rng
        self.failure_rate = failure_rate
        mask = failure_rate <= FAILURE_RATE_EQ_1
        a = np.where(mask, 1 - failure_rate, 0)
        self.pp = np.where(mask, failure_rate / a, 0)  # pp = lp / (1 - lp)
        self.ppow = _column_powers(a, unit_count + 1)

    def draw(self, available_units: int, day: int, stock: int) -> Tuple[int, int]:
        po_candidates = 0
        rate = self.failure_rate[day]

        if rate > 0 and rate <= FAILURE_RATE_EQ_1:
            apparent_available_units = available_units
            if stock >= 0 and stock <= available_units:
                apparent_available_units -= stock
            elif stock > available_units:
                apparent_available_units = 0

            draw = self.rng.next()
            last = self.ppow[day, apparent_available_units]
            if draw > last:
                cumul = last
                for d in range(1, apparent_available_units + 1):
                    last = last * self.pp[day] * (apparent_available_units + 1 - d) / d
                    cumul += last
                    po_candidates = d
                    if draw <= cumul:
                        break
        elif rate > 0:
            po_candidates = available_units
        else:
            po_candidates = 0
        return po_candidates, stock


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
    mask = (rates1 > 0) & (rates1 < rates2)
    rates1[mask] *= (1 - rates2[mask]) / (1 - rates1[mask])
    mask = (rates2 > 0) & (rates2 < rates1)
    rates2[mask] *= (1 - rates1[mask]) / (1 - rates2[mask])


def _compare_apparent_po(current_available_units: int, po_candidates: int, stock: int) -> tuple[int, int]:
    candidate = po_candidates + stock
    if 0 <= candidate <= current_available_units:
        po_candidates = candidate
        stock = 0
    if candidate > current_available_units:
        po_candidates = current_available_units
        stock = candidate - current_available_units
    if candidate < 0:
        po_candidates = 0
        stock = candidate
    return po_candidates, stock


class TimeseriesGenerator:
    def __init__(self, rng: RNG = MersenneTwisterRNG(), days: int = 365) -> None:
        self.rng = rng
        self.days = days

    def _generate_outages(
        self,
        outage_gen_params: OutageGenerationParameters,
        log: ndarray[Any, dtype],
        log_size: int,
        logp: ndarray[Any, dtype],
        number_of_timeseries: int,
        output: OutageOutput,
    ) -> None:
        daily_fo_rate = _compute_failure_rates(outage_gen_params.fo_rate, outage_gen_params.fo_duration)
        daily_po_rate = _compute_failure_rates(outage_gen_params.po_rate, outage_gen_params.po_duration)
        _combine_failure_rates(daily_fo_rate, daily_po_rate)

        fo_drawer = ForcedOutagesDrawer(self.rng, outage_gen_params.unit_count, daily_fo_rate)
        po_drawer = PlannedOutagesDrawer(self.rng, outage_gen_params.unit_count, daily_po_rate)

        fod_generator = make_duration_generator(
            self.rng, outage_gen_params.fo_law, outage_gen_params.fo_volatility, outage_gen_params.fo_duration
        )
        pod_generator = make_duration_generator(
            self.rng, outage_gen_params.po_law, outage_gen_params.po_volatility, outage_gen_params.po_duration
        )

        self.output_generation(
            outage_gen_params,
            fo_drawer,
            fod_generator,
            log,
            log_size,
            logp,
            number_of_timeseries,
            output,
            po_drawer,
            pod_generator,
        )

    def generate_time_series_for_links(self, link: LinkCapacity, number_of_timeseries: int) -> LinkOutputTimeseries:
        """
        generation of multiple timeseries for a given link capacity
        """
        _check_link_capacity(link)

        # TODO: Remove this log size limit, seems useless and error prone if very large durations
        log_size = 4000  # >= 5 * (max(df) + max(dp))
        # the number of starting (if positive)/ stopping (if negative) units (due to FO and PO) at a given time
        log = np.zeros(log_size, dtype=int)
        # same but only for PO; necessary to ensure maximum and minimum PO is respected
        logp = np.zeros(log_size, dtype=int)

        # --- calculation ---
        # the two first generated time series will be dropped, necessary to make system stable and physically coherent
        # as a consequence, N + 2 time series will be computed

        # output that will be returned
        outage_params = OutageOutput(number_of_timeseries, self.days)
        link_output = LinkOutputTimeseries(outage_params)

        self._generate_outages(
            link.outage_gen_params, log, log_size, logp, number_of_timeseries, link_output.outage_output
        )

        hourly_available_units = _daily_to_hourly(link_output.outage_output.available_units)

        link_output.direct_available_power = (
            hourly_available_units * link.nominal_capacity * link.modulation_direct[:, np.newaxis]
        )
        link_output.indirect_available_power = (
            hourly_available_units * link.nominal_capacity * link.modulation_indirect[:, np.newaxis]
        )

        return link_output

    def generate_time_series_for_clusters(
        self,
        cluster: ThermalCluster,
        number_of_timeseries: int,
    ) -> ClusterOutputTimeseries:
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

        # lf and lp represent the forced and programed failure rate
        # failure rate means the probability to enter in outage each day
        # its value is given by: OR / [OR + OD * (1 - OR)]

        # --- calculation ---
        # the two first generated time series will be dropped, necessary to make system stable and physically coherent
        # as a consequence, N + 2 time series will be computed

        # output that will be returned
        outage_output = OutageOutput(number_of_timeseries, self.days)
        cluster_output = ClusterOutputTimeseries(outage_output)

        if cluster.nominal_power == 0 or cluster.outage_gen_params.unit_count == 0:
            # In these cases, we shouldn't perform any calculation. The result will just be a matrix full of zeros.
            return cluster_output

        self._generate_outages(
            cluster.outage_gen_params, log, log_size, logp, number_of_timeseries, cluster_output.outage_output
        )

        #
        hourly_available_units = _daily_to_hourly(cluster_output.outage_output.available_units)
        cluster_output.available_power = (
            hourly_available_units * cluster.nominal_power * cluster.modulation[:, np.newaxis]
        )
        np.round(cluster_output.available_power)
        return cluster_output

    def output_generation(
        self,
        outage_gen_params: OutageGenerationParameters,
        fo_drawer: ForcedOutagesDrawer,
        fod_generator: DurationGenerator,
        log: ndarray[Any, dtype],
        log_size: int,
        logp: ndarray[Any, dtype],
        number_of_timeseries: int,
        output: OutageOutput,
        po_drawer: PlannedOutagesDrawer,
        pod_generator: DurationGenerator,
    ) -> None:
        # dates
        now = 0
        # current number of PO and AU (avlaible units)
        current_planned_outages = 0
        current_available_units = outage_gen_params.unit_count
        # stock is a way to keep the number of PO pushed back due to PO max / antcipated due to PO min
        # stock > 0 number of PO pushed back, stock < 0 number of PO antcipated
        stock = 0
        for ts_index in range(-2, number_of_timeseries):
            for day in range(self.days):
                # = return of units wich were in outage =
                current_planned_outages -= logp[now]
                logp[now] = 0  # set to 0 because this cell will be use again later (in self.log_size days)
                current_available_units += log[now]
                log[now] = 0

                if current_planned_outages > outage_gen_params.npo_max[day]:
                    cible_retour = current_planned_outages - outage_gen_params.npo_max[day]
                    cumul_retour = 0
                    for index in range(1, log_size):
                        if cumul_retour == cible_retour:
                            break
                        if logp[(now + index) % log_size] + cumul_retour >= cible_retour:
                            logp[(now + index) % log_size] -= cible_retour - cumul_retour
                            log[(now + index) % log_size] -= cible_retour - cumul_retour
                            cumul_retour = cible_retour
                        else:
                            if logp[(now + index) % log_size] > 0:
                                cumul_retour += logp[(now + index) % log_size]
                                log[(now + index) % log_size] -= logp[(now + index) % log_size]
                                logp[(now + index) % log_size] = 0
                    current_available_units += cible_retour
                    current_planned_outages = outage_gen_params.npo_max[day]

                fo_candidates = fo_drawer.draw(current_available_units, day)
                po_candidates, stock = po_drawer.draw(current_available_units, day, stock)

                # apparent PO is compared to cur_nb_AU, considering stock
                po_candidates, stock = _compare_apparent_po(current_available_units, po_candidates, stock)

                # = checking min and max PO =
                if po_candidates + current_planned_outages > outage_gen_params.npo_max[day]:
                    # too many PO to place
                    # the excedent is placed in stock
                    stock += po_candidates + current_planned_outages - outage_gen_params.npo_max[day]
                    po_candidates = outage_gen_params.npo_max[day] - current_planned_outages
                    current_planned_outages = outage_gen_params.npo_max[day]
                elif po_candidates + current_planned_outages < outage_gen_params.npo_min[day]:
                    if outage_gen_params.npo_min[day] - current_planned_outages > current_available_units:
                        stock -= current_available_units - po_candidates
                        po_candidates = current_available_units
                        current_planned_outages += po_candidates
                    else:
                        stock -= outage_gen_params.npo_min[day] - (po_candidates + current_planned_outages)
                        po_candidates = outage_gen_params.npo_min[day] - current_planned_outages
                        current_planned_outages = outage_gen_params.npo_min[day]
                else:
                    current_planned_outages += po_candidates

                # = distributing outage in category =
                # pure planed, pure forced, mixed
                mixed_outages, planned_outages, forced_outages = _categorize_outages(
                    current_available_units, po_candidates, fo_candidates
                )

                # = units stopping =
                current_available_units -= planned_outages + forced_outages + mixed_outages

                # = generating outage duration = (from the law)
                po_duration = 0
                fo_duration = 0

                if forced_outages != 0 or mixed_outages != 0:
                    fo_duration = fod_generator.generate_duration(day)
                if planned_outages != 0 or mixed_outages != 0:
                    po_duration = pod_generator.generate_duration(day)

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
                    output.planned_outages[day, ts_index] = planned_outages
                    output.forced_outages[day, ts_index] = forced_outages
                    output.mixed_outages[day, ts_index] = mixed_outages
                    output.planned_outage_durations[day, ts_index] = po_duration
                    output.forced_outage_durations[day, ts_index] = fo_duration
                    output.available_units[day, ts_index] = current_available_units

                now = (now + 1) % log_size
