"""Bounded experimental race scenarios; frequencies are not official win probabilities."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
import json
from math import isfinite
from typing import Callable

import numpy as np

from security.access import Principal, authorize
from security.limits import LimitRejected, SQLiteLimiter

COMPOUNDS = {'SOFT', 'MEDIUM', 'HARD', 'INTERMEDIATE', 'WET'}
LIMITS = ('No traffic, overtaking constraints, incidents, DNF, penalties, safety cars or red flags.',
          'Equal completed-lap decision boundary only; no lapped-car catch-up model.',
          'Fuel, track evolution and driver effects are confounded with assumed pace/degradation.',
          'Independent lap noise is clipped to three standard deviations; parameter uncertainty is included only when supplied.',
          'Intervals and scenario finish frequencies are uncalibrated; counterfactual accuracy is unknown.',
          'Pit timing/compound and additive weather paths are explicit assumptions; tyre suitability is not modeled.')


def _number(value, name, *, minimum=0, maximum=1_000_000):
    if type(value) not in (int, float) or not isfinite(value) or not minimum <= value <= maximum:
        raise ValueError(f'{name} must be finite within [{minimum}, {maximum}]')
    return float(value)


@dataclass(frozen=True)
class CompoundPace:
    compound: str
    pace_s: float  # Expected lap at tyre age zero; fuel is not separately identified.
    slope_s_per_lap: float
    sigma_s: float
    quality: str
    evidence: str
    parameter_draws: tuple[tuple[float, float], ...] = ()


@dataclass(frozen=True)
class PitLoss:
    net_loss_s: float | None = None
    stationary_s: float | None = None
    lane_travel_delta_s: float | None = None

    def __post_init__(self):
        if self.net_loss_s is not None:
            if self.stationary_s is not None or self.lane_travel_delta_s is not None:
                raise ValueError('use net pit loss OR stationary plus lane travel delta; never both')
            _number(self.net_loss_s, 'net pit loss', maximum=600)
        else:
            _number(self.stationary_s, 'stationary pit time', maximum=600)
            _number(self.lane_travel_delta_s, 'pit-lane travel delta excluding stationary time', maximum=600)

    @property
    def total_s(self):
        return self.net_loss_s if self.net_loss_s is not None else self.stationary_s + self.lane_travel_delta_s


@dataclass(frozen=True)
class PitStop:
    lap: int  # Stop charged at the end of this absolute race lap; fresh tyres on the next lap.
    compound: str
    loss: PitLoss


@dataclass(frozen=True)
class DriverScenario:
    driver_id: str
    completed_lap: int
    gap_s: float  # Gap at a shared completed-lap boundary, relative to the leader.
    compound: str
    tyre_age_laps: float
    paces: tuple[CompoundPace, ...]
    pits: tuple[PitStop, ...] = ()


@dataclass(frozen=True)
class SimulationConfig:
    remaining_laps: int
    samples: int
    seed: int
    weather_delta_s: tuple[float, ...]
    weather_assumption: str


@dataclass(frozen=True)
class DriverResult:
    driver_id: str
    expected_finish_position: float
    finish_frequencies: tuple[float, ...]
    finish_time_median_s: float
    finish_time_lower_s: float
    finish_time_upper_s: float
    trajectory_median_s: tuple[float, ...]


@dataclass(frozen=True)
class SimulationResult:
    drivers: tuple[DriverResult, ...]
    seed: int
    samples: int
    assumptions: dict
    limits: tuple[str, ...] = LIMITS
    calibrated: bool = False
    interpretation: str = ('Uncalibrated scenario finish frequencies among supplied participants, '
                           'not official win probabilities. Times are seconds after the decision leader boundary '
                           '(initial gap plus remaining laps), not total race elapsed time. Intervals are 90% simulation quantiles.')


def _validate(drivers, config):
    if not isinstance(config, SimulationConfig) or not isinstance(drivers, (tuple, list)) or not 1 <= len(drivers) <= 22:
        raise ValueError('scenario requires 1–22 drivers')
    if (type(config.remaining_laps) is not int or not 1 <= config.remaining_laps <= 100
            or type(config.samples) is not int or not 1 <= config.samples <= 2000
            or type(config.seed) is not int or not 0 <= config.seed < 2**32
            or not isinstance(config.weather_delta_s, (tuple, list))
            or len(config.weather_delta_s) != config.remaining_laps
            or not isinstance(config.weather_assumption, str) or not 1 <= len(config.weather_assumption.strip()) <= 1000):
        raise ValueError('invalid scenario bounds or explicit weather path')
    for delta in config.weather_delta_s:
        _number(delta, 'weather delta', minimum=-120, maximum=300)
    ids, laps, gaps = set(), set(), []
    for driver in drivers:
        if (not isinstance(driver, DriverScenario) or not isinstance(driver.driver_id, str)
                or not 1 <= len(driver.driver_id.strip()) <= 64 or driver.driver_id in ids
                or type(driver.completed_lap) is not int or not 0 <= driver.completed_lap <= 100
                or driver.completed_lap + config.remaining_laps > 100
                or driver.compound not in COMPOUNDS or not isinstance(driver.paces, (list, tuple))
                or not 1 <= len(driver.paces) <= 5 or not isinstance(driver.pits, (list, tuple)) or len(driver.pits) > 10):
            raise ValueError('invalid driver state or compound pace inputs')
        ids.add(driver.driver_id)
        laps.add(driver.completed_lap)
        gaps.append(_number(driver.gap_s, 'gap to decision leader', maximum=10_000))
        _number(driver.tyre_age_laps, 'current tyre age', maximum=300)
        paces = {}
        for pace in driver.paces:
            if (not isinstance(pace, CompoundPace) or pace.compound not in COMPOUNDS or pace.compound in paces
                    or pace.quality not in {'validated_prefix', 'user_assumption'}
                    or not isinstance(pace.evidence, str) or not 1 <= len(pace.evidence.strip()) <= 1000):
                raise ValueError('missing compound quality/provenance or duplicate compound')
            _number(pace.pace_s, 'age-zero pace', minimum=.001, maximum=1200)
            _number(pace.slope_s_per_lap, 'slope', minimum=-10, maximum=10)
            _number(pace.sigma_s, 'pace uncertainty', maximum=120)
            if not isinstance(pace.parameter_draws, (tuple, list)) or len(pace.parameter_draws) > 1000:
                raise ValueError('compound parameter uncertainty exceeds bounds')
            for draw in pace.parameter_draws:
                if not isinstance(draw, (tuple, list)) or len(draw) != 2:
                    raise ValueError('parameter draws require intercept and slope')
                _number(draw[0], 'draw age-zero pace', minimum=.001, maximum=1200)
                _number(draw[1], 'draw slope', minimum=-10, maximum=10)
            paces[pace.compound] = pace
        if driver.compound not in paces:
            raise ValueError('missing current compound pace/uncertainty')
        previous = driver.completed_lap
        for pit in driver.pits:
            if (not isinstance(pit, PitStop) or type(pit.lap) is not int
                    or not previous < pit.lap < driver.completed_lap + config.remaining_laps
                    or pit.compound not in paces or not isinstance(pit.loss, PitLoss)):
                raise ValueError('future pit laps must be ordered, before finish, and have explicit compound pace/loss')
            previous = pit.lap
    if len(laps) != 1 or min(gaps) != 0:
        raise ValueError('drivers require the same completed-lap boundary and a zero-gap reference leader')


def _simulate_race(drivers, config, *, checkpoint: Callable | None = None):
    """Internal trusted pure numerical core; public callers use simulate_race."""
    _validate(drivers, config)
    rng = np.random.default_rng(config.seed)
    n, horizon, samples = len(drivers), config.remaining_laps, config.samples
    times = np.zeros((samples, n, horizon + 1), dtype=float)
    for index, driver in enumerate(drivers):
        if checkpoint:
            checkpoint()
        times[:, index, 0] = driver.gap_s
        compound, age = driver.compound, driver.tyre_age_laps
        paces = {pace.compound: pace for pace in driver.paces}
        parameters = {}
        for pace in driver.paces:
            if pace.parameter_draws:
                draws = np.asarray(pace.parameter_draws)
                parameters[pace.compound] = draws[rng.integers(len(draws), size=samples)]
            else:
                parameters[pace.compound] = np.broadcast_to((pace.pace_s, pace.slope_s_per_lap), (samples, 2))
        pits = {pit.lap: pit for pit in driver.pits}
        for offset in range(horizon):
            age += 1
            pace = paces[compound]
            selected = parameters[compound]
            expected = selected[:, 0] + selected[:, 1] * age + config.weather_delta_s[offset]
            if np.any(expected - 3 * pace.sigma_s <= 0):
                raise ValueError('pace/uncertainty/weather assumptions can produce nonpositive laps')
            lap_times = expected + np.clip(rng.normal(0, pace.sigma_s, samples), -3 * pace.sigma_s, 3 * pace.sigma_s)
            pit = pits.get(driver.completed_lap + offset + 1)
            if pit:
                lap_times += pit.loss.total_s
                compound, age = pit.compound, 0
            times[:, index, offset + 1] = times[:, index, offset] + lap_times
    # Seeded tie-breaks avoid input-order bias for equal simulated times.
    tie_breaks = rng.random((samples, n))
    order = np.lexsort((tie_breaks, times[:, :, -1]), axis=1)
    ranks = np.empty_like(order)
    np.put_along_axis(ranks, order, np.broadcast_to(np.arange(1, n + 1), (samples, n)), axis=1)
    results = []
    for index, driver in enumerate(drivers):
        low, median, high = np.quantile(times[:, index, -1], (.05, .5, .95))
        frequencies = tuple(float(np.mean(ranks[:, index] == rank)) for rank in range(1, n + 1))
        results.append(DriverResult(driver.driver_id, float(np.mean(ranks[:, index])), frequencies,
                                    float(median), float(low), float(high),
                                    tuple(float(t) for t in np.median(times[:, index], axis=0))))
    return SimulationResult(tuple(results), config.seed, samples,
                            dict(config=asdict(config), drivers=[asdict(d) for d in drivers],
                                 pit_timing='At end of absolute pit lap; new tyre age starts at one next lap.'))


def simulate_race(drivers, config, *, principal: Principal, limiter: SQLiteLimiter,
                  request_id: str, public: bool = False) -> SimulationResult:
    """Authorize and atomically admit each bounded scenario; never redispatch a replay."""
    authorize(principal, 'scenario')
    _validate(drivers, config)
    fingerprint = sha256(json.dumps(dict(drivers=[asdict(d) for d in drivers], config=asdict(config)),
                                     sort_keys=True, allow_nan=False).encode()).hexdigest()
    lease = limiter.reserve(principal, request_id, fingerprint, action='scenario', public=public)
    if lease.replayed:
        raise LimitRejected('request_already_submitted')
    if lease.status != 'active':
        limiter.settle(principal, request_id, failed=True)
        raise LimitRejected('busy')
    try:
        limiter.checkpoint(principal, request_id)
        result = _simulate_race(drivers, config,
                                checkpoint=lambda: limiter.checkpoint(principal, request_id))
        limiter.checkpoint(principal, request_id)
    except Exception:
        limiter.settle(principal, request_id, failed=True)
        raise
    limiter.settle(principal, request_id)
    return result
