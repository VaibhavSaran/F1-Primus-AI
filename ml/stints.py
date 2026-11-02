"""Experimental clean-stint fits at an observed relative clock, never full-race hindsight."""
from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Any, Iterable

import numpy as np

from ml.features import clean_laps


@dataclass(frozen=True)
class StintModel:
    driver_id: str
    compound: str
    stint: int
    intercept_s: float
    slope_s_per_lap: float
    residual_sigma_s: float
    sample_count: int
    first_age_laps: float
    last_age_laps: float
    clock_ms: int
    source_id: str
    bootstrap_parameters: tuple[tuple[float, float], ...]
    residuals_s: tuple[float, ...]
    seed: int


@dataclass(frozen=True)
class StintFit:
    models: tuple[StintModel, ...]
    reasons: tuple[str, ...]
    clock_ms: int
    eligible_laps: int
    provenance: dict[str, Any]
    calibrated: bool = False


@dataclass(frozen=True)
class PaceForecast:
    median_s: tuple[float, ...]
    lower_s: tuple[float, ...]
    upper_s: tuple[float, ...]
    calibrated: bool = False
    interpretation: str = 'Experimental 90% bootstrap intervals; coverage has not been calibrated.'


def _finite(value, minimum=0):
    return type(value) in (int, float) and isfinite(value) and value >= minimum


def fit_stint_model(laps: Iterable[dict], *, clock_ms: int, source: dict,
                    min_laps: int = 6, bootstrap_samples: int = 200, seed: int = 0) -> StintFit:
    """Trusted numeric core. An exposed fitting job must use the shared job guard.

    Source quality is an attestation from the trusted normalization boundary;
    relative-clock filtering alone does not establish historical publication time.
    """
    if (type(clock_ms) is not int or not 0 <= clock_ms <= 86_400_000
            or type(min_laps) is not int or not 6 <= min_laps <= 100
            or type(bootstrap_samples) is not int or not 20 <= bootstrap_samples <= 1000
            or type(seed) is not int or not 0 <= seed < 2**32):
        raise ValueError('invalid stint fitting bounds')
    provenance = dict(source) if isinstance(source, dict) else {}
    provenance.update(scope='observed_clock_prefix', historical_publication_verified=False,
                      fuel_effect='not separated from tyre/driver/track effects',
                      interpretation='Within-stint association, not measured tyre health or causal tyre wear.')
    if (not isinstance(source, dict) or source.get('quality_status') != 'validated'
            or source.get('clock_basis') != 'session_relative_observed'
            or any(not isinstance(source.get(key), str) or not source[key].strip()
                   for key in ('provider', 'source_id'))):
        return StintFit((), ('Insufficient quality/source provenance for a clean observed-prefix fit.',),
                        clock_ms, 0, provenance)
    prefix = []
    for index, lap in enumerate(laps):
        if index >= 20_000:
            raise ValueError('lap input exceeds bounded fitting workload')
        if not isinstance(lap, dict):
            raise ValueError('lap records must be mappings')
        clock = lap.get('clock_ms')
        if type(clock) is not int or not 0 <= clock <= clock_ms:
            continue
        # clean_laps accepts missing deletion/pit flags; strategy fitting requires proof.
        if any(lap.get(key) is not False for key in ('deleted', 'pit_in', 'pit_out')):
            continue
        if (not isinstance(lap.get('driver_id'), str) or not lap['driver_id'].strip()
                or lap.get('compound') not in {'SOFT', 'MEDIUM', 'HARD', 'INTERMEDIATE', 'WET'}
                or not _finite(lap.get('stint'), 1) or int(lap['stint']) != lap['stint']
                or not _finite(lap.get('tyre_age_laps'), 0)
                or type(lap.get('lap_number')) is not int):
            continue
        prefix.append(lap)
    groups, seen = {}, set()
    for lap in clean_laps(prefix):
        identity = (lap['driver_id'], lap['lap_number'])
        if identity in seen:
            raise ValueError('duplicate eligible driver/lap identity')
        seen.add(identity)
        key = (lap['driver_id'], lap['compound'], int(lap['stint']))
        groups.setdefault(key, []).append(lap)
    if len(groups) > 100:
        raise ValueError('too many stint groups')
    models, reasons = [], []
    rng = np.random.default_rng(seed)
    for key, rows in sorted(groups.items()):
        rows = sorted(rows, key=lambda row: (row['clock_ms'], row['lap_number']))[-100:]
        x = np.asarray([row['tyre_age_laps'] for row in rows], dtype=float)
        y = np.asarray([row['lap_time_s'] for row in rows], dtype=float)
        if len(rows) < min_laps or len(set(x)) < min_laps:
            reasons.append(f'{key}: insufficient clean laps or distinct tyre ages (need {min_laps}).')
            continue
        dx = x[:, None] - x[None, :]
        dy = y[:, None] - y[None, :]
        mask = np.triu(np.ones(dx.shape, dtype=bool), 1) & (dx != 0)
        slope = float(np.median(dy[mask] / dx[mask]))
        intercept = float(np.median(y - slope * x))
        residual = y - intercept - slope * x
        scale = float(1.4826 * np.median(np.abs(residual - np.median(residual))))
        retained = np.abs(residual) <= max(3 * scale, 1e-7)
        if np.count_nonzero(retained) < min_laps:
            reasons.append(f'{key}: insufficient clean laps after robust residual filtering.')
            continue
        clean_x, clean_residual = x[retained], residual[retained]
        centered_x = clean_x - np.mean(clean_x)
        denominator = float(np.sum(centered_x ** 2))
        bootstrap = []
        for _ in range(bootstrap_samples):
            errors = rng.choice(clean_residual, len(clean_x), replace=True)
            slope_delta = float(np.sum(centered_x * errors) / denominator)
            bootstrap.append((intercept + float(np.mean(errors)) - slope_delta * float(np.mean(clean_x)),
                              slope + slope_delta))
        models.append(StintModel(*key, intercept, slope, scale, int(np.count_nonzero(retained)),
                                 float(min(clean_x)), float(max(clean_x)),
                                 clock_ms, source['source_id'], tuple(bootstrap),
                                 tuple(float(r) for r in clean_residual), seed))
    if not groups:
        reasons.append('Insufficient clean laps with known deletion, pit, flag, compound and clock evidence.')
    return StintFit(tuple(models), tuple(reasons), clock_ms, len(seen), provenance)


def forecast_stint(model: StintModel, *, tyre_age_laps: float, horizon_laps: int) -> PaceForecast:
    if (not isinstance(model, StintModel) or not _finite(tyre_age_laps)
            or type(horizon_laps) is not int or not 1 <= horizon_laps <= 100
            or tyre_age_laps + horizon_laps > 500):
        raise ValueError('invalid stint forecast state/horizon')
    ages = tyre_age_laps + np.arange(1, horizon_laps + 1)
    parameters = np.asarray(model.bootstrap_parameters)
    rng = np.random.default_rng(model.seed)
    draws = parameters[:, :1] + parameters[:, 1:] * ages
    draws += rng.choice(model.residuals_s, draws.shape, replace=True)
    median = model.intercept_s + model.slope_s_per_lap * ages
    if not np.all(np.isfinite(draws)) or np.any(draws <= 0) or np.any(median <= 0):
        raise ValueError('unsupported extrapolation produces nonpositive lap pace')
    low, high = np.quantile(draws, (.05, .95), axis=0)
    return PaceForecast(tuple(float(v) for v in median), tuple(float(v) for v in np.minimum(low, median)),
                        tuple(float(v) for v in np.maximum(high, median)))
