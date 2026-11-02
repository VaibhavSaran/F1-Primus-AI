"""Explicit local CPU research inference; never downloads from request/UI paths.

Only official safetensors and a pinned upstream TabFM source checkout are used.
TimesFM's ordering is a pace heuristic, not a calibrated race-win prediction.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import gc
from hashlib import sha256
import importlib
import importlib.util
import importlib.metadata
import json
from math import isfinite
from pathlib import Path
import sys
from typing import Sequence

import numpy as np
import pandas as pd

from ml.features import FEATURE_COLUMNS

DEFAULT_ROOT = Path(__file__).resolve().parents[1] / '.local' / 'foundation_models'
TABFM_SOURCE_REVISION = 'fbb665569425fd2f490c6576b3af967876fe11ff'
REVISIONS = {'tabfm': '77cb9cc1b4fd3a9c77fbb9552c218200bb4dab83',
             'timesfm': '1d952420fba87f3c6dee4f240de0f1a0fbc790e3'}
MIN_FREE_BYTES = {'tabfm': 1_500_000_000, 'timesfm': 1_200_000_000}


class FoundationUnavailable(RuntimeError):
    """An explicit dependency, checkpoint or resource gate; no fallback model."""


def trim_cpu_working_set() -> dict:
    """Windows operator helper, only between synchronous inference calls.

    Release this process's resident pages, including reclaimable mapped weights.
    Tensor contents and mappings survive; future accesses can incur page faults.
    This never targets another process or changes a memory admission threshold.
    """
    import psutil
    before = psutil.virtual_memory().available
    result = dict(supported=sys.platform == 'win32', trimmed=False,
                  before_available_bytes=before, after_available_bytes=before, reason='')
    if not result['supported']:
        result['reason'] = 'Working-set trimming is available only on Windows'
        return result
    try:
        import ctypes
        from ctypes import wintypes
        gc.collect()
        kernel = ctypes.WinDLL('kernel32', use_last_error=True)
        psapi = ctypes.WinDLL('psapi', use_last_error=True)
        current = kernel.GetCurrentProcess
        current.argtypes = []
        current.restype = wintypes.HANDLE
        empty = psapi.EmptyWorkingSet
        empty.argtypes = [wintypes.HANDLE]
        empty.restype = wintypes.BOOL
        if not empty(current()):
            raise ctypes.WinError(ctypes.get_last_error())
        result['trimmed'] = True
    except (OSError, AttributeError) as error:
        result['reason'] = str(error)
    result['after_available_bytes'] = psutil.virtual_memory().available
    return result


def _utc(value) -> datetime:
    if isinstance(value, str):
        value = datetime.fromisoformat(value.replace('Z', '+00:00'))
    if not isinstance(value, datetime) or value.tzinfo is None or value.utcoffset() is None:
        raise ValueError('an aware UTC timestamp is required')
    return value.astimezone(timezone.utc)


def checkpoint_info(model_id: str, root: Path | str = DEFAULT_ROOT, *, verify_hashes=False) -> dict:
    """Cheap readiness/provenance for UI; full integrity check runs once at load."""
    if model_id not in REVISIONS:
        raise ValueError('unknown foundation model')
    model_root = Path(root).resolve() / model_id
    try:
        manifest = json.loads((model_root / 'manifest.json').read_text(encoding='utf-8'))
        if manifest['revision'] != REVISIONS[model_id] or manifest['model_id'] != model_id:
            raise ValueError('unexpected checkpoint revision')
        required = ({'regression/config.json', 'regression/model.safetensors', 'LICENSE'}
                    if model_id == 'tabfm' else {'config.json', 'model.safetensors'})
        if {item['path'] for item in manifest['files']} != required:
            raise ValueError('checkpoint manifest has unexpected files')
        for item in manifest['files']:
            path = model_root / item['path']
            if path.is_symlink() or path.stat().st_size != item['bytes']:
                raise ValueError('checkpoint file missing or changed')
            if verify_hashes:
                digest = sha256()
                with path.open('rb') as stream:
                    for chunk in iter(lambda: stream.read(1024 * 1024), b''):
                        digest.update(chunk)
                if digest.hexdigest() != item['sha256']:
                    raise ValueError('checkpoint SHA256 mismatch')
        if any(importlib.util.find_spec(name) is None for name in ('torch', 'safetensors', 'timesfm')):
            raise ValueError('optional CPU model dependencies are not installed')
        info = dict(manifest, checkpoint_ready=True, available=True, reason='',
                    runtime='primusEnv CPU; local files only',
                    weights_license=('non-commercial, non-production' if model_id == 'tabfm'
                                     else 'Apache-2.0'),
                    adapter_version='foundation-runtime-v1',
                    adapter_sha256=sha256(Path(__file__).read_bytes()).hexdigest(),
                    package_versions={name: importlib.metadata.version(name)
                                      for name in ('torch', 'timesfm', 'numpy', 'safetensors')})
        if model_id == 'tabfm':
            source = Path(root).resolve() / 'source' / f'tabfm-{TABFM_SOURCE_REVISION}'
            if not (source / 'tabfm' / '__init__.py').is_file():
                raise ValueError('pinned TabFM source checkout is missing')
            info.update(source_revision=TABFM_SOURCE_REVISION,
                        runtime_support=('Python requirement satisfied; pinned official PyTorch source'
                                         if sys.version_info >= (3,11) else
                                         'unsupported upstream Python 3.10 PyTorch-only experiment'),
                        compute_dtype='float32', weight_storage='read-only safetensors memory mapping')
        return info
    except (OSError, ValueError, KeyError, TypeError) as error:
        return dict(model_id=model_id, available=False, checkpoint_ready=False, reason=str(error),
                    revision=REVISIONS[model_id], adapter_version='foundation-runtime-v1')


def _resource_gate(model_id):
    import psutil
    free = psutil.virtual_memory().available
    if free < MIN_FREE_BYTES[model_id]:
        raise FoundationUnavailable(
            f'{model_id} needs at least {MIN_FREE_BYTES[model_id] / 1e9:.1f} GB free RAM for '
            f'bounded CPU loading; only {free / 1e9:.1f} GB is available. '
            'Close unused applications and retry the operator job; no fallback was run.')


def _ranking(scores, driver_ids, event_id, model_id, units, interpretation):
    values = np.asarray(scores, dtype=float)
    if values.shape != (len(driver_ids),) or not np.isfinite(values).all():
        raise ValueError('foundation model produced invalid or incomplete output')
    order = sorted(range(len(driver_ids)), key=lambda i: (values[i], driver_ids[i]))
    return [dict(event_id=event_id, driver_id=driver_ids[i], prediction_score=float(values[i]),
                 predicted_rank=rank, model_id=model_id, score_units=units,
                 interpretation=interpretation) for rank, i in enumerate(order, 1)]


class _Adapter:
    name: str

    def __init__(self, root: Path | str = DEFAULT_ROOT):
        self.root = Path(root).resolve()
        self._model = None

    def checkpoint_info(self):
        return checkpoint_info(self.name, self.root)

    def _verify(self):
        _resource_gate(self.name)
        info = checkpoint_info(self.name, self.root, verify_hashes=True)
        if not info['available']:
            raise FoundationUnavailable(info['reason'])
        return info

    def close(self):
        self._model = None
        gc.collect()


class TabFMAdapter(_Adapter):
    """Actual upstream in-context regression; context labels must predate cutoff."""
    name = 'tabfm'

    def _estimator(self):
        if self._model is None:
            self._verify()
            source = self.root / 'source' / f'tabfm-{TABFM_SOURCE_REVISION}'
            if str(source) not in sys.path:
                sys.path.insert(0, str(source))
            import torch
            torch.set_num_threads(2)
            from tabfm.src.pytorch.model import TabFM
            from safetensors import safe_open
            config = json.loads((self.root / 'tabfm/regression/config.json').read_text(encoding='utf-8'))
            # No 6.6GB heap copy: retain read-only FP32 safetensor mappings. The OS
            # can reclaim weight pages on this small-RAM CPU host. FP32 is the
            # upstream debugging compute path; latency can include disk paging.
            with torch.device('meta'):
                model = TabFM(**config)
            with safe_open(str(self.root / 'tabfm/regression/model.safetensors'),
                           framework='pt', device='cpu') as tensors:
                expected = set(model.state_dict())
                if set(tensors.keys()) != expected:
                    raise FoundationUnavailable('TabFM checkpoint keys do not match pinned source')
                for key in tensors.keys():
                    tensor = tensors.get_tensor(key)
                    model.load_state_dict({key: tensor}, strict=False, assign=True)
            self._model = model.eval()
        from tabfm import TabFMRegressor
        return TabFMRegressor(model=self._model, n_estimators=1, max_num_rows=100,
                              max_num_features=len(FEATURE_COLUMNS), random_state=42)

    def predict(self, context_rows, target_rows, *, feature_columns: Sequence[str],
                cutoff_utc, target_event_id: str):
        cutoff = _utc(cutoff_utc)
        context, targets = list(context_rows), list(target_rows)
        columns = tuple(feature_columns)
        if (not columns or len(columns) != len(set(columns)) or
                not set(columns).issubset(FEATURE_COLUMNS)):
            raise ValueError('feature_columns must be a unique subset of ML feature schema')
        if not 2 <= len(context) <= 20_000 or not 1 <= len(targets) <= 32:
            raise ValueError('bounded context and nonempty complete target field are required')
        identities = set()
        for row in context:
            key = row.get('event_id'), row.get('driver_id')
            if (not all(isinstance(v, str) and v for v in key) or key in identities or
                    key[0] == target_event_id or
                    _utc(row.get('outcome_available_at_utc')) >= cutoff):
                raise ValueError('context must contain unique, prior-only labelled race rows')
            identities.add(key)
            label = row.get('actual_position')
            if isinstance(label, bool) or not isinstance(label, (int, float)) or not isfinite(label) or label < 1:
                raise ValueError('context needs finite positive finishing-position labels')
        driver_ids = [r.get('driver_id') for r in targets]
        if (any(not isinstance(d, str) or not d for d in driver_ids) or
                len(set(driver_ids)) != len(driver_ids) or
                any(r.get('event_id') != target_event_id for r in targets)):
            raise ValueError('target must contain unique drivers from the requested race')
        # Most recent context, never earliest-only history or target labels.
        context.sort(key=lambda r: (_utc(r['outcome_available_at_utc']), r['event_id'], r['driver_id']))
        context = context[-100:]
        x = pd.DataFrame([{c: r.get(c) for c in columns} for r in context], columns=columns)
        target = pd.DataFrame([{c: r.get(c) for c in columns} for r in targets], columns=columns)
        estimator = self._estimator()
        estimator.fit(x, np.array([r['actual_position'] for r in context], dtype=float))
        scores = estimator.predict(target)
        return _ranking(scores, driver_ids, target_event_id, 'tabfm', 'ordinal_position',
                        'in_context_finishing_position_regression')


@dataclass(frozen=True)
class PaceSeries:
    driver_id: str
    event_id: str
    session_code: str
    completed_at_utc: datetime | str
    values: tuple[float, ...]


class TimesFMAdapter(_Adapter):
    """TimesFM2.5 CPU, median next-lap pace -> explicit uncalibrated ordering."""
    name = 'timesfm'

    def checkpoint_info(self):
        return dict(super().checkpoint_info(), minimum_context_laps=3,
                    recommended_context_laps=32, max_context_laps=256, forecast_horizon=5,
                    context_warning=('Contexts shorter than 32 laps are padded by the upstream '
                                     'model. Short-sequence pace ranks require separate evaluation.'),
                    model_version='2.5-200m', ranking_definition='median future clean lap pace')

    def _load(self):
        if self._model is None:
            self._verify()
            import torch
            import timesfm
            torch.set_num_threads(2)
            from safetensors.torch import load_file
            # Local safetensors mapping avoids Hub calls and a duplicate 925MB model.
            with torch.device('meta'):
                model = timesfm.TimesFM_2p5_200M_torch(torch_compile=False)
            model.model.load_state_dict(load_file(str(self.root / 'timesfm/model.safetensors')),
                                        strict=True, assign=True)
            model.model.eval()
            model.compile(timesfm.ForecastConfig(max_context=256, max_horizon=32,
                          normalize_inputs=True, use_continuous_quantile_head=False,
                          force_flip_invariance=False, infer_is_positive=True,
                          fix_quantile_crossing=True, per_core_batch_size=1))
            self._model = model
        return self._model

    def predict(self, series: Sequence[PaceSeries], *, cutoff_utc, target_event_id: str,
                driver_ids: Sequence[str], horizon: int = 5):
        cutoff = _utc(cutoff_utc)
        ids = list(driver_ids)
        if (not 1 <= len(ids) <= 32 or len(set(ids)) != len(ids) or
                any(not isinstance(d, str) or not d for d in ids) or
                isinstance(horizon, bool) or not isinstance(horizon, int) or not 1 <= horizon <= 32):
            raise ValueError('unique driver field and bounded positive horizon are required')
        grouped = {driver: [] for driver in ids}
        for item in series:
            if (not isinstance(item, PaceSeries) or item.driver_id not in grouped or
                    item.session_code not in ('FP1', 'FP2', 'FP3', 'S', 'R') or
                    item.event_id == target_event_id and item.session_code == 'R' or
                    _utc(item.completed_at_utc) >= cutoff):
                raise ValueError('pace evidence must be completed before forecast and exclude target race')
            values = np.asarray(item.values, dtype=float)
            if values.ndim != 1 or not 1 <= len(values) <= 20_000 or not np.isfinite(values).all() or (values <= 0).any():
                raise ValueError('pace series must contain bounded positive finite lap times')
            grouped[item.driver_id].append(item)
        arrays = []
        for driver in ids:
            ordered = sorted(grouped[driver], key=lambda s: (_utc(s.completed_at_utc), s.event_id,
                                                           s.session_code, len(s.values), tuple(s.values)))
            if not ordered:
                raise ValueError(f'complete driver coverage required: no pace for {driver}')
            # One latest session per driver avoids stitching unrelated race regimes.
            values = np.asarray(ordered[-1].values[-256:], dtype=np.float32)
            if len(values) < 3:
                raise ValueError(f'at least three prior contiguous clean laps required for {driver}')
            arrays.append(values)
        forecasts, _ = self._load().forecast(horizon=horizon, inputs=arrays)
        forecasts = np.asarray(forecasts, dtype=float)
        if forecasts.shape != (len(ids), horizon) or not np.isfinite(forecasts).all() or (forecasts <= 0).any():
            raise ValueError('TimesFM returned invalid or incomplete pace forecasts')
        return _ranking(np.median(forecasts, axis=1), ids, target_event_id, 'timesfm', 'seconds',
                        'pace_based_ranking_heuristic')
