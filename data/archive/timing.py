"""Read bounded, verified retained timing parser frames without general pickle loads."""
from datetime import timedelta
from hashlib import sha256
from io import BytesIO
from pathlib import Path
import pickle
import pickletools
import stat

import numpy as np
import pandas as pd
from pandas._libs.arrays import __pyx_unpickle_NDArrayBacked
from pandas._libs.internals import _unpickle_block
from pandas.core.arrays.timedeltas import TimedeltaArray
from pandas.core.indexes.base import _new_Index
from pandas.core.internals.managers import BlockManager


MAX_TIMING_BYTES = 32 * 1024 * 1024
MAX_TIMING_ROWS = 200_000
COLUMNS = ['driver_id', 'session_time_s', 'position', 'gap_to_leader_raw', 'interval_to_ahead_raw']
# Parser version 15 is the retained FastF1 schema audited here, not SDK release 3.x.
PARSER_VERSION = 15
_GLOBALS = {
    ('pandas.core.frame', 'DataFrame'): pd.DataFrame,
    ('pandas.core.internals.managers', 'BlockManager'): BlockManager,
    ('pandas._libs.internals', '_unpickle_block'): _unpickle_block,
    ('pandas._libs.arrays', '__pyx_unpickle_NDArrayBacked'): __pyx_unpickle_NDArrayBacked,
    ('pandas.core.arrays.timedeltas', 'TimedeltaArray'): TimedeltaArray,
    ('pandas.core.indexes.base', '_new_Index'): _new_Index,
    ('pandas.core.indexes.base', 'Index'): pd.Index,
    ('pandas.core.indexes.range', 'RangeIndex'): pd.RangeIndex,
    ('numpy', 'ndarray'): np.ndarray,
    ('numpy', 'dtype'): np.dtype,
    ('numpy._core.multiarray', '_reconstruct'): np._core.multiarray._reconstruct,
    ('numpy.core.multiarray', '_reconstruct'): np._core.multiarray._reconstruct,
    ('builtins', 'slice'): slice,
    ('datetime', 'timedelta'): timedelta,
}


class _TimingUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        try:
            return _GLOBALS[module, name]
        except KeyError:
            raise ValueError('Unsupported retained timing pickle global') from None

    def persistent_load(self, pid):
        raise ValueError('Retained timing pickle cannot reference external objects')


def _linked(path):
    info = path.lstat()
    return stat.S_ISLNK(info.st_mode) or bool(getattr(info, 'st_file_attributes', 0) & 0x400)


def retained_stream(root, source):
    """Only linked, content-addressed parser files under raw/ are eligible."""
    digest = source['payload_sha256']
    if (source['provider'] != 'fastf1_parser_cache' or not isinstance(digest, str)
            or len(digest) != 64 or any(c not in '0123456789abcdef' for c in digest)):
        raise ValueError('Untrusted timing source')
    relative = Path(source['raw_path'])
    expected = Path('raw') / 'fastf1_parser_cache' / digest / 'payload.bin'
    if relative != expected:
        raise ValueError('Untrusted timing source path')
    path = root
    for part in relative.parts:
        path = path / part
        if _linked(path):
            raise ValueError('Timing source path must not contain links')
    if not path.resolve().is_relative_to((root / 'raw').resolve()) or not path.is_file():
        raise ValueError('Untrusted timing source path')
    if path.stat().st_size > MAX_TIMING_BYTES:
        raise ValueError('Retained timing source exceeds byte limit')
    with path.open('rb') as stream:
        payload = stream.read(MAX_TIMING_BYTES + 1)
    if len(payload) > MAX_TIMING_BYTES or sha256(payload).hexdigest() != digest:
        raise ValueError('Retained timing source hash mismatch or byte limit')
    try:
        # Extension registry entries can bypass find_class after being cached.
        if any(op.name in ('EXT1', 'EXT2', 'EXT4') for op, _, _ in pickletools.genops(payload)):
            raise ValueError('Retained timing pickle extensions are forbidden')
        cached = _TimingUnpickler(BytesIO(payload)).load()
    except pickle.UnpicklingError as exc:
        raise ValueError('Invalid retained timing pickle') from exc
    if type(cached) is not dict or cached.get('version') != PARSER_VERSION:
        raise ValueError('Unsupported retained timing parser version')
    data = cached.get('data')
    if not isinstance(data, (tuple, list)) or len(data) != 3 or type(data[1]) is not pd.DataFrame:
        raise ValueError('Invalid retained timing payload')
    frame = data[1]
    if len(frame) > MAX_TIMING_ROWS or not {'Time', 'Driver', 'Position', 'GapToLeader', 'IntervalToPositionAhead'}.issubset(frame):
        raise ValueError('Invalid retained timing columns or row limit')
    if not pd.api.types.is_timedelta64_dtype(frame.Time):
        raise ValueError('Retained timing clock must be session timedelta')
    return frame


def _text(value):
    return value.strip() if isinstance(value, str) and 0 < len(value.strip()) <= 64 else None


def normalize_stream(frame, entries):
    """Preserve t0-relative clocks and exact session-entry driver number mapping."""
    numbers = {row['driver_number']: row['driver_id'] for row in entries
               if isinstance(row.get('driver_number'), str) and row['driver_number'] not in ('0', '65535')}
    result = pd.DataFrame({
        'driver_id': frame.Driver.map(numbers),
        'session_time_s': frame.Time.dt.total_seconds(),
        'position': pd.to_numeric(frame.Position, errors='coerce'),
        'gap_to_leader_raw': frame.GapToLeader.map(_text),
        'interval_to_ahead_raw': frame.IntervalToPositionAhead.map(_text),
    })
    result = result[result.driver_id.notna() & result.session_time_s.notna()
                    & (result.session_time_s >= 0) & np.isfinite(result.session_time_s)].copy()
    valid = result.position.between(1, len(entries)) & (result.position % 1 == 0)
    result['position'] = result.position.where(valid).astype('Int64')
    return result.sort_values(['session_time_s', 'driver_id'], kind='stable').reset_index(drop=True)


def lap_positions(laps):
    rows = []
    for row in laps:
        position, clock = row.get('lap_position'), row.get('lap_end_clock_ns')
        if (position is None or clock is None or pd.isna(position) or pd.isna(clock)
                or not np.isfinite(position) or position < 1 or position % 1 or clock < 0):
            continue
        rows.append(dict(driver_id=row['driver_id'], session_time_s=clock / 1e9,
                         position=int(position), gap_to_leader_raw=None, interval_to_ahead_raw=None))
    return pd.DataFrame(rows, columns=COLUMNS)
