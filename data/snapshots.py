"""Content-addressed JSON snapshots with integrity checks and no pickle execution."""
from __future__ import annotations

from dataclasses import fields, is_dataclass
from datetime import datetime
from enum import Enum
from hashlib import sha256
import json
from pathlib import Path
import re
import tempfile

from data.contracts import DataSnapshot, DriverSessionRecord, EventKey, SourceRecord, PredictionRequest
from data.news import NewsArticle, NewsResult
from data.weather import WeatherRecord, WeatherResult
from data.sessions import SessionRecord, EventInfo

_TYPES = {cls.__name__: cls for cls in (DataSnapshot, DriverSessionRecord, EventKey, SourceRecord, PredictionRequest,
                                     NewsArticle, NewsResult, WeatherRecord, WeatherResult, SessionRecord, EventInfo)}


def _encode(value):
    if isinstance(value, datetime):
        return {'_datetime': value.isoformat()}
    if isinstance(value, Enum):
        return value.value
    if is_dataclass(value):
        if type(value).__name__ not in _TYPES or _TYPES[type(value).__name__] is not type(value):
            raise ValueError('unsupported snapshot record')
        return {'_record': type(value).__name__, 'fields': {field.name: _encode(getattr(value, field.name)) for field in fields(value)}}
    if isinstance(value, dict):
        return {key: _encode(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return {'_tuple': [_encode(item) for item in value]}
    if isinstance(value, list):
        return [_encode(item) for item in value]
    return value


def _decode(value):
    if isinstance(value, list):
        return [_decode(item) for item in value]
    if isinstance(value, dict):
        if set(value) == {'_tuple'}:
            return tuple(_decode(item) for item in value['_tuple'])
        if set(value) == {'_datetime'}:
            return datetime.fromisoformat(value['_datetime'])
        if set(value) == {'_record', 'fields'}:
            cls = _TYPES.get(value['_record'])
            if cls is None:
                raise ValueError('unsupported snapshot record')
            return cls(**{key: _decode(item) for key, item in value['fields'].items()})
        return {key: _decode(item) for key, item in value.items()}
    return value


def _canonical(payload) -> bytes:
    return json.dumps(payload, sort_keys=True, separators=(',', ':'), allow_nan=False).encode('utf-8')


def save_snapshot(snapshot: DataSnapshot, directory: str | Path) -> Path:
    if not re.fullmatch(r'[A-Za-z0-9_-]{1,100}', snapshot.snapshot_id):
        raise ValueError('snapshot_id must be a safe identifier')
    payload = _encode(snapshot)
    digest = sha256(_canonical(payload)).hexdigest()
    root = Path(directory)
    root.mkdir(parents=True, exist_ok=True)
    path = root / f'{snapshot.snapshot_id}-{digest}.json'
    envelope = {'schema_version': 1, 'sha256': digest, 'payload': payload}
    if path.exists():
        load_snapshot(path)
        return path
    # Replace only a complete JSON file; readers never see a partially written artifact.
    with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', dir=root, delete=False) as handle:
        json.dump(envelope, handle, sort_keys=True, allow_nan=False)
        temporary = Path(handle.name)
    temporary.replace(path)
    return path


def load_snapshot(path: str | Path) -> DataSnapshot:
    try:
        envelope = json.loads(Path(path).read_text(encoding='utf-8'))
        if envelope['schema_version'] != 1:
            raise ValueError('unsupported snapshot schema')
        digest = sha256(_canonical(envelope['payload'])).hexdigest()
        if digest != envelope['sha256']:
            raise ValueError('snapshot integrity mismatch')
        result = _decode(envelope['payload'])
        if not isinstance(result, DataSnapshot):
            raise ValueError('snapshot payload is not a DataSnapshot')
        return result
    except (KeyError, TypeError, json.JSONDecodeError) as exc:
        raise ValueError('invalid snapshot envelope') from exc
