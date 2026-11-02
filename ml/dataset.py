"""Join future outcome labels to frozen past-only features, never to feature inputs."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Iterable

from data.contracts import DataSnapshot, require_utc
from ml.features import FEATURE_COLUMNS, HistoryView, build_features


@dataclass(frozen=True)
class OutcomeRecord:
    event_id: str
    driver_id: str
    position: int | None
    event_time_utc: datetime
    available_at_utc: datetime
    status: str = 'Finished'

    def __post_init__(self):
        object.__setattr__(self, 'event_time_utc', require_utc(self.event_time_utc))
        object.__setattr__(self, 'available_at_utc', require_utc(self.available_at_utc))
        if self.available_at_utc < self.event_time_utc:
            raise ValueError('outcome cannot be available before event')
        if self.position is not None and (type(self.position) is not int or self.position < 1):
            raise ValueError('position must be a positive integer or None')
        if not isinstance(self.status, str):
            raise ValueError('status must be text')
        object.__setattr__(self, 'status', self.status.strip().upper())
        if not self.event_id or not self.driver_id:
            raise ValueError('event_id and driver_id are required')


@dataclass(frozen=True)
class DatasetBundle:
    rows: tuple[dict[str, Any], ...]
    feature_columns: tuple[str, ...] = FEATURE_COLUMNS
    schema_version: str = 'dataset-v1'


def build_dataset(snapshots: Iterable[DataSnapshot], outcomes: Iterable[OutcomeRecord],
                  *, mode=None, history: HistoryView | Iterable[dict] = ()) -> DatasetBundle:
    history = history if isinstance(history, HistoryView) else HistoryView(tuple(history))
    labels = {}
    for outcome in outcomes:
        key = (outcome.event_id, outcome.driver_id)
        if key in labels:
            raise ValueError('duplicate outcome event/driver')
        labels[key] = outcome
    rows, seen = [], set()
    for snapshot in snapshots:
        for feature_row in build_features(snapshot, history, mode=mode).rows:
            key = (feature_row['event_id'], feature_row['driver_id'])
            if key in seen:
                raise ValueError('duplicate dataset event/driver snapshot')
            seen.add(key)
            outcome = labels.get(key)
            row = dict(feature_row, snapshot_id=snapshot.snapshot_id)
            row.update(actual_position=outcome.position if outcome else None,
                       event_time_utc=outcome.event_time_utc if outcome else None,
                       outcome_available_at_utc=outcome.available_at_utc if outcome else None,
                       status=outcome.status if outcome else 'pending',
                       outcome_status='completed' if outcome and (outcome.position is not None or outcome.status in {'DNS', 'DSQ', 'DNF'}) else 'pending',
                       eligible=bool(outcome and outcome.position is not None and outcome.status not in {'DNS', 'DSQ'}))
            rows.append(row)
    return DatasetBundle(tuple(rows))
