from monitoring import TrackingStore, TrackingQueries
from hashlib import sha256


def test_request_health_is_owner_scoped_and_does_not_expose_raw_operations(tmp_path):
    store = TrackingStore(tmp_path / 'tracking.sqlite3')
    for identity, owner, latency, status in [('a', 'alice', 20, 'success'), ('b', 'bob', 900, 'error'), ('c', 'alice', 30, 'error')]:
        store.record_operational_event(dict(event_id=identity, owner_id=owner, status=status,
            occurred_at_utc='2026-11-01T00:00:00+00:00', latency_ms=latency))
    for identity, owner in [('d', 'alice'), ('e', 'bob')]:
        store.record_operational_event(dict(event_id=identity,
            owner_id='chat-' + sha256(owner.encode()).hexdigest()[:24], status='denied',
            occurred_at_utc='2026-11-01T00:00:00+00:00', latency_ms=2))
    query = TrackingQueries(store, owner_id='alice', role='viewer')
    health = query.request_health()
    assert health == dict(requests=3, failures=1, denied=1, p95_latency_ms=30)
    assert 'events' not in health and 'billed_cost' not in health
    assert TrackingQueries(store, role='viewer').request_health()['requests'] == 0
