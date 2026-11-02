from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
import sqlite3
from datetime import datetime, timezone

import pytest

from security.access import AccessDenied, Principal
from security.limits import LimitPolicy, LimitRejected, SQLiteLimiter


OPS = Principal('ops', 'operator')


def limiter(tmp_path, **policy):
    return SQLiteLimiter(tmp_path / 'limits.db', policy=replace(LimitPolicy(), **policy))


def reserve(store, identity='alice', request='r1', **kwargs):
    return store.reserve(Principal(identity, 'ai_user'), request, 'payload', **kwargs)


def test_idempotency_owner_isolation_and_rate_reset(tmp_path):
    store = limiter(tmp_path, per_minute=2)
    first = reserve(store)
    assert first.status == 'active'
    assert reserve(store).replayed
    with pytest.raises(LimitRejected, match='request_conflict'):
        reserve(store, request='r1', action='scenario')
    with pytest.raises(AccessDenied):
        store.checkpoint(Principal('bob', 'ai_user'), 'r1')
    with pytest.raises(LimitRejected, match='request_conflict'):
        reserve(store, identity='bob')
    with pytest.raises(LimitRejected, match='user_busy'):
        reserve(store, request='r2')
    store.settle(Principal('alice', 'ai_user'), 'r1')
    reserve(store, request='r2')
    store.settle(Principal('alice', 'ai_user'), 'r2')
    with pytest.raises(LimitRejected) as error:
        reserve(store, request='r3')
    assert error.value.code == 'minute_limit' and error.value.reset_at > 0


def test_atomic_admission_across_independent_connections(tmp_path):
    store = limiter(tmp_path, global_active=2, queue_size=3)
    path = store.path
    def submit(i):
        try:
            return reserve(SQLiteLimiter(path, policy=store.policy), str(i), str(i)).status
        except LimitRejected as error:
            return error.code
    with ThreadPoolExecutor(max_workers=12) as pool:
        results = list(pool.map(submit, range(12)))
    assert results.count('active') == 2
    assert results.count('queued') == 3
    assert results.count('queue_full') == 7


def test_daily_rate_and_multitab_atomicity(tmp_path):
    store = limiter(tmp_path, per_minute=1000, per_day=2)
    for i in range(2):
        reserve(store, request=str(i))
        store.settle(Principal('alice', 'ai_user'), str(i))
    with pytest.raises(LimitRejected, match='day_limit'):
        reserve(store, request='third')
    fresh = limiter(tmp_path / 'other')
    def submit(i):
        try:
            return reserve(fresh, request=str(i)).status
        except LimitRejected as error:
            return error.code
    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(submit, range(8)))
    assert results.count('active') == 1 and results.count('user_busy') == 7


def test_queued_jobs_need_start_and_expire(tmp_path):
    now = [1700000000.0]
    store = SQLiteLimiter(tmp_path / 'limits.db', policy=LimitPolicy(global_active=1), clock=lambda: now[0])
    reserve(store)
    lease = reserve(store, 'bob', 'r2')
    assert lease.status == 'queued'
    with pytest.raises(LimitRejected, match='not_active'):
        store.checkpoint(Principal('bob', 'ai_user'), 'r2')
    store.settle(Principal('alice', 'ai_user'), 'r1')
    assert store.start(Principal('bob', 'ai_user'), 'r2').status == 'active'
    now[0] += 46
    with pytest.raises(LimitRejected, match='deadline_exceeded'):
        store.checkpoint(Principal('bob', 'ai_user'), 'r2')
    assert reserve(store, 'carol', 'r3').status == 'active'


def test_tool_search_retry_caps_and_message_size(tmp_path):
    store = limiter(tmp_path)
    reserve(store)
    alice = Principal('alice', 'ai_user')
    store.checkpoint(alice, 'r1', search=True)
    with pytest.raises(LimitRejected, match='search_limit'):
        store.checkpoint(alice, 'r1', search=True)
    store.checkpoint(alice, 'r1', tool=True)
    store.checkpoint(alice, 'r1', tool=True)
    with pytest.raises(LimitRejected, match='tool_limit'):
        store.checkpoint(alice, 'r1', tool=True)
    store.checkpoint(alice, 'r1', retry=True)
    with pytest.raises(LimitRejected, match='retry_limit'):
        store.checkpoint(alice, 'r1', retry=True)
    store.validate_input('hello', input_tokens=5, context_tokens=100, output_tokens=1000)
    for kwargs in ({'message': 'x' * 4001, 'input_tokens': 1},
                   {'message': 'x', 'input_tokens': 1, 'context_tokens': 12001},
                   {'message': 'x', 'input_tokens': 1, 'output_tokens': 1001}):
        with pytest.raises(LimitRejected, match='input_limit'):
            store.validate_input(**kwargs)


def test_paid_public_disabled_without_verifier_budget_and_enablement(tmp_path):
    store = limiter(tmp_path)
    with pytest.raises(LimitRejected, match='public_disabled'):
        reserve(store, paid=True, public=True, worst_case_cost=10)
    with pytest.raises(LimitRejected, match='budget_unconfigured'):
        reserve(store, paid=True, worst_case_cost=10)
    store.configure_budget(OPS, daily_micro=100, monthly_micro=1000)
    assert reserve(store, paid=True, worst_case_cost=10).status == 'active'


def test_budget_reservation_is_atomic_and_unknown_cost_never_refunded(tmp_path):
    now = [1700000000.0]
    store = SQLiteLimiter(tmp_path / 'limits.db', clock=lambda: now[0])
    store.configure_budget(OPS, daily_micro=100, monthly_micro=200)
    def submit(i):
        try:
            return reserve(store, str(i), str(i), paid=True, worst_case_cost=60).status
        except LimitRejected as error:
            return error.code
    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(submit, range(8)))
    assert results.count('active') == 1 and results.count('daily_budget') == 7
    usage = store.usage(OPS)
    assert usage['reserved_micro'] == 60
    admitted = usage['requests'][0]
    store.settle(Principal(admitted['owner'], 'ai_user'), admitted['request_id'], failed=True)
    now[0] += 86400 * 40
    assert store.usage(OPS)['daily_used_micro'] == 60
    with pytest.raises(LimitRejected, match='daily_budget'):
        reserve(store, 'new', 'new', paid=True, worst_case_cost=60)
    store.reconcile(OPS, admitted['request_id'], actual_cost=20)
    assert reserve(store, 'new', 'new', paid=True, worst_case_cost=60).status == 'active'


def test_settlement_idempotence_actual_overrun_and_operator_controls(tmp_path):
    store = limiter(tmp_path)
    store.configure_budget(OPS, daily_micro=100, monthly_micro=100)
    reserve(store, paid=True, worst_case_cost=50)
    alice = Principal('alice', 'ai_user')
    store.settle(alice, 'r1', actual_cost=120)
    store.settle(alice, 'r1', actual_cost=120)
    with pytest.raises(LimitRejected, match='settlement_conflict'):
        store.settle(alice, 'r1', actual_cost=0)
    with pytest.raises(LimitRejected, match='daily_budget'):
        reserve(store, 'bob', 'r2', paid=True, worst_case_cost=1)
    for operation in (lambda: store.usage(alice), lambda: store.set_kill_switch(alice, True),
                      lambda: store.configure_budget(alice, daily_micro=1, monthly_micro=1)):
        with pytest.raises(AccessDenied):
            operation()
    store.set_kill_switch(OPS, True)
    with pytest.raises(LimitRejected, match='disabled'):
        reserve(store, 'bob', 'free')


def test_unavailable_ledger_fails_closed_and_malformed_input(tmp_path):
    store = limiter(tmp_path)
    store.path = str(tmp_path / 'missing' / 'limits.db')
    with pytest.raises(LimitRejected, match='ledger_unavailable'):
        reserve(store)
    store = limiter(tmp_path / 'valid')
    for kwargs in ({'worst_case_cost': -1}, {'paid': True, 'worst_case_cost': float('nan')},
                   {'paid': False, 'worst_case_cost': 1}):
        with pytest.raises(LimitRejected, match='invalid_request'):
            reserve(store, **kwargs)
    with pytest.raises(AccessDenied):
        store.reserve(None, 'r1', 'payload')


def test_day_boundary_counts_midnight_requests(tmp_path):
    now = datetime(2026, 10, 1, tzinfo=timezone.utc).timestamp()
    store = SQLiteLimiter(tmp_path / 'limits.db', policy=LimitPolicy(per_day=1), clock=lambda: now)
    reserve(store)
    store.settle(Principal('alice', 'ai_user'), 'r1')
    with pytest.raises(LimitRejected, match='day_limit'):
        reserve(store, request='r2')


def test_monthly_budget_and_reset_metadata(tmp_path):
    now = [datetime(2026, 10, 1, tzinfo=timezone.utc).timestamp()]
    store = SQLiteLimiter(tmp_path / 'limits.db', clock=lambda: now[0])
    store.configure_budget(OPS, daily_micro=100, monthly_micro=100)
    reserve(store, paid=True, worst_case_cost=75)
    store.settle(Principal('alice', 'ai_user'), 'r1', actual_cost=75)
    now[0] += 86400
    with pytest.raises(LimitRejected) as error:
        reserve(store, 'bob', 'r2', paid=True, worst_case_cost=30)
    assert error.value.code == 'monthly_budget'
    assert error.value.reset_at == datetime(2026, 11, 1, tzinfo=timezone.utc).timestamp()


def test_verified_public_identity_explicit_enablement_and_budget(tmp_path):
    store = SQLiteLimiter(tmp_path / 'limits.db', oidc_verifier=lambda token: {'sub': token, 'role': 'ai_user'})
    alice = store.authenticate('alice')
    with pytest.raises(LimitRejected, match='public_disabled'):
        store.reserve(alice, 'r1', 'f', public=True)
    with pytest.raises(LimitRejected, match='budget_unconfigured'):
        store.enable_public(OPS, True)
    store.configure_budget(OPS, daily_micro=100, monthly_micro=100)
    store.enable_public(OPS, True)
    with pytest.raises(LimitRejected, match='public_disabled'):
        reserve(store, public=True)
    assert store.reserve(alice, 'r1', 'f', public=True, paid=True, worst_case_cost=10).status == 'active'
    store.set_circuit_breaker(OPS, True)
    with pytest.raises(LimitRejected, match='disabled'):
        store.checkpoint(alice, 'r1')
    store.set_circuit_breaker(OPS, False)
    store.checkpoint(alice, 'r1')
    store.enable_public(OPS, False)
    with pytest.raises(LimitRejected, match='public_disabled'):
        store.checkpoint(alice, 'r1')


def test_expired_paid_lease_can_be_reconciled_without_other_requests(tmp_path):
    now = [1700000000.0]
    store = SQLiteLimiter(tmp_path / 'limits.db', clock=lambda: now[0])
    store.configure_budget(OPS, daily_micro=100, monthly_micro=100)
    reserve(store, paid=True, worst_case_cost=75)
    now[0] += 46
    store.reconcile(OPS, 'r1', actual_cost=20)
    assert store.usage(OPS)['reserved_micro'] == 0
    assert store.usage(OPS)['requests'][0]['status'] == 'expired'


def test_locked_or_corrupt_ledger_returns_bounded_safe_error(tmp_path):
    store = limiter(tmp_path)
    with sqlite3.connect(store.path) as db:
        db.execute('BEGIN IMMEDIATE')
        with pytest.raises(LimitRejected, match='ledger_unavailable'):
            reserve(store)
    (tmp_path / 'corrupt.db').write_bytes(b'not a database secret-token')
    with pytest.raises(LimitRejected) as error:
        SQLiteLimiter(tmp_path / 'corrupt.db')
    assert str(error.value) == 'ledger_unavailable'


@pytest.mark.parametrize('field,value', [('per_minute', 0), ('global_active', -1),
                                        ('queue_size', -1), ('deadline_seconds', True)])
def test_invalid_policy_fails_closed(field, value):
    with pytest.raises(ValueError):
        replace(LimitPolicy(), **{field: value})


@pytest.mark.parametrize('kwargs', [{'request_id': ''}, {'request_id': 'bad\n'},
                                    {'fingerprint': ''}, {'paid': 1}, {'public': 'yes'},
                                    {'worst_case_cost': True}])
def test_admission_malformed_fields(tmp_path, kwargs):
    store = limiter(tmp_path)
    fields = dict(request_id='r1', fingerprint='f')
    fields.update(kwargs)
    with pytest.raises(LimitRejected, match='invalid_request'):
        store.reserve(Principal('alice', 'ai_user'), **fields)


def test_concurrent_queue_promotion_has_only_one_dispatch_grant(tmp_path):
    store = limiter(tmp_path, global_active=1)
    reserve(store)
    reserve(store, 'bob', 'r2')
    store.settle(Principal('alice', 'ai_user'), 'r1')
    def promote(_):
        return store.start(Principal('bob', 'ai_user'), 'r2').replayed
    with ThreadPoolExecutor(max_workers=6) as pool:
        grants = list(pool.map(promote, range(6)))
    assert grants.count(False) == 1
    assert grants.count(True) == 5


def test_idempotent_replay_after_expiry_does_not_create_new_job(tmp_path):
    now = [1700000000.0]
    store = SQLiteLimiter(tmp_path / 'limits.db', clock=lambda: now[0])
    reserve(store)
    now[0] += 46
    replay = reserve(store)
    assert replay.status == 'expired' and replay.replayed
    with pytest.raises(LimitRejected, match='deadline_exceeded'):
        store.start(Principal('alice', 'ai_user'), 'r1')


def test_sliding_minute_resets_at_recorded_boundary(tmp_path):
    now = [1700000000.0]
    store = SQLiteLimiter(tmp_path / 'limits.db', policy=LimitPolicy(per_minute=1), clock=lambda: now[0])
    reserve(store)
    store.settle(Principal('alice', 'ai_user'), 'r1')
    now[0] += 59
    with pytest.raises(LimitRejected) as error:
        reserve(store, request='r2')
    assert error.value.reset_at == 1700000060.0
    now[0] += 1
    assert reserve(store, request='r2').status == 'active'
