"""SQLite admission ledger shared by workers, with integer micro-unit costs.

Use one server-owned local disk path, never session-specific or network SQLite.
Admission/start/checkpoint are mandatory at direct job/tool entrypoints. Callers
must bound execution to the returned deadline; a lease cannot terminate Python
or guarantee cancellation/refunds at an upstream provider.
"""
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
import sqlite3
import time
from typing import Callable, Optional

from security.access import Principal, authorize, oidc_principal


class LimitRejected(RuntimeError):
    def __init__(self, code: str, reset_at: Optional[float] = None):
        self.code, self.reset_at = code, reset_at
        super().__init__(code)


@dataclass(frozen=True)
class LimitPolicy:
    per_minute: int = 6
    per_day: int = 100
    global_active: int = 5
    queue_size: int = 10
    deadline_seconds: int = 45
    tools: int = 3
    searches: int = 1
    retries: int = 1
    message_chars: int = 4000
    context_tokens: int = 12000
    output_tokens: int = 1000

    def __post_init__(self):
        for name, value in vars(self).items():
            if type(value) is not int or value < (0 if name in ('queue_size', 'retries', 'searches', 'tools') else 1):
                raise ValueError('invalid_limit_policy')


@dataclass(frozen=True)
class Lease:
    request_id: str
    owner: str
    status: str
    deadline: float
    replayed: bool = False


def _amount(value):
    if type(value) is not int or not 0 <= value <= 10**15:
        raise LimitRejected('invalid_request')
    return value


class SQLiteLimiter:
    def __init__(self, path, *, policy: LimitPolicy = LimitPolicy(), clock: Callable = time.time,
                 oidc_verifier=None):
        self.path = str(Path(path))
        self.policy, self.clock, self.oidc_verifier = policy, clock, oidc_verifier
        try:
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)
            with self._transaction() as db:
                db.execute('CREATE TABLE IF NOT EXISTS settings (name TEXT PRIMARY KEY, value INTEGER NOT NULL)')
                db.execute('''CREATE TABLE IF NOT EXISTS requests (
                    request_id TEXT PRIMARY KEY, owner TEXT NOT NULL, fingerprint TEXT NOT NULL,
                    action TEXT NOT NULL, paid INTEGER NOT NULL, public INTEGER NOT NULL,
                    reserved INTEGER NOT NULL, actual INTEGER, created REAL NOT NULL,
                    deadline REAL NOT NULL, status TEXT NOT NULL,
                    tools INTEGER NOT NULL DEFAULT 0, searches INTEGER NOT NULL DEFAULT 0,
                    retries INTEGER NOT NULL DEFAULT 0)''')
                db.execute("CREATE UNIQUE INDEX IF NOT EXISTS pending_owner ON requests(owner) WHERE status IN ('active','queued')")
                db.execute('CREATE INDEX IF NOT EXISTS owner_created ON requests(owner,created)')
        except (OSError, sqlite3.Error):
            raise LimitRejected('ledger_unavailable') from None

    @contextmanager
    def _transaction(self):
        db = None
        try:
            db = sqlite3.connect(self.path, timeout=0.25, isolation_level=None)
            db.row_factory = sqlite3.Row
            db.execute('BEGIN IMMEDIATE')
            yield db
            db.commit()
        except sqlite3.Error:
            if db is not None:
                db.rollback()
            raise LimitRejected('ledger_unavailable') from None
        finally:
            if db is not None:
                db.close()

    def authenticate(self, token: str) -> Principal:
        return oidc_principal(token, verifier=self.oidc_verifier)

    def _now(self):
        value = float(self.clock())
        if not 0 < value < 10**11:
            raise LimitRejected('ledger_unavailable')
        return value

    @staticmethod
    def _periods(now):
        dt = datetime.fromtimestamp(now, timezone.utc)
        day = dt.replace(hour=0, minute=0, second=0, microsecond=0).timestamp()
        month = dt.replace(day=1, hour=0, minute=0, second=0, microsecond=0).timestamp()
        return day, month

    @staticmethod
    def _setting(db, name, default=0):
        row = db.execute('SELECT value FROM settings WHERE name=?', (name,)).fetchone()
        return row['value'] if row else default

    @staticmethod
    def _put(db, name, value):
        db.execute('INSERT INTO settings VALUES (?,?) ON CONFLICT(name) DO UPDATE SET value=excluded.value',
                   (name, value))

    @staticmethod
    def _expire(db, now):
        # Expiry releases local slots, never an unresolved monetary reservation.
        db.execute("UPDATE requests SET status='expired' WHERE status IN ('active','queued') AND deadline<=?", (now,))

    @staticmethod
    def _lease(row, replayed=False):
        return Lease(row['request_id'], row['owner'], row['status'], row['deadline'], replayed)

    @staticmethod
    def _owned(db, principal, request_id):
        row = db.execute('SELECT * FROM requests WHERE request_id=?', (request_id,)).fetchone()
        if row is None:
            raise LimitRejected('request_not_found')
        authorize(principal, row['action'], resource_owner=row['owner'])
        return row

    def _enabled(self, db):
        if self._setting(db, 'kill') or self._setting(db, 'circuit_open'):
            raise LimitRejected('disabled')

    def _public_gate(self, db, principal, public):
        if public and (not callable(self.oidc_verifier) or principal.source != 'oidc'
                       or not self._setting(db, 'public_enabled')):
            raise LimitRejected('public_disabled')

    @staticmethod
    def _spend(db, since):
        # Unknown bills survive calendar resets until trusted usage reconciliation.
        row = db.execute('''SELECT COALESCE(SUM(CASE WHEN actual IS NULL THEN reserved
                          WHEN created>=? THEN actual ELSE 0 END),0) AS total
                          FROM requests WHERE paid=1''', (since,)).fetchone()
        return row['total']

    def validate_input(self, message: str, *, input_tokens: int, context_tokens: int = 0,
                       output_tokens: int = 1000):
        counts = (input_tokens, context_tokens, output_tokens)
        if (not isinstance(message, str) or not message.strip() or len(message) > self.policy.message_chars
                or any(type(n) is not int or n < 0 for n in counts)
                or input_tokens + context_tokens + output_tokens > self.policy.context_tokens
                or output_tokens > self.policy.output_tokens):
            raise LimitRejected('input_limit')

    def reserve(self, principal: Principal, request_id: str, fingerprint: str, *, action: str = 'forecast',
                paid: bool = False, public: bool = False, worst_case_cost: int = 0) -> Lease:
        authorize(principal, action)
        if action not in ('forecast', 'chat', 'scenario', 'train', 'promote'):
            raise LimitRejected('invalid_request')
        if (not isinstance(request_id, str) or not 1 <= len(request_id) <= 128
                or not isinstance(fingerprint, str) or not 1 <= len(fingerprint) <= 256
                or any(ord(c) < 32 for c in request_id + fingerprint)
                or type(paid) is not bool or type(public) is not bool):
            raise LimitRejected('invalid_request')
        _amount(worst_case_cost)
        if (paid and worst_case_cost == 0) or (not paid and worst_case_cost != 0):
            raise LimitRejected('invalid_request')
        now = self._now()
        day, month = self._periods(now)
        with self._transaction() as db:
            self._enabled(db)
            self._expire(db, now)
            self._public_gate(db, principal, public)
            existing = db.execute('SELECT * FROM requests WHERE request_id=?', (request_id,)).fetchone()
            if existing:
                wanted = (principal.identity, fingerprint, action, int(paid), int(public), worst_case_cost)
                actual = tuple(existing[k] for k in ('owner', 'fingerprint', 'action', 'paid', 'public', 'reserved'))
                if actual != wanted:
                    raise LimitRejected('request_conflict')
                return self._lease(existing, True)
            for since, limit, code, reset in ((now - 60, self.policy.per_minute, 'minute_limit', now + 60),
                                              (day, self.policy.per_day, 'day_limit', day + 86400)):
                comparison = '>' if code == 'minute_limit' else '>='
                rows = db.execute(f'SELECT created FROM requests WHERE owner=? AND created{comparison}? ORDER BY created',
                                  (principal.identity, since)).fetchall()
                if len(rows) >= limit:
                    if code == 'minute_limit':
                        reset = rows[0]['created'] + 60
                    raise LimitRejected(code, reset)
            if db.execute("SELECT 1 FROM requests WHERE owner=? AND status IN ('active','queued')",
                          (principal.identity,)).fetchone():
                raise LimitRejected('user_busy')
            active = db.execute("SELECT COUNT(*) FROM requests WHERE status='active'").fetchone()[0]
            status = 'active' if active < self.policy.global_active else 'queued'
            if status == 'queued' and db.execute("SELECT COUNT(*) FROM requests WHERE status='queued'").fetchone()[0] >= self.policy.queue_size:
                raise LimitRejected('queue_full')
            if paid:
                daily, monthly = self._setting(db, 'daily_micro'), self._setting(db, 'monthly_micro')
                if not daily or not monthly:
                    raise LimitRejected('budget_unconfigured')
                if self._spend(db, day) + worst_case_cost > daily:
                    unresolved = db.execute('SELECT COALESCE(SUM(reserved),0) FROM requests WHERE paid=1 AND actual IS NULL').fetchone()[0]
                    raise LimitRejected('daily_budget', day + 86400 if unresolved + worst_case_cost <= daily else None)
                if self._spend(db, month) + worst_case_cost > monthly:
                    dt = datetime.fromtimestamp(month, timezone.utc)
                    next_month = dt.replace(year=dt.year + int(dt.month == 12), month=dt.month % 12 + 1).timestamp()
                    unresolved = db.execute('SELECT COALESCE(SUM(reserved),0) FROM requests WHERE paid=1 AND actual IS NULL').fetchone()[0]
                    raise LimitRejected('monthly_budget', next_month if unresolved + worst_case_cost <= monthly else None)
            db.execute('''INSERT INTO requests
                          (request_id,owner,fingerprint,action,paid,public,reserved,created,deadline,status)
                          VALUES (?,?,?,?,?,?,?,?,?,?)''',
                       (request_id, principal.identity, fingerprint, action, int(paid), int(public),
                        worst_case_cost, now, now + self.policy.deadline_seconds, status))
            return self._lease(db.execute('SELECT * FROM requests WHERE request_id=?', (request_id,)).fetchone())

    def start(self, principal: Principal, request_id: str) -> Lease:
        """Promote a queued job once; an already-active lease is a replay."""
        with self._transaction() as db:
            self._enabled(db)
            row = self._owned(db, principal, request_id)
            self._public_gate(db, principal, row['public'])
            now = self._now()
            self._expire(db, now)
            if row['deadline'] <= now:
                raise LimitRejected('deadline_exceeded', row['deadline'])
            if row['status'] == 'queued':
                active = db.execute("SELECT COUNT(*) FROM requests WHERE status='active'").fetchone()[0]
                if active >= self.policy.global_active:
                    raise LimitRejected('busy')
                db.execute("UPDATE requests SET status='active' WHERE request_id=?", (request_id,))
            elif row['status'] != 'active':
                raise LimitRejected('not_active')
            return self._lease(db.execute('SELECT * FROM requests WHERE request_id=?', (request_id,)).fetchone(),
                               replayed=row['status'] == 'active')

    def checkpoint(self, principal: Principal, request_id: str, *, tool: bool = False,
                   search: bool = False, retry: bool = False) -> Lease:
        if any(type(n) is not bool for n in (tool, search, retry)):
            raise LimitRejected('invalid_request')
        with self._transaction() as db:
            self._enabled(db)
            row = self._owned(db, principal, request_id)
            self._public_gate(db, principal, row['public'])
            if row['deadline'] <= self._now():
                raise LimitRejected('deadline_exceeded', row['deadline'])
            if row['status'] != 'active':
                raise LimitRejected('not_active')
            for value, count, maximum, code in ((search, row['searches'], self.policy.searches, 'search_limit'),
                                                (tool or search, row['tools'], self.policy.tools, 'tool_limit'),
                                                (retry, row['retries'], self.policy.retries, 'retry_limit')):
                if value and count >= maximum:
                    raise LimitRejected(code)
            db.execute('UPDATE requests SET tools=tools+?,searches=searches+?,retries=retries+? WHERE request_id=?',
                       (int(tool or search), int(search), int(retry), request_id))
            return self._lease(row)

    def settle(self, principal: Principal, request_id: str, *, actual_cost: Optional[int] = None,
               failed: bool = False):
        if actual_cost is not None:
            _amount(actual_cost)
        if type(failed) is not bool:
            raise LimitRejected('invalid_request')
        with self._transaction() as db:
            row = self._owned(db, principal, request_id)
            if not row['paid'] and actual_cost not in (None, 0):
                raise LimitRejected('invalid_request')
            if row['actual'] is not None and row['actual'] != actual_cost:
                raise LimitRejected('settlement_conflict')
            if row['status'] in ('completed', 'failed', 'expired'):
                if row['actual'] != actual_cost:
                    raise LimitRejected('settlement_conflict')
                return
            status = 'failed' if failed else 'completed'
            db.execute('UPDATE requests SET status=?,actual=? WHERE request_id=?',
                       (status, actual_cost, request_id))

    def reconcile(self, principal: Principal, request_id: str, *, actual_cost: int):
        authorize(principal, 'operate')
        _amount(actual_cost)
        with self._transaction() as db:
            self._expire(db, self._now())
            row = self._owned(db, principal, request_id)
            if not row['paid'] or row['status'] in ('active', 'queued'):
                raise LimitRejected('reconciliation_not_ready')
            if row['actual'] is not None and row['actual'] != actual_cost:
                raise LimitRejected('settlement_conflict')
            db.execute('UPDATE requests SET actual=? WHERE request_id=?', (actual_cost, request_id))

    def configure_budget(self, principal: Principal, *, daily_micro: int, monthly_micro: int):
        authorize(principal, 'operate')
        if not _amount(daily_micro) or not _amount(monthly_micro):
            raise LimitRejected('invalid_request')
        with self._transaction() as db:
            self._put(db, 'daily_micro', daily_micro)
            self._put(db, 'monthly_micro', monthly_micro)

    def set_kill_switch(self, principal: Principal, enabled: bool):
        self._control(principal, 'kill', enabled)

    def set_circuit_breaker(self, principal: Principal, enabled: bool):
        self._control(principal, 'circuit_open', enabled)

    def enable_public(self, principal: Principal, enabled: bool):
        authorize(principal, 'operate')
        if type(enabled) is not bool:
            raise LimitRejected('invalid_request')
        if enabled and not callable(self.oidc_verifier):
            raise LimitRejected('public_disabled')
        with self._transaction() as db:
            if enabled and (not self._setting(db, 'daily_micro') or not self._setting(db, 'monthly_micro')):
                raise LimitRejected('budget_unconfigured')
            self._put(db, 'public_enabled', int(enabled))

    def _control(self, principal, name, enabled):
        authorize(principal, 'operate')
        if type(enabled) is not bool:
            raise LimitRejected('invalid_request')
        with self._transaction() as db:
            self._put(db, name, int(enabled))

    def usage(self, principal: Principal):
        authorize(principal, 'raw_ops')
        now = self._now()
        day, month = self._periods(now)
        with self._transaction() as db:
            self._expire(db, now)
            requests = [dict(row) for row in db.execute('''SELECT request_id,owner,status,created,deadline,
                        reserved,actual,tools,searches,retries FROM requests ORDER BY created DESC LIMIT 100''')]
            reserved = db.execute('SELECT COALESCE(SUM(reserved),0) FROM requests WHERE paid=1 AND actual IS NULL').fetchone()[0]
            return {'daily_used_micro': self._spend(db, day), 'monthly_used_micro': self._spend(db, month),
                    'reserved_micro': reserved, 'daily_budget_micro': self._setting(db, 'daily_micro'),
                    'monthly_budget_micro': self._setting(db, 'monthly_micro'),
                    'disabled': bool(self._setting(db, 'kill') or self._setting(db, 'circuit_open')),
                    'requests': requests}
