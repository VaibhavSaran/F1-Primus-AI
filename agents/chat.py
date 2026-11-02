"""Private grounded Ask F1 pilot: deterministic, read-only, no provider or fetch.

The trusted server supplies store/history/snapshots and authenticated principals.
Source text is never interpreted as instructions. This deliberately has no LLM
loop, URL fetcher, file reader, code executor, or action-producing tool.
"""
from __future__ import annotations

from collections import OrderedDict
from datetime import datetime, timezone
from hashlib import sha256
import html
import ipaddress
import json
from math import isfinite
import re
from time import perf_counter
import unicodedata
from urllib.parse import urlsplit
from uuid import uuid4

import tiktoken

from data.contracts import require_utc
from data.profiles import build_profiles
from monitoring.queries import TrackingQueries
from security.access import AccessDenied, authorize
from security.limits import LimitRejected


# A byte BPE gives a conservative bound without downloading a model vocabulary.
# Replace only alongside an explicitly configured model and its token budget.
_TOKENS = tiktoken.Encoding(name='primus-byte-v1', pat_str=r'(?s).',
                           mergeable_ranks={bytes([i]): i for i in range(256)}, special_tokens={})
_ID = re.compile(r'[A-Za-z0-9][A-Za-z0-9_.:@-]{0,159}\Z')
_ALIASES = {'NOR': ('norris', 'lando'), 'PIA': ('piastri', 'oscar'),
            'VER': ('verstappen', 'max'), 'HAM': ('hamilton', 'lewis'),
            'LEC': ('leclerc', 'charles'), 'RUS': ('russell', 'george'),
            'ALO': ('alonso', 'fernando'), 'SAI': ('sainz', 'carlos'),
            'GAS': ('gasly',), 'OCO': ('ocon',), 'ALB': ('albon',),
            'STR': ('stroll',), 'TSU': ('tsunoda',), 'HUL': ('hulkenberg',),
            'BEA': ('bearman',), 'ANT': ('antonelli',), 'HAD': ('hadjar',),
            'LAW': ('lawson',), 'BOR': ('bortoleto',), 'COL': ('colapinto',),
            'PER': ('perez', 'sergio'), 'BOT': ('bottas', 'valtteri'), 'VET': ('vettel', 'sebastian'),
            'RIC': ('ricciardo', 'daniel'), 'RAI': ('raikkonen', 'kimi'), 'MAG': ('magnussen', 'kevin'),
            'LAT': ('latifi',), 'GIO': ('giovinazzi',), 'KVY': ('kvyat',), 'GRO': ('grosjean',),
            'SCH': ('schumacher',), 'KUB': ('kubica',), 'AIT': ('aitken',), 'FIT': ('fittipaldi',)}
_TOPIC = re.compile(r'\b(f1|formula\s*(one|1)|races?|racing|drivers?|teams?|grand prix|gp|'
    r'forecast|model|prediction|telemetry|lap|laps|stint|tyre|tyres|tire|tires|pit|undercut|overcut|'
    r'weather|temperature|humidity|wind|pressure|winner|won|results?|classification|cancelled|canceled|retire|retired|dnf|withdraw|withdrawn|profile|'
    r'porpoising|drs|ers|mgu|dirty air|deg|degradation|brake bias|lift and coast|parc ferm[e\u00e9]|'
    r'qualifying|sprint|downforce|aero|grid|safety car|track limits|fia|regulations|'
    r'slipstream|slipstreaming|blue flags?|front wing|rear wing|compound|crossover|wet|tow|'
    r'blistering|graining|lockup|lock-up|understeer|oversteer|rain|sector|sectors|'
    r'floor|diffuser|drag|ballast|torque|power unit|double waved|yellow flags?|red flag|'
    r'mclaren|ferrari|mercedes|red bull|williams|aston martin|alpine|haas|sauber|'
    r'norris|piastri|verstappen|hamilton|leclerc|russell|alonso|sainz|gasly|ocon|albon|'
    r'stroll|tsunoda|hulkenberg|bearman|antonelli|hadjar|lawson|bortoleto|colapinto|'
    r'perez|bottas|vettel|ricciardo|raikkonen|magnussen|latifi|giovinazzi|kvyat|grosjean|schumacher|kubica|aitken|fittipaldi)\b', re.I)
_UNSAFE = re.compile(r'ignore\s+(all\s+)?(previous|prior|above|system)|system\s+prompt|'
    r'developer\s+(message|instructions)|reveal\s+(secrets?|instructions)|api[ _-]*key|'
    r'credentials|password|\.env\b|\bshell\b|\bsql\b|execute\s+(python|code)|'
    r'https?://|file://|javascript:|<script|override\s+(policy|instructions)|'
    r'disable\s+(guard|quota|limit)|bypass\s+(?:the\s+)?(?:safety|guard|policy|rules)|'
    r'(?:unrestricted|jailbreak)|\bDAN\b|(?:hidden|internal)\s+(?:instructions|configuration)|'
    r'read\s+(?:the\s+)?(?:local\s+)?file|[a-z]:\\', re.I)
# Scope is conservative: an F1 keyword cannot launder a second, unrelated task.
# This is defense in depth; the tool allowlist and deterministic renderer remain
# the primary boundary, with no generative execution or user-directed fetch.
_OFF_TOPIC_TASK = re.compile(r'\b(recipe|homework|malware|ransomware|phishing|cryptocurrency|'
    r'stocks?|portfolio|investment|medical|diagnosis|dating|poem|essay)\b|'
    r'\b(?:write|generate|execute|compile)\b.{0,50}\b(?:code|script|program|python|javascript)\b|'
    r'\b(?:translate|create\s+an?\s+image|draw\s+an?\s+(?:image|picture))\b', re.I)

_RACE_ALIASES = (
    ('mexic','mexico','mexican'), ('spanish','spain','barcelona'), ('italian','italy','monza'),
    ('austrian','austria','spielberg'), ('australian','australia','melbourne'),
    ('belgian','belgium','spa'), ('british','britain','silverstone'),
    ('united states','usa','austin'), ('sao paulo','brazil','brazilian'),
    ('japanese','japan','suzuka'), ('chinese','china','shanghai'),
    ('canadian','canada','montreal'), ('monaco','monte carlo'),
    ('saudi','jeddah'), ('dutch','netherlands','zandvoort'),
    ('hungarian','hungary','budapest'), ('azerbaijan','baku'),
    ('portuguese','portugal','portimao'), ('emilia romagna','imola'),
    ('tuscan','tuscany','mugello'), ('russian','russia','sochi'))


def _race_words(value):
    value=value.lower()
    for aliases in _RACE_ALIASES:
        if any(re.search(r'\b'+re.escape(a)+r'\b',value) for a in aliases):
            value+=' '+aliases[0]
    return set(re.findall(r'[a-z0-9]+',value))-{'grand','prix','gp','f1','city'}


def _normalize_message(value):
    # Remove zero-width formatting before checks so invisible separators cannot
    # split control phrases; keep ordinary Unicode names for evidence matching.
    value = unicodedata.normalize('NFKC', value)
    return ''.join(c for c in value if unicodedata.category(c) != 'Cf').lower()



def _time(value):
    if isinstance(value, str):
        value = datetime.fromisoformat(value.replace('Z', '+00:00'))
    return require_utc(value)


def _identifier(value):
    if not isinstance(value, str) or not _ID.fullmatch(value):
        raise ValueError('invalid tool identifier')
    return value


def _count(value):
    return len(_TOKENS.encode(value, disallowed_special=()))


def _label(value):
    return re.sub(r'[^A-Za-z0-9 _.:@-]', '', str(value))[:160]


def safe_markdown(value):
    """Keep vetted FIA links only. No images, raw HTML or arbitrary destinations.

    No destinations are ever fetched here, so DNS/redirect SSRF has no transport.
    The display allowlist also excludes private-address and malicious links.
    """
    value = html.unescape(str(value))
    value = re.sub(r'<[^>]*>', '', value)
    value = re.sub(r'!\[[^\]]*\]\([^)]*\)', '', value)
    def link(match):
        label, target = match.group(1), match.group(2)
        try:
            parsed = urlsplit(target)
            host = parsed.hostname or ''
            try:
                ipaddress.ip_address(host)
                return label
            except ValueError:
                pass
            if (parsed.scheme == 'https' and host in {'www.fia.com', 'fia.com'}
                    and not parsed.username and not parsed.password and parsed.port in {None, 443}
                    and not any(c in target for c in '\\<>\r\n')):
                return f'[{label}]({target})'
        except ValueError:
            pass
        return label
    value = re.sub(r'\[([^\]]*)\]\(([^)]*)\)', link, value)
    value = re.sub(r'(?<!\()\b(?:https?|file|javascript|data):[^\s]+', '', value, flags=re.I)
    return value.replace('<', '&lt;').replace('>', '&gt;')


def _selected(entity, message):
    tokens = set(re.findall(r'[a-z0-9]+', message.lower()))
    words = set(re.findall(r'[a-z0-9]+', entity.lower()))
    return bool(words & tokens or any(name in tokens for name in _ALIASES.get(entity.upper(), ())))


class ChatService:
    """All public chat/tool entrypoints authorize and consult the shared ledger.

    Retain one service per trusted process for bounded duplicate response replay.
    A restarted process denies completed duplicates instead of repeating tools.
    Raw prompts/history are not logged or retained by this service.
    """

    def __init__(self, limiter, store, *, history=(), snapshots=(), archive_context=None, provider=None):
        self.limiter, self.store = limiter, store
        self.history = tuple(history.rows if hasattr(history, 'rows') else history)
        self.snapshots = tuple(snapshots)
        self.archive_context = archive_context
        self.provider = provider
        if len(self.history) > 20000 or len(self.snapshots) > 32:
            raise ValueError('configured evidence limit exceeded')
        self._responses = OrderedDict()

    def execute_tool(self, principal, request_id, name, arguments, *, as_of_utc):
        """Strict allowlist; callable directly only with an active owned lease."""
        authorize(principal, 'chat')
        cutoff = _time(as_of_utc)
        schemas = {'saved_run': {'run_id', 'driver_id'}, 'evaluation': {'run_id'},
                   'catalog': {'season', 'race', 'kind'},
                   'pending_forecast': {'event_id', 'model_id'},
                   'profiles': {'driver_id', 'driver_ids', 'team_id'},
                   'lap_summary': {'event_id', 'driver_id', 'session_id', 'snapshot_id'},
                   'archive': {'event_id', 'session_id', 'driver_id', 'team_id', 'question'}}
        if name not in schemas or not isinstance(arguments, dict) or set(arguments) - schemas[name]:
            raise ValueError('tool or arguments not allowed')
        if name == 'catalog':
            self.limiter.checkpoint(principal, request_id, tool=True)
            return self._catalog(principal, arguments, cutoff)
        if name in {'saved_run', 'evaluation'} and 'run_id' not in arguments:
            raise ValueError('run_id required')
        if name == 'lap_summary' and not {'event_id', 'driver_id'} <= set(arguments):
            raise ValueError('event_id and driver_id required')
        if name == 'archive' and (not {'event_id', 'session_id', 'question'} <= set(arguments)
                or arguments['question'] not in {'results', 'laps', 'tyres', 'weather', 'context', 'profile'}):
            raise ValueError('archive scope and supported question required')
        if name == 'profiles' and 'driver_id' in arguments and 'driver_ids' in arguments:
            raise ValueError('choose driver_id or driver_ids')
        for key, value in arguments.items():
            if key == 'driver_ids':
                if not isinstance(value, list) or not 1 <= len(value) <= 2 or len(set(map(str, value))) != len(value):
                    raise ValueError('driver_ids requires one or two distinct identifiers')
                for identifier in value:
                    _identifier(identifier)
            else:
                _identifier(value)
        self.limiter.checkpoint(principal, request_id, tool=True)
        if name == 'pending_forecast':
            from ml.pending_forecast import cached_pending_forecast
            if set(arguments)!={'event_id','model_id'}:
                raise ValueError('event and model required')
            try:
                saved=cached_pending_forecast(arguments['event_id'],arguments['model_id'],owner_id=principal.identity)
            except ValueError:
                return None
            if _time(saved['as_of_utc'])>cutoff or _time(saved['recorded_at_utc'])>cutoff:
                return None
            return dict({k:saved[k] for k in ('run_id','event_id','model_id','as_of_utc','snapshot_id')},
                ranking=[{k:r[k] for k in ('driver_id','predicted_rank','prediction_score','predicted_pace_s','winner_probability','grid_position') if k in r}
                         for r in saved['ranking'][:32]])
        if name == 'archive':
            if self.archive_context is None:
                return None
            evidence = self.archive_context.query(**arguments, as_of_utc=cutoff)
            if evidence and arguments['question'] == 'profile':
                bundle = evidence['profiles']
                profiles = [p for p in bundle['drivers'] if arguments.get('driver_id') == p['driver_id']]
                profiles += [p for p in bundle['teams'] if arguments.get('team_id') == p['team_id']]
                evidence = dict(reference=bundle['snapshot_id'], season=bundle['season'], rows=[
                    {k: p.get(k) for k in ('display_name', 'entity_id', 'recent_form', 'affiliation',
                        'lineup', 'reliability', 'source_ids', 'source_urls')} for p in profiles[:2]])
                for row in evidence['rows']:
                    row['source_count'] = len(row['source_ids'])
                    row['source_ids'] = row['source_ids'][:8]
                    if row.get('affiliation'):
                        row['affiliation_display_name'] = next((p['display_name'] for p in bundle['teams']
                            if p['team_id'] == row['affiliation']['team_id']), row['affiliation']['team_id'])
            if evidence and arguments['question'] in {'laps', 'tyres', 'results'}:
                evidence['entry_count'] = len(evidence.get('rows', []))
                evidence['rows'] = evidence.get('rows', [])[:24 if self.provider else 6]
            return self._bounded(evidence) if evidence else None
        query = TrackingQueries(self.store, owner_id=principal.identity, role=principal.role)
        if name in {'saved_run', 'evaluation'}:
            saved = query.get_forecast(arguments['run_id'])
            if saved is None or _time(saved['as_of_utc']) > cutoff or _time(saved['recorded_at_utc']) > cutoff:
                return None
            if name == 'saved_run':
                ranking = saved['ranking']
                if arguments.get('driver_id'):
                    ranking = [r for r in ranking if r['driver_id'] == arguments['driver_id']]
                    if not ranking:
                        return None
                return self._bounded(dict(run_id=saved['run_id'], event_id=saved['event_id'], model_id=saved['model_id'],
                            as_of_utc=saved['as_of_utc'], snapshot_id=saved['snapshot_id'],
                            driver_id=arguments.get('driver_id'),
                            ranking=[{k: row[k] for k in ('driver_id', 'predicted_rank', 'prediction_score',
                                      'predicted_pace_s', 'winner_probability', 'grid_position') if k in row}
                                     for row in ranking[:32]]))
            revisions = [r for r in query.outcomes(saved['event_id']) if _time(r['published_at_utc']) <= cutoff]
            latest = max(revisions, key=lambda r: (_time(r['published_at_utc']), r['revision_id'])) if revisions else None
            scores = [r for r in query.score_history(saved['run_id'])
                      if latest and r['revision_id'] == latest['revision_id'] and _time(r['scored_at_utc']) <= cutoff
                      and r.get('evaluation', {}).get('outcome_status') == 'completed']
            if not scores:
                return None
            score = scores[-1]
            return self._bounded(dict(run_id=saved['run_id'], event_id=saved['event_id'],
                        scored_at_utc=score['scored_at_utc'], revision_id=score['revision_id'],
                        metrics={k: v for k, v in score['metrics'].items()
                                 if isinstance(v, (int, float)) and not isinstance(v, bool) and isfinite(v)}))
        if name == 'profiles':
            profiles = build_profiles(self._visible_history(principal), cutoff)
            driver_ids = arguments.get('driver_ids', [arguments.get('driver_id')])
            drivers = [p for p in profiles['drivers'] if p['driver_id'] in driver_ids]
            teams = [p for p in profiles['teams'] if arguments.get('team_id') == p['team_id']]
            fields = {'entity_id', 'kind', 'as_of_utc', 'recent_form', 'reliability', 'affiliation', 'lineup'}
            def compact(profile):
                return dict({k: v for k, v in profile.items() if k in fields},
                            source_ids=profile['source_ids'][:8], source_count=len(profile['source_ids']),
                            source_complete=all(r['source_ids'] for r in profile['lineage']),
                            lineage_reference=profiles['snapshot_id'])
            return self._bounded(dict(snapshot_id=profiles['snapshot_id'], as_of_utc=profiles['as_of_utc'],
                                      drivers=[compact(p) for p in drivers], teams=[compact(p) for p in teams]))
        candidates, denied = [], False
        for snapshot in self.snapshots:
            if (snapshot.event.event_id != arguments['event_id'] or snapshot.as_of_utc > cutoff
                    or (arguments.get('snapshot_id') and snapshot.snapshot_id != arguments['snapshot_id'])):
                continue
            if snapshot.payloads.get('owner_id'):
                try:
                    authorize(principal, 'read', resource_owner=snapshot.payloads['owner_id'],
                              is_public=snapshot.payloads.get('is_public') is True)
                except AccessDenied:
                    denied = True
                    continue
            records = [r for r in snapshot.payloads.get('driver_sessions', ())
                       if r.event == snapshot.event and r.driver_id == arguments['driver_id']
                       and r.available_at_utc <= cutoff and r.source.available_by(cutoff)
                       and (not arguments.get('session_id') or r.session_id == arguments['session_id'])]
            if records:
                candidates.append((snapshot, records))
        if not candidates:
            if denied:
                raise AccessDenied()
            return None
        sessions = {record.session_id for _, records in candidates for record in records}
        if len(sessions) != 1:
            return None
        session_id = _identifier(next(iter(sessions)))
        snapshot, records = max(candidates, key=lambda pair: (pair[0].as_of_utc, pair[0].snapshot_id))
        laps = []
        for record in records:
            for lap in record.values.get('laps', ())[:5000]:
                available = lap.get('available_at_utc')
                end = lap.get('lap_end_utc')
                if available is None or _time(available) > cutoff or end is None or _time(end) > cutoff:
                    continue
                number, seconds = lap.get('lap_number'), lap.get('lap_time_s')
                if (type(number) is int and number > 0 and isinstance(seconds, (int, float))
                        and not isinstance(seconds, bool) and isfinite(seconds) and seconds > 0):
                    laps.append((record.session_id, number, seconds, _identifier(record.source.source_id)))
        unique = {}
        for session, number, seconds, source in laps:
            if (session, number) in unique and unique[(session, number)][0] != seconds:
                raise ValueError('conflicting duplicate lap evidence')
            unique[(session, number)] = (seconds, source)
        if not unique:
            return None
        return self._bounded(dict(snapshot_id=_identifier(snapshot.snapshot_id), event_id=snapshot.event.event_id,
                    snapshot_as_of_utc=snapshot.as_of_utc.isoformat(), session_id=session_id,
                    driver_id=arguments['driver_id'], sample_count=len(unique),
                    mean_lap_time_s=sum(v[0] for v in unique.values()) / len(unique),
                    definition='Mean timestamped observed laps, not clean pace or degradation',
                    source_ids=sorted({v[1] for v in unique.values()})[:8], as_of_utc=cutoff.isoformat()))

    def _bounded(self, payload):
        if _count(json.dumps(payload, sort_keys=True)) > min(16000 if self.provider else 8000, self.limiter.policy.context_tokens):
            return None
        return payload

    def _visible_history(self, principal):
        # Rows without ownership are trusted, curated sporting history supplied by
        # the server. Explicitly owned records always retain owner/public policy.
        return [r for r in self.history if not r.get('owner_id') or r.get('is_public') is True
                or r['owner_id'] == principal.identity or principal.role == 'operator']

    def _catalog(self, principal, arguments, cutoff):
        season, race, kind = arguments.get('season'), arguments.get('race'), arguments.get('kind', 'sessions')
        if (not isinstance(season,str) or not re.fullmatch(r'(?:19|20)\d{2}',season)
                or not isinstance(race,str) or len(race)>100 or kind not in {'sessions','forecasts'}):
            raise ValueError('invalid catalog scope')
        words = _race_words(race)
        # Only fixed projections from the trusted adapter; no model-supplied SQL.
        sessions = self.archive_context.reader.available_sessions() if self.archive_context else []
        matching = [s for s in sessions if str(s['season']) == season and
                    (not words or words & _race_words(s['event_name']))]
        if kind == 'sessions':
            return {'rows':[{k:s[k] for k in ('session_id','event_name','session_code','season')}
                            for s in matching[:180]], 'note':'Saved sessions only; absence does not prove a session was held or cancelled.'}
        event_ids={s['session_id'].rsplit(':',1)[0] for s in matching}
        saved=TrackingQueries(self.store,owner_id=principal.identity,role=principal.role).forecasts(as_of_utc=cutoff)
        rows=[r for r in saved if r['event_id'].split(':')[0]==season
              and (not race or r['event_id'] in event_ids)]
        return {'rows':[{k:r[k] for k in ('run_id','event_id','model_id','mode','as_of_utc')}
                         for r in rows[-24:]]}

    def ask(self, principal, message, *, request_id, as_of_utc=None, run_id=None,
            event_id=None, driver_id=None, team_id=None, session_id=None, snapshot_id=None, conversation=()):
        authorize(principal, 'chat')
        if not isinstance(message, str) or not message.strip() or len(message) > min(4000, self.limiter.policy.message_chars):
            raise ValueError('message length invalid')
        cutoff = _time(as_of_utc or datetime.now(timezone.utc))
        for value in (run_id, event_id, driver_id, team_id, session_id, snapshot_id):
            if value is not None:
                _identifier(value)
        if _count(message) > min(12000, self.limiter.policy.context_tokens):
            raise ValueError('message token limit exceeded')
        context = dict(as_of_utc=cutoff.isoformat(), run_id=run_id, event_id=event_id,
                       driver_id=driver_id, team_id=team_id, session_id=session_id, snapshot_id=snapshot_id)
        identity = dict(message=message, context=context, driver_id=driver_id, team_id=team_id,
                        role=principal.role, conversation=conversation)
        fingerprint = sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
        paid=self.provider is not None
        lease = self.limiter.reserve(principal, request_id, fingerprint, action='chat', paid=paid, public=False,
                                    worst_case_cost=self.provider.limits.reserve_micro if paid else 0)
        cache_key = (principal.identity, principal.role, request_id)
        if lease.replayed:
            if lease.status == 'completed' and cache_key in self._responses:
                return json.loads(self._responses[cache_key])
            raise LimitRejected('request_already_submitted', lease.deadline)
        if lease.status != 'active':
            self.limiter.settle(principal, request_id, actual_cost=0, failed=True)
            raise LimitRejected('busy', lease.deadline)
        failed = False
        started, tool_calls = perf_counter(), 0
        result = None
        actual_cost = None if paid else 0
        try:
            self.limiter.checkpoint(principal, request_id)
            normalized = _normalize_message(message)
            result = dict(status='insufficient_data',
                          answer='I do not have sufficient saved evidence for that F1 question at this cutoff.',
                          citations=[], context=context, handoff=None)
            known_codes = set(re.findall(r'\b[A-Z]{3}\b', message)) & set(_ALIASES)
            subject = None
            if self.archive_context and event_id and session_id:
                subject = self.archive_context.resolve_driver(message, session_id)
                if subject:
                    context['mentioned_driver_id'] = subject
            if _UNSAFE.search(normalized) or _OFF_TOPIC_TASK.search(normalized) or not (_TOPIC.search(normalized) or known_codes or subject):
                result.update(status='refused', answer='Ask an F1, telemetry, profile or saved-model question. I cannot access secrets, execute code or fetch user URLs.')
                actual_cost=0
            elif paid:
                bound_event=event_id
                if run_id and not bound_event:
                    saved=TrackingQueries(self.store,owner_id=principal.identity,role=principal.role).get_forecast(run_id)
                    bound_event=saved['event_id'] if saved else None
                mismatch=bool(bound_event and any(y!=bound_event.split(':')[0] for y in re.findall(r'\b(?:19|20)\d{2}\b',normalized)))
                if self.archive_context and event_id and session_id:
                    mismatch=mismatch or not self.archive_context.scope_matches(normalized,event_id,session_id)
                    sessions=self.archive_context.reader.available_sessions()
                    named_words=_race_words(normalized)
                    named=[s for s in sessions if str(s['season'])==event_id.split(':')[0] and
                           named_words & _race_words(s['event_name'])]
                    mismatch=mismatch or any(s['session_id'].rsplit(':',1)[0]!=event_id for s in named)
                    resolved=self.archive_context.resolve_driver(message,session_id) if driver_id else None
                    mismatch=mismatch or bool(driver_id and resolved and resolved!=driver_id)
                if mismatch:
                    result.update(status='clarification',answer='Your question conflicts with the bound event or session. Refine the question or remove the additional context.')
                    actual_cost=0
                else:
                    def execute(name,arguments):
                        nonlocal tool_calls
                        tool_calls+=1
                        # Provider requests cannot escape an explicit optional binding.
                        for key,value in (('event_id',event_id),('session_id',session_id),('driver_id',driver_id),('team_id',team_id),('run_id',run_id)):
                            if value and key in arguments and arguments[key]!=value:
                                raise ValueError('tool request conflicts with bound context')
                        return self.execute_tool(principal,request_id,name,arguments,as_of_utc=cutoff)
                    generated=self.provider.answer(message,context=context,conversation=conversation,execute=execute,
                        checkpoint=lambda **kw:self.limiter.checkpoint(principal,request_id,**kw))
                    actual_cost=generated.get('usage',{}).get('cost_micro')
                    evidence=generated.pop('evidence',())
                    result.update(generated)
                    numeric=[e for e in evidence if e['kind'] in ('pending_forecast','saved_run') and e.get('data',{}).get('ranking')]
                    if numeric and not re.search(r'\b(evaluation|accuracy|mae|error)\b',normalized):
                        self._provider_forecast_answer(result,numeric[-1],normalized)
            elif (re.search(r'\b(generate|create|run|new|simulate)\b', normalized) and re.search(r'\b(forecast|prediction|scenario|strategy)\b', normalized)) or re.search(r'\bwill\b.*\b(win|winner|finish)\b', normalized):
                route = 'Strategy Lab' if re.search(r'\b(scenario|strategy|simulate)\b', normalized) else 'Predictions'
                result.update(status='handoff', answer=f'Open {route} to review inputs and submit an explicit request.', handoff={'route': route})
            else:
                if self.archive_context is not None and event_id and session_id and not run_id:
                    name, arguments = self._archive_route(normalized, event_id, session_id, driver_id, team_id,
                                                          raw_message=message)
                else:
                    name, arguments = self._route(normalized, run_id, event_id, driver_id, team_id, principal,
                                                  session_id, snapshot_id)
                if name:
                    tool_calls += 1
                    evidence = self.execute_tool(principal, request_id, name, arguments, as_of_utc=cutoff)
                    if _count(message + json.dumps(evidence, sort_keys=True)) > min(12000, self.limiter.policy.context_tokens):
                        raise ValueError('context token limit exceeded')
                    self._answer(result, name, evidence, normalized)
            self.limiter.checkpoint(principal, request_id)
            result['answer'] = safe_markdown(result['answer'])
            output_limit = self.limiter.policy.output_tokens if paid else min(1000, self.limiter.policy.output_tokens)
            if _count(result['answer']) > output_limit:
                fallback = 'Insufficient answer budget. Select one driver or a narrower saved result.'
                result.update(status='insufficient_data', answer=fallback[:output_limit], citations=[])
            self.limiter.settle(principal, request_id, actual_cost=actual_cost)
            self._responses[cache_key] = json.dumps(result)
            while len(self._responses) > 32:
                self._responses.popitem(last=False)
            return result
        except Exception:
            failed = True
            raise
        finally:
            if failed:
                self.limiter.settle(principal, request_id, actual_cost=actual_cost, failed=True)
            try:
                self.store.record_operational_event(dict(event_id='chat-'+uuid4().hex,
                    owner_id='chat-'+sha256(principal.identity.encode()).hexdigest()[:24],
                    model_id=self.provider.model_id if paid else 'grounded-chat-v1', occurred_at_utc=datetime.now(timezone.utc),
                    status='error' if failed else 'success', latency_ms=(perf_counter()-started)*1000,
                    input_tokens=result.get('usage',{}).get('input_tokens',_count(message)) if result else _count(message),
                    output_tokens=result.get('usage',{}).get('output_tokens',_count(result['answer'])) if result else 0,
                    tool_calls=tool_calls, estimated_cost=(actual_cost or 0)/1000000, billed_cost=(actual_cost or 0)/1000000, cache_hit=False))
            except Exception:
                # Monitoring cannot invalidate a completed, grounded answer.
                pass

    def _archive_route(self, message, event_id, session_id, driver_id, team_id, *, raw_message=None):
        if not self.archive_context.scope_matches(message, event_id, session_id):
            return None, {}
        resolved = self.archive_context.resolve_driver(raw_message or message, session_id)
        if driver_id and resolved and resolved != driver_id:
            return None, {}
        driver_id = driver_id or resolved
        team_id = team_id or self.archive_context.resolve_team(message, session_id)
        # An explicit known driver outside this session must not become a whole-grid answer.
        if not driver_id and any(re.search(r'\b' + re.escape(alias) + r'\b', message)
                for code, aliases in _ALIASES.items() for alias in aliases):
            return None, {}
        if not driver_id and re.search(r'\b(?:did|does)\s+\w+\s+(?:use|finish|retire|withdraw)\b', message):
            return None, {}
        if re.search(r'\b(why|happen|happened|cancelled|canceled|dnf|retire|retired|withdraw|withdrawn)\b', message):
            question = 'context'
        elif re.search(r'\b(profile|form|affiliation|lineup|reliability|team)\b', message):
            question = 'profile'
        elif re.search(r'\b(weather|rain|temperature|humidity|wind|pressure)\b', message):
            question = 'weather'
        elif re.search(r'\b(tyre|tyres|tire|tires|compound|compounds|stint|stints)\b', message):
            question = 'tyres'
        elif re.search(r'\b(lap|laps|pace|fastest|quickest)\b', message):
            question = 'laps'
        elif re.search(r'\b(result|results|classification|position|finish|finished|winner|win|won|race|qualifying|sprint)\b', message):
            question = 'results'
        else:
            return None, {}
        return 'archive', {k: v for k, v in dict(event_id=event_id, session_id=session_id,
            driver_id=driver_id, team_id=team_id, question=question).items() if v}

    @staticmethod
    def _provider_forecast_answer(result,evidence,message):
        """Numeric forecasts are rendered from tool records, never generated text."""
        saved=evidence['data']
        rows=sorted(saved['ranking'],key=lambda r:r['predicted_rank'])
        count=re.search(r'\btop\s+(\d{1,2}|five|three|ten)\b',message)
        limit=10
        if count:
            limit=int(count[1]) if count[1].isdigit() else {'five':5,'three':3,'ten':10}[count[1]]
            limit=max(1,min(32,limit))
        cid=evidence['id']
        lines=[f"Saved **{_label(saved['model_id'])}** forecast for {_label(saved['event_id'])}; cutoff {saved['as_of_utc']}. [{cid}]\n",
               '| Predicted rank | Driver | Model score |','| --- | --- | --- |']
        for r in rows[:limit]:
            driver=_label(r['driver_id']).removeprefix('fastf1:driver:').replace('_',' ').title()
            lines.append(f"| {r['predicted_rank']} | {driver} | {r['prediction_score']:g} |")
        lines.append('\nThese are model estimates, not observed results or calibrated win probabilities. No new model was run in this chat.')
        result.update(status='answered',answer='\n'.join(lines),sources=[],
            citations=[dict(id=cid,kind=evidence['kind'],reference=saved['run_id'],as_of_utc=saved['as_of_utc'])])
        result['context'].update(event_id=saved['event_id'],forecast_as_of_utc=saved['as_of_utc'],snapshot_id=saved['snapshot_id'])

    def _route(self, message, run_id, event_id, driver_id, team_id, principal, session_id, snapshot_id):
        if run_id:
            name = ('evaluation' if re.search(r'\b(mae|accuracy|evaluation|error|performance)\b', message)
                    else 'saved_run')
            arguments = {'run_id': run_id}
            if driver_id and name == 'saved_run':
                arguments['driver_id'] = driver_id
            return name, arguments
        if event_id and driver_id and re.search(r'\b(lap|laps|stint|tyre|tire|pace)\b', message):
            return 'lap_summary', {k: v for k, v in dict(event_id=event_id, driver_id=driver_id,
                                                        session_id=session_id, snapshot_id=snapshot_id).items() if v}
        if driver_id or team_id:
            return 'profiles', {k: v for k, v in dict(driver_id=driver_id, team_id=team_id).items() if v}
        history = self._visible_history(principal)
        entities = sorted({r.get('driver_id') for r in history if r.get('driver_id')})
        drivers = [d for d in entities if _selected(d, message)][:2]
        teams = sorted({r.get('team_id') for r in history if r.get('team_id')})
        team = next((t for t in teams if _selected(t, message)), None)
        if drivers or team:
            return 'profiles', {k: v for k, v in dict(driver_ids=drivers, team_id=team).items() if v}
        return None, {}

    @staticmethod
    def _answer(result, name, evidence, message):
        if not evidence:
            return
        cutoff = result['context']['as_of_utc']
        if name == 'archive':
            ChatService._archive_answer(result, evidence, message)
            return
        if name == 'saved_run':
            ranking = evidence['ranking']
            if evidence.get('driver_id'):
                selected = [r for r in ranking if r['driver_id'] == evidence['driver_id']]
            else:
                selected = [r for r in ranking if _selected(r['driver_id'], message)]
                selected = selected or sorted(ranking, key=lambda r: r['predicted_rank'])[:3]
            if not selected:
                return
            lines = [f"{_label(r['driver_id'])}: estimated rank {r['predicted_rank']}, saved score {r['prediction_score']:g}. [1]"
                     for r in selected[:6]]
            answer = ('Saved model ' + _label(evidence['model_id']) + ' for ' + _label(evidence['event_id']) + '.\n\n'
                      + '\n\n'.join(lines) + '\n\nThese are estimates. This record alone does not establish a causal explanation or calibrated win odds.')
            reference = evidence['run_id']
            result['context']['event_id'] = evidence['event_id']
            result['context']['forecast_as_of_utc'] = evidence['as_of_utc']
        elif name == 'evaluation':
            selected = [(k, v) for k, v in evidence['metrics'].items()
                        if k in {'position_mae', 'rank_spearman', 'winner_accuracy', 'pace_mae_seconds'}]
            if not selected:
                return
            answer = 'Saved evaluation: ' + '; '.join(f'{_label(k)} = {v:g}' for k, v in selected) + '. [1] One run does not establish general accuracy.'
            reference = evidence['run_id'] + ':' + evidence['revision_id']
        elif name == 'profiles':
            profiles = evidence['drivers'] + evidence['teams']
            if not profiles:
                return
            lines = []
            for p in profiles[:2]:
                form = p['recent_form']
                if not p['source_ids'] or not p['source_complete']:
                    continue
                average = f"{form['mean_position']:g}" if form['mean_position'] is not None else 'unavailable'
                lines.append(f"{_label(p['entity_id'])}: recent mean classified position {average} across {form['sample_count']} entries / {form['event_count']} events. [1]")
            if not lines:
                return
            answer = '\n\n'.join(lines) + '\n\nDescriptive last-five-event window; team/track effects and small samples can confound comparisons.'
            reference = evidence['snapshot_id']
        else:
            answer = (f"{_label(evidence['driver_id'])}: observed mean lap time {evidence['mean_lap_time_s']:g} s over {evidence['sample_count']} timestamped laps. [1] "
                      'This includes unfiltered observed laps; it is not a clean-pace or tyre-degradation estimate.')
            reference = evidence['snapshot_id']
            result['context'].update(session_id=evidence['session_id'], snapshot_id=evidence['snapshot_id'],
                                     snapshot_as_of_utc=evidence['snapshot_as_of_utc'])
        result.update(status='answered', answer=answer,
                      citations=[dict(id='1', kind=name, reference=reference, as_of_utc=cutoff)])

    @staticmethod
    def _archive_answer(result, evidence, message):
        from agents.archive_context import text
        lines, citations = [], []
        causal_evidence_missing = False
        if 'records' in evidence:
            for row in evidence['records'][:3]:
                # Recorded classification never provides an independently verified incident cause.
                if row['evidence_kind'] == 'archive_result':
                    if re.search(r'\bwhy\b', message):
                        causal_evidence_missing = True
                        summary = f"Recorded status: {text(row.get('raw_status') or row['status'])}. No independently verified incident cause is saved at this cutoff."
                    else:
                        summary = f"Recorded status: {text(row.get('raw_status') or row['status'])}; cause independently unverified."
                else:
                    summary = text(row['summary'], 320)
                    normalized_source = _normalize_message(summary)
                    if _UNSAFE.search(normalized_source) or _OFF_TOPIC_TASK.search(normalized_source):
                        continue
                number = str(len(citations) + 1)
                lines.append(f"{summary} [{number}]")
                citations.append(dict(id=number, kind='event_context', reference=row['record_id'],
                    publisher=text(row['publisher']), url=row['url'], known_at_utc=row['known_at']))
        else:
            rows = evidence.get('rows', [])
            if re.search(r'\b(winner|won|win)\b', message) and any('position' in row for row in rows):
                rows = [row for row in rows if row.get('position') == 1]
            elif re.search(r'\b(fastest|quickest)\b', message):
                rows = [row for row in rows if row.get('best_lap_s') is not None]
            limit = 1 if re.search(r'\b(winner|won|win|fastest|quickest)\b', message) else 6
            for row in rows[:limit]:
                label = text(row['display_name'])
                if 'recent_form' in row:
                    form = row['recent_form']
                    position = form['mean_position']
                    line = f"{label} ({evidence['season']}): mean classified position {position:g}" if position is not None else f"{label}: classified position unavailable"
                    line += f" over {form['event_count']} recent saved races."
                    if row.get('affiliation'):
                        line += f" Observed team: {text(row['affiliation_display_name'])}."
                elif 'position' in row:
                    position = f"{row['position']:g}" if row['position'] is not None else 'unknown'
                    line = f"{label}: position {position}; recorded status {text(row['status'])}."
                elif re.search(r'\b(tyre|tyres|tire|tires|compound|compounds|stint|stints)\b', message):
                    compounds = '; '.join(f'{text(k)} {v} laps' for k, v in row['compounds'].items())
                    line = f"{label}: {compounds or 'compound unavailable'}."
                else:
                    best = f"{row['best_lap_s']:.3f} s" if row['best_lap_s'] is not None else 'unavailable'
                    line = f"{label}: best recorded lap {best}; {row['timed_lap_count']} timed laps."
                lines.append(line + ' [1]')
            for row in evidence.get('measurements', []):
                lines.append(f"{row['label']}: {row['minimum']:g} to {row['maximum']:g} ({row['sample_count']} samples). [1]")
            if lines:
                source_ids = evidence.get('source_ids', sorted({s for r in evidence.get('rows', []) for s in r.get('source_ids', [])}))
                urls = evidence.get('source_urls', sorted({u for r in evidence.get('rows', []) for u in r.get('source_urls', [])}))
                citations = [dict(id='1', kind='archive', reference=evidence['reference'],
                    source_ids=source_ids[:8], source_urls=urls[:8], known_at_utc=evidence.get('known_at_utc'),
                    as_of_utc=result['context']['as_of_utc'])]
        if not lines:
            return
        note = ('\n\nObserved saved laps; no clean-pace, tyre-life or causal inference.'
                if any('timed_lap_count' in r for r in evidence.get('rows', [])) else '')
        if evidence.get('entry_count', 0) > 6 and not re.search(r'\b(winner|won|win|fastest|quickest)\b', message):
            note += '\n\nShowing six entries; select a driver to narrow the answer.'
        header = f"{evidence.get('event_name', 'Saved evidence')} \u00b7 {result['context']['session_id']}\n\n"
        has_curated_context = any(r['evidence_kind'] != 'archive_result' for r in evidence.get('records', []))
        result.update(status='insufficient_data' if causal_evidence_missing and not has_curated_context else 'answered',
                      answer=header + '\n\n'.join(lines) + note, citations=citations)
