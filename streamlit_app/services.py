"""Read trusted local artifacts lazily. Rendering never contacts a provider."""
from __future__ import annotations

from dataclasses import asdict, is_dataclass
from importlib.util import find_spec
import json
import os
import re
from pathlib import Path
import sqlite3

ROOT = Path(__file__).resolve().parents[1]
LOCAL = ROOT / ".local"

MODELS = {"grid": "Grid baseline", "qualifying": "Qualifying baseline", "rolling_form": "Rolling form", "ridge": "Ridge",
          "random_forest": "Random forest", "gradient_boosting": "Gradient boosting",'xgboost':'XGBoost'}
MODELS.update(weekend_form='Weekend form · qualifying, practice & sprint',
              tabfm='TabFM · tabular research')


def release_root():
    return Path(os.environ.get('F1_SERVING_RELEASE_ROOT', LOCAL)).resolve()


def calendar():
    from data.catalog import load_calendar
    return load_calendar(release_root() / 'catalog' / 'calendar.json')


def session_inventory():
    records = {}
    for path in local_files(release_root() / 'sessions', 'session-*.json')[:64]:
        try:
            if path.stat().st_size > 50_000_000:
                continue
            row = json.loads(path.read_text(encoding='utf-8'))
            if not row.get('laps') and row['session_id'] not in {'Q', 'S'}:
                continue
            record = {'season': row['season'], 'round_number': row['round_number'],
                            'session_id': row['session_id'], 'laps': len(row.get('laps', [])),
                            'drivers': len(row.get('entries', [])), 'positions': len(row.get('positions', []))}
            key = record['season'], record['round_number'], record['session_id']
            if record['drivers'] and (key not in records or record['laps'] > records[key]['laps']):
                records[key] = record
        except (OSError, ValueError, KeyError, TypeError):
            continue
    return sorted(records.values(), key=lambda row: (row['season'], row['round_number'], row['session_id']), reverse=True)


def plain(value):
    if is_dataclass(value):
        return asdict(value)
    return value


def local_files(directory, pattern):
    root = directory.resolve()
    if not root.is_dir():
        return []
    return sorted((path for path in root.glob(pattern)
                   if path.is_file() and not path.is_symlink() and path.resolve().parent == root), reverse=True)


def snapshots():
    from data.snapshots import load_snapshot
    records, errors = [], 0
    manifest = release_root() / 'archive_ml' / 'manifest.json'
    active = None
    if manifest.is_file():
        try:
            active = {row['snapshot_id'] for row in json.loads(manifest.read_text(encoding='utf-8'))['snapshots']}
        except (OSError, ValueError, KeyError, TypeError):
            active = set()
            errors += 1
    for path in local_files(release_root() / "snapshots", "*.json"):
        try:
            snapshot = load_snapshot(path)
            if active is None or not snapshot.snapshot_id.startswith('archive-') or snapshot.snapshot_id in active:
                records.append(snapshot)
        except (ValueError, OSError, TypeError, UnicodeError):
            errors += 1
    return records, errors


def release_reference(value, directory):
    """Resolve a portable release reference or an existing local absolute path."""
    root = (release_root() / directory).resolve()
    path = Path(value)
    if path.is_absolute():
        if release_root() != LOCAL.resolve():
            raise ValueError('Hosted release references must be relative')
    else:
        path = release_root() / path
    if path.is_symlink() or path.resolve().parent != root:
        raise ValueError('Untrusted release reference')
    return path


def snapshots_for_event(context):
    """Use the frozen index so a page never reloads every race's feature history."""
    manifest = release_root() / 'archive_ml' / 'manifest.json'
    if not manifest.is_file():
        return snapshots()
    from data.snapshots import load_snapshot
    event_id = f"{context['season']}:{context['round_number']}"
    records, errors = [], 0
    try:
        index = json.loads(manifest.read_text(encoding='utf-8'))
        for row in index['snapshots']:
            if row['event_id'] != event_id:
                continue
            path = release_reference(row['path'], 'snapshots')
            if path.stat().st_size > 50_000_000:
                raise ValueError('Untrusted snapshot reference')
            snapshot = load_snapshot(path)
            if snapshot.event.event_id != event_id or snapshot.snapshot_id != row['snapshot_id']:
                raise ValueError('Snapshot index identity mismatch')
            records.append(snapshot)
    except (OSError, ValueError, KeyError, TypeError):
        errors += 1
    try:
        from ml.pending_forecast import prepared_pending_snapshot
        pending=prepared_pending_snapshot(event_id,owner_id='local',archive_root=LOCAL/'archive_ml')
        if pending.snapshot_id not in {s.snapshot_id for s in records}:
            records.insert(0,pending)
    except (OSError,ValueError,KeyError,TypeError):
        pass
    return records, errors


def matches(snapshot, context):
    return (snapshot.event.season == context["season"] and
            snapshot.event.round_number == context["round_number"] and
            snapshot.as_of_utc == context["cutoff"])


def bind_saved_cutoff(snapshot, context):
    """The minute-resolution picker selects a saved cutoff, never rounds its data."""
    selected = dict(context)
    if snapshot and selected['cutoff'].replace(second=0, microsecond=0) == snapshot.as_of_utc.replace(second=0, microsecond=0):
        selected['cutoff'] = snapshot.as_of_utc
    return selected


def artifact_ids(candidate):
    # Listing trusted artifact IDs never executes a joblib payload.
    from ml.artifacts import MODEL_ID
    return [path.stem for path in local_files(release_root() / "models", "*.joblib")
            if MODEL_ID.fullmatch(path.stem) and path.stem.startswith(candidate + "-finishing_position-")]


def eligible_artifact_ids(candidate, snapshot):
    """List research artifacts from metadata only; never unpickle on a render."""
    ids = artifact_ids(candidate)
    if snapshot is None:
        return ids
    research = snapshot.payloads.get('historical_reconstruction') is True
    eligible = []
    training_dates = {}
    for model_id in ids:
        path = release_root() / 'models' / (model_id + '.json')
        if not path.is_file() or path.is_symlink() or path.stat().st_size > 1_000_000:
            if not research:
                eligible.append(model_id)
            continue
        try:
            record = json.loads(path.read_text(encoding='utf-8'))
            meta = record['metadata']
            if record['model_id'] != model_id or record['candidate'] != candidate:
                continue
            if bool(meta.get('research_reconstruction_only')) != research:
                continue
            if snapshot.event.event_id in meta.get('training_event_ids', []):
                continue
            if research and meta.get('cohort_sha256') != snapshot.payloads.get('cohort_sha256'):
                continue
            timestamp = datetime_from_metadata(meta.get('training_max_event_time_utc'))
            if timestamp is None or timestamp >= snapshot.as_of_utc:
                continue
            eligible.append(model_id)
            training_dates[model_id] = timestamp
        except (OSError, ValueError, KeyError, TypeError):
            continue
    floor = datetime_from_metadata('1970-01-01T00:00:00+00:00')
    return sorted(eligible, key=lambda value: training_dates.get(value, floor), reverse=True)


def model_label(value):
    model, separator, context = str(value).partition(' · ')
    candidate = model.split('-finishing_position-')[0]
    if re.fullmatch(r'(timesfm|tabfm)-[0-9a-f]{16}', model):
        candidate = model.split('-')[0]
    label = 'TimesFM 2.5 · archived' if candidate=='timesfm' else MODELS.get(candidate, candidate)
    if candidate != model:
        label += ' · version ' + model[-8:]
    return label + (separator + context if separator else '')


def event_labels():
    database = release_root() / 'f1_2025_2026' / 'f1.sqlite3'
    if not database.is_file():
        return {}
    with sqlite3.connect(database.resolve().as_uri() + '?mode=ro', uri=True) as connection:
        return {event: f'{name} · {season}' for event, name, season in
                connection.execute('SELECT event_id,event_name,season FROM events')}


def datetime_from_metadata(value):
    from datetime import datetime
    if not isinstance(value, str):
        return None
    parsed = datetime.fromisoformat(value.replace('Z', '+00:00'))
    return parsed if parsed.utcoffset() is not None else None


def driver_labels(event_id, snapshot=None):
    """Event-scoped display names; identifiers remain the underlying values."""
    labels = {}
    if snapshot:
        for entry in snapshot.payloads.get('entries', ()):
            name = entry.get('display_name') or entry.get('full_name') or entry.get('driver_name')
            if name:
                labels[entry['driver_id']] = name
    directory = release_root() / 'f1_2025_2026'
    if (directory / 'f1.sqlite3').is_file():
        from data.archive.reader import ArchiveReader
        reader = ArchiveReader(directory)
        sessions = [s for s in reader.available_sessions() if s['session_id'].rsplit(':', 1)[0] == event_id]
        sessions.sort(key=lambda s: s['session_code'] != 'Q')
        for session in sessions[:1]:
            for entry in reader.tables(session['session_id'])['entries'].to_dict('records'):
                name = entry.get('display_name') or entry.get('abbreviation')
                if name:
                    labels[entry['driver_id']] = name
                    if entry.get('abbreviation'):
                        labels[entry['abbreviation']] = name
    return labels


def generate(context, snapshot, model_id, artifact_id=None):
    if not matches(snapshot, context):
        raise ValueError("Saved snapshot event and cutoff must match the selected forecast context.")
    saved = cached_forecast(snapshot, model_id)
    if saved:
        return saved
    from data.contracts import EventKey, PredictionRequest
    from flows.jobs import guarded_forecast
    from security.limits import SQLiteLimiter
    from monitoring.store import TrackingStore
    artifact = None
    foundation_receipt = None
    if model_id in ('tabfm', 'timesfm'):
        receipts = dict(foundation_receipts(snapshot,model_id))
        if artifact_id not in receipts:
            raise ValueError('Choose a completed research inference matching this exact saved cutoff.')
        foundation_receipt = receipts[artifact_id]
    elif model_id not in ("qualifying", "grid", "rolling_form", "weekend_form"):
        from ml.artifacts import ArtifactStore
        if artifact_id not in eligible_artifact_ids(model_id, snapshot):
            raise ValueError("Choose an eligible trusted local model artifact.")
        artifact = ArtifactStore(release_root() / "models").load(artifact_id)
    request = PredictionRequest(EventKey(context["season"], context["round_number"]),
                                context["mode"], context["cutoff"])
    schedule = snapshot.payloads.get("schedule")
    options = {}
    if foundation_receipt is not None:
        options['foundation_receipt'] = foundation_receipt
    if artifact and snapshot.payloads.get('historical_reconstruction'):
        if artifact.metadata.get('research_reconstruction_only') is not True:
            raise ValueError('A matching archive research artifact is required')
        options['research_reconstruction'] = True
    return guarded_forecast(local_identity(), SQLiteLimiter(LOCAL / "limits.sqlite3"),
                             TrackingStore(LOCAL / "tracking.sqlite3"), request=request, snapshot=snapshot,
                             model_id=model_id, artifact=artifact, schedule=schedule, **options)


def cached_forecast(snapshot, candidate):
    if snapshot is None or not snapshot.snapshot_id.startswith('pending-'):
        return None
    from ml.pending_forecast import cached_pending_forecast
    local_identity()
    try:
        run = cached_pending_forecast(snapshot.event.event_id,candidate,owner_id='local',archive_root=LOCAL/'archive_ml')
        if run['snapshot_id'] == snapshot.snapshot_id and run['as_of_utc'] == snapshot.as_of_utc.isoformat():
            return run
    except (OSError,ValueError,KeyError,TypeError):
        pass
    return None


def chat_history():
    from agents.chat_history import ChatHistory
    return ChatHistory(LOCAL/'chat_history.sqlite3')


def foundation_receipts(snapshot,candidate):
    if snapshot is None or snapshot.payloads.get('historical_reconstruction') is not True:
        return []
    from flows.foundation_prediction import load_receipt
    records=[]
    for path in (release_root()/'foundation_models'/'predictions').glob('*.json'):
        if not re.fullmatch('[0-9a-f]{64}',path.stem):continue
        try:
            value=load_receipt(path,snapshot,candidate)
            if value['receipt_id']==path.stem:records.append((path.stem,value))
        except (ValueError,OSError,KeyError,TypeError):continue
    return sorted(records,key=lambda r:r[0])


def pending_snapshot_for(selected):
    if not selected.get('season') or not selected.get('round_number'):return None
    from ml.pending_forecast import prepared_pending_snapshot
    try:
        return prepared_pending_snapshot(f"{selected['season']}:{selected['round_number']}",owner_id='local',archive_root=LOCAL/'archive_ml')
    except (OSError,ValueError,KeyError,TypeError):return None


def pending_models(snapshot):
    local_identity()
    from ml.pending_forecast import cached_pending_forecast
    from ml.active_study import ACTIVE_MODELS
    results={}
    for model in ACTIVE_MODELS:
        try:
            result=cached_pending_forecast(snapshot.event.event_id,model,owner_id='local',archive_root=LOCAL/'archive_ml')
            results[model]=result if result['snapshot_id']==snapshot.snapshot_id else None
        except (OSError,ValueError,KeyError,TypeError):results[model]=None
    return results


def tracking_queries():
    if not (LOCAL / "tracking.sqlite3").is_file():
        return None
    from monitoring.store import TrackingStore
    from monitoring.queries import TrackingQueries
    return TrackingQueries(TrackingStore(LOCAL / "tracking.sqlite3"), owner_id="local", role="viewer")


def reports():
    return local_files(LOCAL / 'legacy_reports', '*.md') + local_files(ROOT / "data" / "reports", "*.md")


def dependency_readiness():
    return [{"Integration": label, "Installed": find_spec(module) is not None,
             "Credential": ("Configured locally; connection unverified" if os.getenv(key) else "Not configured")
             if key else "Not required for local reads", "Connection": "Not contacted"}
            for label, module, key in [("FastF1", "fastf1", None), ("MLflow", "mlflow", None),
                                       ("Anthropic", "anthropic", "ANTHROPIC_API_KEY"),
                                       ("Google Gemini", "langchain_google_genai", "GEMINI_API_KEY"),
                                       ("Tavily", "tavily", "TAVILY_API_KEY")]]


def json_export(value):
    return json.dumps(value, indent=2, default=str, allow_nan=False)


def local_identity():
    """Trusted local configuration only. Remote/OIDC mode is unavailable by default."""
    import streamlit as st
    from security.access import local_principal
    address=st.get_option('server.address')
    return local_principal(enabled=os.getenv('F1_ACCESS_MODE','local')=='local',
                           loopback_only=address in {'localhost','127.0.0.1','::1'},
                           identity='local',role='ai_user')


def current_study():
    try:
        pointer=json.loads((release_root()/'archive_ml'/'active_study.json').read_text(encoding='utf-8'))
        directory=release_reference(pointer['path'], 'model_studies')
        path=directory/'report.json'
        if path.is_symlink() or path.stat().st_size>8_000_000:raise ValueError('invalid study file')
        raw=path.read_bytes()
        from hashlib import sha256
        if sha256(raw).hexdigest()!=pointer['report_sha256']:raise ValueError('active study changed')
        report=json.loads(raw)
        if not report.get('cohort_event_ids') or not report.get('models'):return None
        return directory.name,report
    except (OSError,ValueError,KeyError,TypeError):
        return None


def evaluations():
    records=[]
    active_study=current_study()
    if active_study:records.append(('Current active model comparison',active_study[1]))
    for path in local_files(release_root() / 'evaluations','evaluation-*.json')[:64]:
        if path.stat().st_size>2_000_000:continue
        try:
            payload=json.loads(path.read_text(encoding='utf-8'))
            if isinstance(payload,dict) and payload.get('cohort_event_ids') and payload.get('status') in {'evaluated_development','evaluated_reconstruction'}:
                records.append((path.name,payload))
        except (ValueError,OSError):continue
    active = {}
    try:
        active = json.loads((release_root() / 'archive_ml' / 'benchmark.json').read_text(encoding='utf-8'))
    except (OSError, ValueError, TypeError):
        pass
    return sorted(records, key=lambda row: (
        row[0]=='Current active model comparison',
        bool(active) and row[1].get('cohort_sha256') == active.get('cohort_sha256') and row[1].get('version') == active.get('version'),
        row[1].get('status') == 'evaluated_reconstruction'), reverse=True)


def model_studies():
    """Operator-created, current-cohort reports only; no user-supplied paths."""
    active=current_study()
    if active:return [active]
    try:
        cohort = json.loads((release_root()/'archive_ml'/'manifest.json').read_text(encoding='utf-8'))['cohort_sha256']
    except (OSError, ValueError, KeyError):
        return []
    records = []
    paths = sorted((release_root()/'model_studies').glob('*/report.json'), key=lambda p:p.stat().st_mtime, reverse=True)
    for path in paths[:16]:
        if not re.fullmatch('[0-9a-f]{64}', path.parent.name) or path.stat().st_size > 8_000_000:
            continue
        try:
            payload = json.loads(path.read_text(encoding='utf-8'))
            if payload.get('version') == 'historical-comparison-v1' and payload.get('cohort_sha256') == cohort:
                records.append((path.parent.name,payload))
        except (OSError,ValueError,TypeError):
            continue
    return records


def study_race(study_id,event_id):
    if not re.fullmatch('[0-9a-f]{64}',study_id):raise ValueError('invalid study identity')
    folder=release_root()/'model_studies'/study_id
    report=json.loads((folder/'report.json').read_text(encoding='utf-8'))
    if event_id not in report.get('races',{}):
        if event_id==report['case_study']['event_id']:return report['case_study']
        raise ValueError('race is outside saved comparison')
    path=folder/'predictions.json'
    if path.is_symlink() or path.stat().st_size>20_000_000:raise ValueError('invalid prediction file')
    raw=path.read_bytes()
    from hashlib import sha256
    if sha256(raw).hexdigest()!=report.get('predictions_sha256'):raise ValueError('prediction digest mismatch')
    predictions=json.loads(raw)
    return dict(report['races'][event_id],models={model:[r for r in predictions.get(model,[]) if r['event_id']==event_id]
                                                for model in report['models']})


def chat_service(snapshot=None):
    from agents.chat import ChatService
    from security.limits import SQLiteLimiter,LimitPolicy
    from monitoring.store import TrackingStore
    history=snapshot.payloads.get('history',()) if snapshot else ()
    from agents.claude_provider import ClaudeProvider
    provider=ClaudeProvider.from_env()
    policy=LimitPolicy(tools=8,searches=2,deadline_seconds=90,output_tokens=8000,context_tokens=80000) if provider else LimitPolicy()
    return ChatService(SQLiteLimiter(LOCAL / 'limits.sqlite3',policy=policy),TrackingStore(LOCAL / 'tracking.sqlite3'),
                       history=history,snapshots=(snapshot,) if snapshot else (), archive_context=archive_context(),provider=provider)


def archive_context():
    directory = release_root() / 'f1_2025_2026'
    if not (directory / 'f1.sqlite3').is_file():
        return None
    from data.archive.reader import ArchiveReader
    from data.archive.event_context import EventContextStore
    from agents.archive_context import ArchiveContext
    return ArchiveContext(ArchiveReader(directory), EventContextStore(release_root() / 'event_context' / 'context.sqlite3'))
