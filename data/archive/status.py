"""Refresh a local data-status ledger without fetching or modifying archive data.

Run with primusEnv: python -m data.archive.status --verify-files
"""
import argparse
from collections import Counter, defaultdict
from datetime import date, datetime, timezone
import json
from pathlib import Path
import sqlite3
from uuid import uuid4

from .notebook_audit import audit_year
from .weather_audit import audit_weather

YEARS = (2020, 2021, 2022, 2023, 2024, 2025, 2026)


def summarize(f1, weather):
    held = [row for row in f1['sessions'] if row['action'] != 'not_held']
    counts = Counter(row['action'] for row in held)
    expected = len(held) if held else None
    saved = sum(row.get('version_id') is not None for row in held)
    integrity = f1.get('integrity', {})
    unknown_dates = sum(not row.get('date') for row in held)
    healthy = (integrity.get('sqlite') == 'ok' and not integrity.get('foreign_key_errors')
               and not integrity.get('file_errors'))
    complete = bool(expected and not unknown_dates and saved == expected and healthy
                    and not counts['reprocess_identity'] and not counts['repair_local'])
    return dict(year=f1['year'], through_date=f1.get('through_date'),
        expected=expected, saved=saved, missing=counts['download'] if expected else None,
        complete_sessions=counts['complete'], partial=counts['review_partial'],
        reprocess=counts['reprocess_identity'], repair=counts['repair_local'],
        cancelled=len(f1['sessions'])-len(held), unknown_dates=unknown_dates,
        coverage_complete=complete,
        state='Complete coverage' if complete else 'Incomplete' if expected else 'Calendar unavailable',
        weather_expected=weather.get('expected_events'), weather_complete=weather.get('complete_events', 0),
        weather_counts=weather.get('counts', {}),
        files_verified=integrity.get('files_verified', 0), file_errors=len(integrity.get('file_errors', [])))


def collect_status(project, years=YEARS, through_2026='2026-11-02', *, verify_files=False, emit=print):
    through_2026 = date.fromisoformat(through_2026).isoformat()
    project = Path(project).resolve()
    started = datetime.now(timezone.utc).isoformat()
    result = dict(started_at_utc=started, through_2026=through_2026,
                  file_hashes_checked=verify_files, years=[])
    for year in years:
        emit(f'Checking {year} F1 coverage' + (' and file hashes…' if verify_files else '…'))
        cutoff = through_2026 if year == 2026 else None
        try:
            f1 = audit_year(project/'.local/f1_2025_2026', year,
                            verify_files=verify_files, through_date=cutoff)
        except (OSError, sqlite3.Error) as error:
            f1 = dict(year=year, through_date=cutoff, sessions=[], counts={},
                      integrity=dict(error=str(error), sqlite='unavailable', file_errors=[]))
        event_ids = sorted({row['session_id'].rsplit(':', 1)[0] for row in f1['sessions']})
        emit(f'Checking {year} weather…')
        weather = audit_weather(project/'.local/f1_weather', event_ids, verify_files=verify_files)
        if not event_ids:
            weather['expected_events'] = None
        summary = summarize(f1, weather)
        result['years'].append(dict(summary=summary, f1=f1, weather=weather))
        emit(f"{year}: {summary['saved']}/{summary['expected'] or '?'} sessions; "
             f"{summary['partial']} partial; weather {summary['weather_complete']}/{summary['weather_expected'] or '?'}")
    result['checked_at_utc'] = datetime.now(timezone.utc).isoformat()
    return result


def markdown(report):
    rows = ['# F1 local data status', '', f"Checked: {report['checked_at_utc']} (UTC).",
            f"2026 scope: race weekends through **{report['through_2026']} inclusive**; later weekends excluded.",
            '', '| Year | Saved / held sessions | Missing | Partial | F1 coverage | Weather weekends |',
            '|---|---:|---:|---:|---|---:|']
    for item in report['years']:
        s = item['summary']
        rows.append(f"| {s['year']} | {s['saved']}/{s['expected'] or '?'} | "
                    f"{s['missing'] if s['missing'] is not None else '?'} | {s['partial']} | "
                    f"{s['state']} | {s['weather_complete']}/{s['weather_expected'] or '?'} |")
    rows += ['', 'Saved sessions include partial publications. Complete coverage does not mean gap-free telemetry.',
             'Weather is the separate retrospective weather archive, not proof of complete F1 telemetry.',
             'Cancelled sessions are excluded from held-session totals. No downloads or authentication are performed.',
             'File hashes: ' + ('verified during this run.' if report['file_hashes_checked'] else 'not checked in this quick refresh.'), '']
    for item in report['years']:
        s, f1, weather = item['summary'], item['f1'], item['weather']
        rows += [f"## {s['year']}", '',
                 f"Quality: {s['complete_sessions']} complete, {s['partial']} partial, "
                 f"{s['reprocess']} identity repairs, {s['repair']} local repairs. "
                 f"Cancelled: {s['cancelled']}. Verified F1 files: {s['files_verified']}."]
        if f1['integrity'].get('error'):
            rows.append('Archive unavailable: ' + f1['integrity']['error'])
        if f1.get('checkpoint_warning'):
            rows.append(f1['checkpoint_warning'])
        missing = defaultdict(list)
        for row in f1['sessions']:
            if row['action'] in ('download', 'reprocess_identity', 'repair_local'):
                missing[row['event']].append(row['session'])
            if row['action'] == 'download' and row['checkpoint_status'] in ('blocked', 'deferred', 'failed', 'running'):
                rows.append(f"Checkpoint {row['session_id']}: {row['checkpoint_status']} — {row['detail']}")
        rows += ['', '### Remaining F1 sessions', '']
        rows.extend(f"- {event}: {', '.join(sessions)}" for event, sessions in missing.items())
        if not missing:
            rows.append('None.' if s['expected'] else 'Unknown until a calendar is available.')
        rows += ['', '### Partial coverage', '']
        partial = [r for r in f1['sessions'] if r['action'] == 'review_partial']
        rows.extend(f"- {r['event']} / {r['session']}: {r['detail']}" for r in partial)
        if not partial:
            rows.append('None recorded.')
        cancelled = [r for r in f1['sessions'] if r['action'] == 'not_held']
        if cancelled:
            rows += ['', '### Cancelled sessions excluded', '']
            rows.extend(f"- {r['event']} / {r['session']}: {r['detail']} ({r.get('reference', '')})" for r in cancelled)
        rows += ['', '### Weather pending or needing review', '']
        unresolved = [r for r in weather.get('events', []) if r['status'] != 'complete']
        rows.extend(f"- {r['event_id']}: {r['status']} — {r.get('detail', '')}" for r in unresolved)
        if not unresolved:
            rows.append('None.' if s['weather_expected'] else 'Expected scope unavailable.')
        rows.append('')
    return '\n'.join(rows)


def save_status(report, directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    encoded = json.dumps(report, indent=2, ensure_ascii=False)
    # Immutable run snapshots retain the history; current files are replaced atomically.
    history = directory/'history'
    history.mkdir(exist_ok=True)
    with (history/f'{stamp}-{uuid4().hex[:8]}.json').open('x', encoding='utf-8') as stream:
        stream.write(encoded)
    for name, content in [('current.json', encoded), ('current.md', markdown(report))]:
        temporary = directory/f'.{name}.{uuid4().hex}.tmp'
        temporary.write_text(content, encoding='utf-8')
        temporary.replace(directory/name)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project', type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument('--years', type=int, nargs='+', choices=range(2020, 2027), default=list(YEARS))
    parser.add_argument('--through-2026', default='2026-11-02')
    parser.add_argument('--verify-files', action='store_true')
    args = parser.parse_args()
    report = collect_status(args.project, tuple(dict.fromkeys(args.years)), args.through_2026,
                            verify_files=args.verify_files)
    directory = args.project/'.local/data_status'
    save_status(report, directory)
    print(f'Status saved to {directory / "current.md"}', flush=True)


if __name__ == '__main__':
    main()
