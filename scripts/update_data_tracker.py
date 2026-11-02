"""Write an aggregate public tracker from read-only local archive checks.

Run from the project root: python -m scripts.update_data_tracker
No provider calls, authentication, collection, or archive writes occur.
"""
import argparse
from pathlib import Path

from data.archive.status import collect_status


def public_markdown(report):
    """Project only numeric summaries; never copy local paths or error payloads."""
    rows = [
        '# Data tracker', '',
        f"Checked (UTC): {report['checked_at_utc']}",
        f"2026 scope: race weekends through **{report['through_2026']} inclusive**.", '',
        '| Year | Saved / eligible sessions | Complete | Partial | Repair targets | Missing | Cancelled excluded | Independent weather weekends |',
        '|---|---:|---:|---:|---:|---:|---:|---:|',
    ]
    for item in report['years']:
        s = item['summary']
        rows.append(
            f"| {s['year']} | {s['saved']}/{s['expected'] or '?'} | "
            f"{s['complete_sessions']} | {s['partial']} | {s['reprocess'] + s['repair']} | "
            f"{s['missing'] if s['missing'] is not None else '?'} | "
            f"{s['cancelled']} | {s['weather_complete']}/{s['weather_expected'] or '?'} |"
        )
    rows += [
        '',
        'Saved includes partial publications; complete counts follow the current archive quality rules.',
        'A complete session count does not prove gap-free telemetry. Independent weather is delayed',
        'retrospective reanalysis, separate from measured session weather and pre-race forecasts.',
        'Eligible totals follow the retained local calendar and known cancelled sessions, not a fresh',
        'official-calendar reconciliation. Unknown scope is shown as `?`.',
        '',
        'File hashes and Parquet row counts: ' + (
            'checked during this refresh.' if report['file_hashes_checked'] else
            '**not checked** in this quick refresh.'),
        'This snapshot contains aggregate counts only; private diagnostics remain local.',
        '',
        '## Refresh locally', '',
        '```powershell',
        "& '.\\primusEnv\\Scripts\\python.exe' -m scripts.update_data_tracker",
        '# Optional full file validation:',
        "& '.\\primusEnv\\Scripts\\python.exe' -m scripts.update_data_tracker --verify-files",
        '```', '',
        'These commands read local archives offline and replace this public Markdown snapshot.',
        'They do not collect data, sign in, schedule work, upload, commit or publish anything.',
        'For private session-level diagnostics use `-m data.archive.status`.',
        '',
        'See the [collection guide](data-collection-2020-2026.md),',
        '[notebook index](../notebooks/README.md), and',
        '[dated refresh plan](data-refresh-calendar.md).',
        'Future public datasets require source-by-source redistribution review; local access',
        'and an open-source client license do not establish rights to redistribute feed data.',
        '',
    ]
    return '\n'.join(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--through-2026', default='2026-11-02')
    parser.add_argument('--verify-files', action='store_true')
    args = parser.parse_args()
    project = Path(__file__).resolve().parents[1]
    report = collect_status(project, through_2026=args.through_2026,
                            verify_files=args.verify_files)
    destination = project / 'docs/data-tracker.md'
    destination.write_text(public_markdown(report), encoding='utf-8')
    print('Updated docs/data-tracker.md (aggregate counts only).')


if __name__ == '__main__':
    main()
