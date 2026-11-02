"""Small deterministic analysis exports; dense telemetry stays in the archive."""
from hashlib import sha256
from io import BytesIO
import json
from zipfile import ZIP_DEFLATED, ZipFile, ZipInfo

import pandas as pd

TABLES = ('entries', 'results', 'laps', 'stints', 'pit_visits', 'track_status', 'race_control', 'weather', 'coverage')


def analysis_bundle(tables):
    """Export only allowlisted selected-session tables under their version pin."""
    meta = tables.get('meta', {})
    if not meta.get('session_id') or not meta.get('version_id'):
        raise ValueError('A saved session and version are required')
    entries = tables.get('entries', pd.DataFrame())
    drivers = entries.get('display_name', pd.Series(dtype=str)).dropna().tolist()
    lines = ['# Saved session analysis', '',
        f"{meta.get('event_name', 'Selected race')} · {meta.get('full_name', 'Selected session')}",
        f"Session: {meta['session_id']}; version: {meta['version_id']}", '',
        'Drivers: ' + ', '.join(str(name) for name in drivers), '',
        'Tables contain recorded observations and documented derived stint/pit summaries. Missing values remain missing.',
        'A pit visit is not necessarily a stationary tyre-change duration. Retrospective weather is not a pre-race forecast.',
        'Partial coverage and unknown identities remain declared in coverage.csv. Dense telemetry is not duplicated in this small export.']
    files = {'README.md': ('\n'.join(lines) + '\n').encode('utf-8')}
    for name in TABLES:
        frame = tables.get(name)
        if not isinstance(frame, pd.DataFrame):
            continue
        # Source text is data. Prevent exported text from becoming spreadsheet formulas.
        frame = frame.copy().map(lambda value: "'" + value if isinstance(value, str) and value.startswith(('=', '+', '-', '@')) else value)
        files[name + '.csv'] = frame.to_csv(index=False, lineterminator='\n').encode('utf-8')
    manifest = dict(format_version=1, session_id=meta['session_id'], version_id=meta['version_id'],
        files={name: sha256(payload).hexdigest() for name, payload in files.items()})
    files['manifest.json'] = json.dumps(manifest, sort_keys=True, indent=2).encode('utf-8')
    buffer = BytesIO()
    with ZipFile(buffer, 'w') as package:
        for name, payload in sorted(files.items()):
            info = ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = ZIP_DEFLATED
            package.writestr(info, payload)
    return buffer.getvalue()
