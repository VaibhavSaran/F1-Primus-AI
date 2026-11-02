from io import BytesIO
from zipfile import ZipFile
import hashlib
import json
import pandas as pd
import pytest

from flows.analysis_report import analysis_bundle


def saved_tables():
    return dict(meta=dict(session_id='2020:6:FP1', version_id='pinned-version',
        event_name='Spanish Grand Prix', full_name='Practice 1', season=2020),
        entries=pd.DataFrame([dict(driver_id='driver-one', display_name='Driver One', team_name_at_event='Team')]),
        laps=pd.DataFrame([dict(driver_id='driver-one', lap_number=2, lap_time_s=80.123)]),
        results=pd.DataFrame([dict(driver_id='driver-one', status_raw='Finished')]),
        coverage=pd.DataFrame([dict(dataset_kind='laps', status='available', rows=1)]),
        weather=pd.DataFrame([dict(session_time_s=120, air_temperature_c=25)]))


def test_bundle_is_deterministic_pinned_and_contains_real_selected_rows():
    tables = saved_tables()
    first = analysis_bundle(tables)
    assert first == analysis_bundle(tables)
    with ZipFile(BytesIO(first)) as package:
        manifest = json.loads(package.read('manifest.json'))
        assert manifest['version_id'] == 'pinned-version'
        assert manifest['session_id'] == '2020:6:FP1'
        for path, digest in manifest['files'].items():
            assert hashlib.sha256(package.read(path)).hexdigest() == digest
        laps = pd.read_csv(BytesIO(package.read('laps.csv')))
        assert laps.lap_time_s.tolist() == [80.123]
        assert 'Driver One' in package.read('README.md').decode()
        assert not any(name.startswith('raw/') or '..' in name for name in package.namelist())
    assert tables['laps'].lap_time_s.tolist() == [80.123]


def test_bundle_rejects_missing_identity_and_never_exports_unexpected_payloads():
    tables = saved_tables()
    tables['secrets'] = pd.DataFrame([dict(password='dummy-password')])
    with ZipFile(BytesIO(analysis_bundle(tables))) as package:
        assert 'secrets.csv' not in package.namelist()
    tables['meta'].pop('version_id')
    with pytest.raises(ValueError, match='version'):
        analysis_bundle(tables)


def test_untrusted_text_cannot_inject_spreadsheet_formulas():
    tables = saved_tables()
    tables['results'].loc[0, 'status_raw'] = '=external_formula()'
    with ZipFile(BytesIO(analysis_bundle(tables))) as package:
        assert "'=external_formula()" in package.read('results.csv').decode()
