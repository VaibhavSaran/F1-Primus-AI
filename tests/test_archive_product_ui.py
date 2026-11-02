from datetime import datetime, timezone
import json
import pandas as pd
from streamlit.testing.v1 import AppTest
from streamlit_app import services


def test_time_picker_precision_binds_the_original_snapshot_without_moving_cutoff():
    exact = datetime(2026, 3, 8, 3, 59, 59, 999999, tzinfo=timezone.utc)
    snapshot = type('Snapshot', (), dict(as_of_utc=exact))()
    controls = dict(cutoff=exact.replace(second=0, microsecond=0), mode='post_qualifying')
    assert services.bind_saved_cutoff(snapshot, controls)['cutoff'] == exact
    assert controls['cutoff'] != exact
    changed = dict(controls, cutoff=controls['cutoff'].replace(minute=58))
    assert services.bind_saved_cutoff(snapshot, changed)['cutoff'] == changed['cutoff']


def test_current_event_reads_only_its_manifest_snapshot(monkeypatch, tmp_path):
    from data.contracts import DataSnapshot, EventKey
    from data.snapshots import save_snapshot
    monkeypatch.setattr(services, 'LOCAL', tmp_path)
    snapshot = DataSnapshot('archive-current', EventKey(2025, 1), datetime(2025, 3, 1, tzinfo=timezone.utc))
    path = save_snapshot(snapshot, tmp_path / 'snapshots')
    (tmp_path / 'archive_ml').mkdir()
    (tmp_path / 'archive_ml' / 'manifest.json').write_text(json.dumps(dict(snapshots=[dict(event_id='2025:1', snapshot_id=snapshot.snapshot_id, path=str(path.resolve()))])))
    monkeypatch.setattr(services, 'snapshots', lambda: (_ for _ in ()).throw(AssertionError('Global snapshot scan')))
    records, errors = services.snapshots_for_event(dict(season=2025, round_number=1))
    assert errors == 0 and [row.snapshot_id for row in records] == ['archive-current']


def test_validation_defaults_to_the_active_archive_evaluation(monkeypatch, tmp_path):
    monkeypatch.setattr(services, 'LOCAL', tmp_path)
    (tmp_path / 'archive_ml').mkdir()
    (tmp_path / 'archive_ml' / 'benchmark.json').write_text(json.dumps(dict(version='archive-v2', cohort_sha256='current')))
    reports = tmp_path / 'evaluations'
    reports.mkdir()
    (reports / 'evaluation-z-legacy.json').write_text(json.dumps(dict(status='evaluated_reconstruction', cohort_event_ids=['old'])))
    (reports / 'evaluation-archive-current.json').write_text(json.dumps(dict(status='evaluated_reconstruction',
        version='archive-v2', cohort_sha256='current', cohort_event_ids=['new'])))
    assert services.evaluations()[0][0] == 'evaluation-archive-current.json'


def forecast_example():
    from streamlit_app.screens import render_run
    render_run(dict(run_id='saved-run', event_id='2020:6', season=2020, round_number=6,
        model_id='grid', mode='post_qualifying', as_of_utc='2020-08-15T16:00:00+00:00',
        snapshot_id='saved-input', status='research', warnings=[],
        ranking=[dict(driver_id='driver-one', predicted_rank=1, prediction_score=1.0)],
        provenance=dict(source_ids=['source-one'], missing_inputs={'driver-one': ['expected_pit_loss_s']}),
        briefing='A sourced research result.'), 'example')


def profile_example():
    from streamlit_app.screens import archive_profile_panel
    archive_profile_panel(dict(season=2025))


def test_driver_and_team_profile_selectors_keep_their_own_name_maps(monkeypatch):
    from types import SimpleNamespace
    common = dict(recent_form=dict(mean_position=2.0, event_count=1),
        reliability=dict(classified_dnf_count=0), source_ids=['source'], source_urls=[],
        lineage=[dict(event_id='2025:1', actual_position=2, status='FINISHED')])
    profiles = dict(drivers=[dict(common, entity_id='driver-one', driver_id='driver-one', display_name='Driver One')],
        teams=[dict(common, entity_id='team-one', team_id='team-one', display_name='Team One')])
    monkeypatch.setattr(services, 'archive_context', lambda: SimpleNamespace(profiles=lambda *a: profiles))
    monkeypatch.setattr(services, 'event_labels', lambda: {})
    app = AppTest.from_function(profile_example).run()
    app.selectbox(key='archive_profile_drivers').select('driver-one').run()
    assert not app.exception
    assert app.selectbox(key='archive_profile_drivers').options == ['Driver One']
    assert app.selectbox(key='archive_profile_teams').options == ['Team One']


def test_forecast_display_has_full_names_readable_evidence_and_no_json(monkeypatch, tmp_path):
    monkeypatch.setattr(services, 'LOCAL', tmp_path)
    monkeypatch.setattr(services, 'driver_labels', lambda *a, **k: {'driver-one': 'Driver One'}, raising=False)
    app = AppTest.from_function(forecast_example).run()
    assert not app.exception
    assert not app.get('json') and not app.get('code')
    assert any('Driver' in table.value and 'Driver One' in table.value['Driver'].tolist() for table in app.dataframe)
    assert len(app.get('download_button')) >= 2


def test_research_model_eligibility_requires_prior_same_cohort_artifact(monkeypatch, tmp_path):
    monkeypatch.setattr(services, 'LOCAL', tmp_path)
    root = tmp_path / 'models'
    root.mkdir()
    model_id = 'ridge-finishing_position-' + 'a' * 32
    (root / (model_id + '.joblib')).write_bytes(b'listing must not deserialize')
    record = dict(model_id=model_id, candidate='ridge', target='finishing_position',
        metadata=dict(research_reconstruction_only=True, cohort_sha256='pinned-cohort',
            training_event_ids=['2020:1'], training_max_event_time_utc='2020-01-01T00:00:00+00:00'))
    (root / (model_id + '.json')).write_text(json.dumps(record))
    snapshot = type('Snapshot', (), dict(payloads=dict(historical_reconstruction=True, cohort_sha256='pinned-cohort'),
        event=type('Event', (), dict(event_id='2020:6'))(), as_of_utc=datetime(2020, 8, 15, tzinfo=timezone.utc)))()
    assert services.eligible_artifact_ids('ridge', snapshot) == [model_id]
    record['metadata']['training_event_ids'].append('2020:6')
    (root / (model_id + '.json')).write_text(json.dumps(record))
    assert services.eligible_artifact_ids('ridge', snapshot) == []
