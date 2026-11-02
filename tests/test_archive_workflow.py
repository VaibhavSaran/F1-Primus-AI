from datetime import datetime, timezone
import json
import sqlite3
from pathlib import Path
import pytest
from data.archive.schema import SCHEMA_SQL


def test_common_evaluation_requires_both_roster_directions():
    from ml.archive_workflow import matching_field
    predicted = [dict(driver_id='one')]
    outcomes = [dict(driver_id='one'), dict(driver_id='two')]
    assert matching_field(predicted, outcomes) is False
    assert matching_field(outcomes, predicted) is False
    assert matching_field(outcomes, outcomes) is True


def test_fold_keeps_feature_rows_without_labels_for_full_field_review():
    from ml.archive_workflow import chronological_folds
    past = dict(event_id='2020:1', driver_id='one', forecast_cutoff_utc='2020-01-01T00:00:00+00:00',
        event_time_utc='2020-01-02T00:00:00+00:00', outcome_available_at_utc='2020-01-02T01:00:00+00:00')
    target = dict(past, event_id='2020:2', forecast_cutoff_utc='2020-02-01T00:00:00+00:00',
        event_time_utc='2020-02-02T00:00:00+00:00', outcome_available_at_utc='2020-02-02T01:00:00+00:00')
    missing = dict(target, driver_id='two', event_time_utc=None, outcome_available_at_utc=None)
    folds = chronological_folds([past, target, missing], min_train_events=1)
    assert len(folds) == 1 and len(folds[0][1]) == 2


def archive(tmp_path):
    root = tmp_path / 'archive'; root.mkdir()
    c = sqlite3.connect(root / 'f1.sqlite3'); c.executescript(SCHEMA_SQL)
    c.execute("INSERT INTO seasons(year) VALUES(2020)")
    c.execute("INSERT INTO events(event_id,season,round_number) VALUES('2020:1',2020,1)")
    sessions = []
    for code, day in [('FP1',1),('Q',2),('R',3)]:
        start = int(datetime(2020,7,day,12,tzinfo=timezone.utc).timestamp()*1e9)
        c.execute('INSERT INTO sessions(session_id,event_id,session_code,scheduled_start_utc_ns) VALUES(?,?,?,?)',(f'2020:1:{code}','2020:1',code,start))
        v = code.lower()
        c.execute("INSERT INTO session_versions(version_id,session_id,created_at_utc,normalizer_version,dataset_hash,status,observed_start_utc_ns,observed_end_utc_ns) VALUES(?,?,?,'test','hash','published',?,?)",(v,f'2020:1:{code}','2026-01-01T00:00:00+00:00',start,start+3600*10**9))
        sessions.append(dict(session_id=f'2020:1:{code}',version_id=v,season=2020,round_number=1,session_code=code,dataset_hash='hash'))
        for i in (1,2):
            d=f'd{i}'; c.execute('INSERT OR IGNORE INTO drivers(driver_id) VALUES(?)',(d,))
            c.execute('INSERT INTO session_entries(version_id,driver_id) VALUES(?,?)',(v,d))
            c.execute('INSERT INTO session_results(version_id,driver_id,finishing_position,grid_position,status_raw) VALUES(?,?,?,?,?)',(v,d,i,99,'Finished'))
            c.execute("INSERT INTO laps(version_id,driver_id,lap_number,lap_time_ns,track_status_raw,provider_is_accurate) VALUES(?,?,2,?,'1',1)",(v,d,(80+i)*10**9))
    c.commit(); c.close()
    cohort=tmp_path/'manifest.json'; cohort.write_text(json.dumps(dict(sessions=sessions,archive_roots=dict(f1=str(root)))))
    return cohort,root


def test_archive_snapshot_uses_earlier_pinned_sessions_and_separates_labels(tmp_path):
    from ml.archive_workflow import prepare_archive_workflow
    from data.snapshots import load_snapshot
    cohort,root=archive(tmp_path)
    result=prepare_archive_workflow(cohort,output_root=tmp_path/'out',snapshot_root=tmp_path/'snap')
    repeated=prepare_archive_workflow(cohort,output_root=tmp_path/'out',snapshot_root=tmp_path/'snap')
    assert repeated==result
    snapshot=load_snapshot(result['snapshots'][0]['path'])
    assert snapshot.payloads['historical_reconstruction'] is True
    assert {r.session_id for r in snapshot.payloads['driver_sessions']}=={'FP1','Q'}
    assert all('grid_position' not in r.values for r in snapshot.payloads['driver_sessions'])
    assert not any('actual_position' in entry for entry in snapshot.payloads['entries'])
    assert snapshot.sources[0].fetched_at_utc.year==2026
    rows=json.loads(Path(result['dataset_path']).read_text())
    assert len(rows)==2 and rows[0]['actual_position']==1
    assert rows[0]['qualifying_position']==1 and rows[0]['grid_position'] is None
    # A newly preferred publication cannot alter a pinned cohort.
    c=sqlite3.connect(root/'f1.sqlite3'); c.execute("UPDATE session_versions SET dataset_hash='changed' WHERE version_id='q'"); c.commit(); c.close()
    with pytest.raises(ValueError,match='pin'):
        prepare_archive_workflow(cohort,output_root=tmp_path/'out2',snapshot_root=tmp_path/'snap2')


def test_development_folds_never_train_future_or_2026():
    from ml.archive_workflow import chronological_folds
    rows=[dict(event_id=f'{y}:{r}',season=y,event_time_utc=f'{y}-03-{r:02}T12:00:00+00:00',outcome_available_at_utc=f'{y}-03-{r:02}T15:00:00+00:00',forecast_cutoff_utc=f'{y}-03-{r:02}T11:00:00+00:00') for y in (2024,2025,2026) for r in (1,2)]
    folds=chronological_folds(rows,min_train_events=1)
    assert len(folds)==3
    for train,test in folds:
        assert all(r['season']<2026 for r in train+test)
        assert max(r['outcome_available_at_utc'] for r in train)<test[0]['forecast_cutoff_utc']

def test_learned_reconstruction_requires_explicit_research_and_cannot_enter_live(tmp_path):
    from test_historical_forecast import reconstruction
    from flows.prediction_service import generate_forecast
    from ml.training import train_model, TrainingConfig
    from ml.features import FEATURE_COLUMNS
    from dataclasses import replace
    request,snapshot,schedule=reconstruction()
    rows=[]
    for i,driver in enumerate(('A','B'),1):
        row=dict.fromkeys(FEATURE_COLUMNS)
        row.update(circuit_id='circuit',team_id='team',compound='SOFT',event_id='2020:1',driver_id=driver,actual_position=i,event_time_utc='2020-01-01T12:00:00+00:00',outcome_available_at_utc='2020-01-01T15:00:00+00:00')
        rows.append(row)
    artifact=train_model(rows,TrainingConfig())
    artifact.metadata['research_reconstruction_only']=True
    with pytest.raises(ValueError,match='research'):
        generate_forecast(request,snapshot,model_id='ridge',artifact=artifact,schedule=schedule)
    result=generate_forecast(request,snapshot,model_id='ridge',artifact=artifact,schedule=schedule,research_reconstruction=True)
    assert len(result['ranking'])==2
    assert result['provenance']['historical_publication_verified'] is False
    live=replace(snapshot,payloads=dict(snapshot.payloads,historical_reconstruction=False))
    with pytest.raises(ValueError,match='research'):
        generate_forecast(request,live,model_id='ridge',artifact=artifact,schedule=schedule,research_reconstruction=True)


def test_qualifying_roster_excludes_practice_reserve_and_keeps_untimed_entrant(tmp_path):
    from ml.archive_workflow import prepare_archive_workflow
    from data.snapshots import load_snapshot
    cohort,root=archive(tmp_path)
    c=sqlite3.connect(root/'f1.sqlite3')
    for driver,version in [('reserve','fp1'),('untimed','q'),('phantom','q')]:
        c.execute('INSERT INTO drivers(driver_id) VALUES(?)',(driver,))
        c.execute('INSERT INTO session_entries(version_id,driver_id) VALUES(?,?)',(version,driver))
    c.execute("INSERT INTO session_results(version_id,driver_id,finishing_position) VALUES('q','untimed',3)")
    c.execute("INSERT INTO session_results(version_id,driver_id) VALUES('q','phantom')")
    c.execute("INSERT INTO laps(version_id,driver_id,lap_number,lap_time_ns) VALUES('fp1','reserve',2,81000000000)")
    c.commit();c.close()
    m=prepare_archive_workflow(cohort,output_root=tmp_path/'out',snapshot_root=tmp_path/'snap')
    snapshot=load_snapshot(m['snapshots'][0]['path'])
    assert {e['driver_id'] for e in snapshot.payloads['entries']}=={'d1','d2','untimed'}
    untimed=next(r for r in json.loads(Path(m['dataset_path']).read_text()) if r['driver_id']=='untimed')
    assert untimed['practice_pace_delta_s'] is None and untimed['actual_position'] is None

def test_pin_cannot_relabel_session_as_another_event(tmp_path):
    from ml.archive_workflow import prepare_archive_workflow
    cohort,_=archive(tmp_path); data=json.loads(cohort.read_text()); data['sessions'][1]['round_number']=2; cohort.write_text(json.dumps(data))
    with pytest.raises(ValueError,match='pin'):
        prepare_archive_workflow(cohort,output_root=tmp_path/'out',snapshot_root=tmp_path/'snap')

def test_qualifying_baseline_is_not_a_fabricated_grid(tmp_path):
    from ml.archive_workflow import prepare_archive_workflow
    from data.snapshots import load_snapshot
    from data.contracts import PredictionRequest
    from flows.prediction_service import generate_forecast
    cohort,_=archive(tmp_path)
    m=prepare_archive_workflow(cohort,output_root=tmp_path/'out',snapshot_root=tmp_path/'snap'); s=load_snapshot(m['snapshots'][0]['path'])
    request=PredictionRequest(s.event,'post_qualifying',s.as_of_utc)
    q=generate_forecast(request,s,model_id='qualifying',schedule=s.payloads['schedule'])
    grid=generate_forecast(request,s,model_id='grid',schedule=s.payloads['schedule'])
    assert [r['driver_id'] for r in q['ranking']]==['d1','d2']
    assert all(r['baseline_source']=='qualifying' for r in q['ranking'])
    assert grid['ranking']==[]
