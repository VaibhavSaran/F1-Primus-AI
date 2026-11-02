def test_context_drops_target_future_holdout_and_unlabelled_rows():
    from ml.foundation_study import context_before
    base=dict(event_id='2024:1',driver_id='a',actual_position=2,status='Finished',
        outcome_available_at_utc='2024-01-01T13:00:00+00:00',event_time_utc='2024-01-01T12:00:00+00:00')
    rows=[base,dict(base,event_id='2025:19'),dict(base,event_id='2026:1'),
        dict(base,event_id='2024:2',status='DNS'),dict(base,event_id='2024:3',actual_position=None),
        dict(base,event_id='2024:4',outcome_available_at_utc='2025-10-19T18:59:59+00:00')]
    assert context_before(rows,'2025:19','2025-10-19T18:59:59+00:00')==[base]


def test_foundation_lineage_rejects_mixed_cohorts_and_snapshot_ids():
    from ml.foundation_study import validate_lineage
    import pytest
    from types import SimpleNamespace
    manifest=dict(cohort_sha256='one')
    snapshot=SimpleNamespace(snapshot_id='saved',event=SimpleNamespace(event_id='2025:19'),payloads=dict(cohort_sha256='one'))
    pin=dict(snapshot_id='saved',event_id='2025:19')
    validate_lineage(manifest,dict(cohort_sha256='one'),snapshot,pin)
    with pytest.raises(ValueError):validate_lineage(manifest,dict(cohort_sha256='two'),snapshot,pin)
    with pytest.raises(ValueError):validate_lineage(manifest,dict(cohort_sha256='one'),snapshot,dict(pin,snapshot_id='other'))


def test_retry_attempt_namespace_is_explicit_and_preserves_completed_evidence(tmp_path):
    from ml.foundation_study import attempt_directory
    import pytest
    assert attempt_directory(tmp_path,'timesfm','a'*64,0).name=='timesfm-'+'a'*64
    assert attempt_directory(tmp_path,'timesfm','a'*64,1).name.endswith('-attempt-1')
    with pytest.raises(ValueError):attempt_directory(tmp_path,'timesfm','a'*64,-1)


def test_low_memory_trim_recovers_headroom_without_lowering_admission(monkeypatch):
    from ml.foundation_study import ensure_memory_headroom
    from types import SimpleNamespace
    import psutil
    import ml.foundation_runtime as runtime
    values=iter([100_000_000,400_000_000])
    monkeypatch.setattr(psutil,'virtual_memory',lambda:SimpleNamespace(available=next(values)))
    calls=[]
    monkeypatch.setattr(runtime,'trim_cpu_working_set',lambda:calls.append(True) or dict(trimmed=True))
    assert ensure_memory_headroom('tabfm')['trimmed'] is True
    assert calls==[True]


def test_failed_trim_keeps_memory_guard_closed(monkeypatch):
    from ml.foundation_study import ensure_memory_headroom
    from types import SimpleNamespace
    import pytest,psutil
    import ml.foundation_runtime as runtime
    monkeypatch.setattr(psutil,'virtual_memory',lambda:SimpleNamespace(available=100_000_000))
    monkeypatch.setattr(runtime,'trim_cpu_working_set',lambda:dict(trimmed=False))
    with pytest.raises(RuntimeError,match='memory headroom'):ensure_memory_headroom('tabfm')


def test_pre_race_pace_builder_splits_pit_and_compound_gaps(tmp_path):
    from test_archive_workflow import archive
    from ml.archive_workflow import prepare_archive_workflow
    from data.snapshots import load_snapshot
    from ml.historical_study import pre_race_pace_rows
    import sqlite3
    cohort,root=archive(tmp_path)
    c=sqlite3.connect(root/'f1.sqlite3')
    for lap in range(3,9):
        c.execute("INSERT INTO laps(version_id,driver_id,lap_number,lap_time_ns,track_status_raw,provider_is_accurate,pit_out_clock_ns) VALUES('fp1','d1',?,?,'1',1,?)",(lap,80*10**9,100 if lap==4 else None))
    c.commit();c.close()
    m=prepare_archive_workflow(cohort,output_root=tmp_path/'out',snapshot_root=tmp_path/'snap')
    result=pre_race_pace_rows(load_snapshot(m['snapshots'][0]['path']),cohort)
    assert len(result)==1
    assert result[0]['session_code']=='FP1'
    assert result[0]['values']==[80.0]*4
