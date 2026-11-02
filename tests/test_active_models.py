"""Active research cohorts retain complete fields and chronological training."""
import pytest
from test_training import training_rows


def test_xgboost_registered_and_artifact_id_accepted():
    from ml.training import candidate_registry
    from ml.artifacts import MODEL_ID
    assert 'xgboost' in candidate_registry()
    assert MODEL_ID.fullmatch('xgboost-finishing_position-'+'a'*32)


def test_xgboost_metadata_is_strict_json_and_predicts_unseen_categories():
    import json
    from ml.training import train_model,predict_model,TrainingConfig
    artifact=train_model(training_rows(),TrainingConfig(candidate='xgboost'))
    json.dumps(artifact.metadata,allow_nan=False)
    rows=predict_model(artifact,[dict(training_rows()[0],team_id='unseen',driver_id='rookie')])
    assert len(rows)==1 and rows[0]['predicted_rank']==1


def test_nested_benchmarks_are_latest_complete_shared_events():
    from ml.active_study import nested_benchmarks
    events = [f'2025:{i}' for i in range(1, 102)]
    rows = [dict(event_id=e, driver_id=d, prediction_score=i, actual_position=i,
                 outcome_status='completed', status='FINISHED')
            for e in events for i,d in enumerate(('a','b','c'),1)]
    result = nested_benchmarks({'qualifying': rows, 'tabfm': rows}, events)
    assert result['100']['cohort_event_ids'] == events[-100:]
    for n in (5,10,50,100):
        assert result[str(n)]['cohort_event_ids'] == events[-n:]
        assert all(v['n_events'] == n for v in result[str(n)]['models'].values())
    with pytest.raises(ValueError, match='same complete'):
        nested_benchmarks({'qualifying': rows,'tabfm': rows[:-1]},events)


def test_pending_target_has_no_labels_or_2026_training_context():
    from ml.pending_forecast import inference_rows
    rows = training_rows()
    target = [dict(rows[0],event_id='2026:19',actual_position=999)]
    train, clean = inference_rows(rows+target,'2026:19','2026-11-01T19:59:59+00:00')
    assert all(int(r['event_id'].split(':')[0]) < 2026 for r in train)
    assert 'actual_position' not in clean[0]
    assert 'outcome_status' not in clean[0]


def test_prepared_pending_snapshot_rejects_other_owner(tmp_path,monkeypatch):
    import json
    from ml.pending_forecast import prepared_pending_snapshot
    (tmp_path/'pending-2026-19.json').write_text(json.dumps(dict(owner_id='alice',snapshot_path='never-read')))
    with pytest.raises(ValueError,match='owner'):
        prepared_pending_snapshot('2026:19',owner_id='bob',archive_root=tmp_path)


def test_pending_cutoff_cannot_precede_retrieval_or_follow_race(tmp_path,monkeypatch):
    import json
    from datetime import timedelta
    from dataclasses import replace
    from test_prediction_service import inputs
    from ml.pending_forecast import pending_snapshot
    import data.snapshots
    request,snapshot,schedule=inputs()
    snapshot=replace(snapshot,payloads=dict(snapshot.payloads,schedule=schedule,
        driver_sessions=[replace(r,session_id='Q') for r in snapshot.payloads['driver_sessions']]))
    (tmp_path/'manifest.json').write_text(json.dumps(dict(snapshots=[dict(event_id='2025:12',path='mock')])) )
    monkeypatch.setattr(data.snapshots,'load_snapshot',lambda p:snapshot)
    with pytest.raises(ValueError,match='retrieval'):
        pending_snapshot('2025:12',archive_root=tmp_path,now=request.as_of_utc-timedelta(hours=4))
    with pytest.raises(ValueError,match='before race'):
        pending_snapshot('2025:12',archive_root=tmp_path,now=request.as_of_utc+timedelta(days=2))
