from dataclasses import replace
from hashlib import sha256
import json
import pytest


def receipt(snapshot):
    from ml.foundation_runtime import REVISIONS
    payload=dict(schema='foundation-receipt-v1',candidate='timesfm',
        event_id=snapshot.event.event_id,snapshot_id=snapshot.snapshot_id,
        as_of_utc=snapshot.as_of_utc.isoformat(),cohort_sha256=snapshot.payloads.get('cohort_sha256'),
        source_ids=sorted(s.source_id for s in snapshot.sources),research_only=True,
        checkpoint=dict(revision=REVISIONS['timesfm']),
        predictions=[dict(event_id=snapshot.event.event_id,driver_id=e['driver_id'],prediction_score=i)
                     for i,e in enumerate(snapshot.payloads['entries'],1)])
    return payload


def test_foundation_receipt_cannot_cross_event_cutoff_or_field(tmp_path):
    from test_historical_forecast import reconstruction
    from flows.foundation_prediction import save_receipt,load_receipt
    _,snapshot,_=reconstruction()
    path=save_receipt(receipt(snapshot),root=tmp_path)
    assert load_receipt(path,snapshot,'timesfm')['candidate']=='timesfm'
    with pytest.raises(ValueError): load_receipt(path,replace(snapshot,snapshot_id='other'),'timesfm')
    with pytest.raises(ValueError): load_receipt(path,snapshot,'tabfm')
    bad=json.loads(path.read_text());bad['predictions'][0]['prediction_score']=999
    path.write_text(json.dumps(bad))
    with pytest.raises(ValueError,match='digest'): load_receipt(path,snapshot,'timesfm')


def test_foundation_prediction_is_guarded_research_and_versioned(tmp_path):
    from test_historical_forecast import reconstruction
    from flows.foundation_prediction import save_receipt,load_receipt
    from flows.prediction_service import generate_forecast,forecast_run_id
    request,snapshot,schedule=reconstruction()
    value=load_receipt(save_receipt(receipt(snapshot),root=tmp_path),snapshot,'timesfm')
    result=generate_forecast(request,snapshot,schedule=schedule,model_id='timesfm',foundation_receipt=value)
    assert result['ranking'][0]['predicted_rank']==1
    assert result['provenance']['validation_status']=='historical_reconstruction'
    assert 'pace' in result['briefing'].lower()
    assert result['run_id']==forecast_run_id(request,snapshot,model_id='timesfm',foundation_receipt=value)
    with pytest.raises(ValueError):
        generate_forecast(request,replace(snapshot,payloads=dict(snapshot.payloads,historical_reconstruction=False)),schedule=schedule,model_id='timesfm',foundation_receipt=value)


def test_ui_receipt_lookup_requires_exact_snapshot_and_preserves_full_names(tmp_path,monkeypatch):
    from test_historical_forecast import reconstruction
    from flows.foundation_prediction import save_receipt
    from streamlit_app import services
    _,snapshot,_=reconstruction()
    monkeypatch.setattr(services,'LOCAL',tmp_path)
    path=save_receipt(receipt(snapshot),root=tmp_path/'foundation_models'/'predictions')
    found=services.foundation_receipts(snapshot,'timesfm')
    assert [r[0] for r in found]==[path.stem]
    assert services.foundation_receipts(replace(snapshot,snapshot_id='other'),'timesfm')==[]
    assert 'TimesFM' in services.model_label('timesfm-'+path.stem[:16])


def test_checkpoint_model_identity_is_shared_across_race_receipts(tmp_path):
    from test_historical_forecast import reconstruction
    from flows.foundation_prediction import save_receipt,load_receipt
    from flows.prediction_service import generate_forecast,forecast_run_id
    request,snapshot,schedule=reconstruction()
    snapshot=replace(snapshot,payloads=dict(snapshot.payloads,history=[dict(
        event_id='2024:1',driver_id='A',actual_position=2,
        event_time_utc='2024-01-01T12:00:00+00:00',outcome_available_at_utc='2024-01-01T13:00:00+00:00')]))
    value=receipt(snapshot);value['protocol_sha256']='a'*64
    first=load_receipt(save_receipt(value,root=tmp_path),snapshot,'timesfm')
    result=generate_forecast(request,snapshot,schedule=schedule,model_id='timesfm',foundation_receipt=first,history=[])
    assert forecast_run_id(request,snapshot,model_id='timesfm',foundation_receipt=first)!=result['run_id']
    assert result['run_id']==forecast_run_id(request,snapshot,model_id='timesfm',foundation_receipt=first,history=[])
    value['predictions'][0]['prediction_score']=99
    second=load_receipt(save_receipt(value,root=tmp_path),snapshot,'timesfm')
    again=generate_forecast(request,snapshot,schedule=schedule,model_id='timesfm',foundation_receipt=second)
    assert result['model_id']==again['model_id']
    assert result['run_id']!=again['run_id']
