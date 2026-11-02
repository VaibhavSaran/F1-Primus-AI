import json
from pathlib import Path
from test_prediction_service import inputs
from flows.pre_race_pipeline import pre_race_pipeline, save_report


def test_pipeline_bundle_exports_agree_and_repeat_is_idempotent(tmp_path):
    request,snapshot,schedule=inputs()
    result=pre_race_pipeline(request=request,snapshot=snapshot,schedule=schedule,model_id='grid',reports_dir=tmp_path)
    bundle=Path(result['report_path'])
    saved=json.loads(bundle.read_text(encoding='utf-8'))
    assert saved['run_id']==result['run_id']
    assert saved['ranking']==result['ranking']
    assert saved['report']==result['report']
    assert save_report(result,directory=tmp_path)==bundle
    assert len(list(tmp_path.glob('*.json')))==1


def test_operator_cli_uses_saved_history_and_excludes_future_results(tmp_path,monkeypatch,capsys):
    from test_prediction_service import history_inputs
    from data.snapshots import save_snapshot
    from flows.pre_race_pipeline import main
    request,snapshot,schedule=history_inputs()
    path=save_snapshot(snapshot,tmp_path/'snapshots')
    monkeypatch.chdir(tmp_path)
    main(['--snapshot',str(path),'--model','rolling_form'])
    response=json.loads(capsys.readouterr().out)
    saved=json.loads(Path(response['report_path']).read_text(encoding='utf-8'))
    assert [(r['driver_id'],r['prediction_score'],r['baseline_source']) for r in saved['ranking']]==[('B',1.0,'driver_form'),('A',2.0,'driver_form')]
