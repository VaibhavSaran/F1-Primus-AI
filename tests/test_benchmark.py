from datetime import datetime,timedelta,timezone
from pathlib import Path
import json
import pytest
from ml.features import FEATURE_COLUMNS
from ml.protocols import EvaluationProtocol
from ml.benchmark import benchmark


def rows():
    start=datetime(2025,1,1,tzinfo=timezone.utc)
    result=[]
    for event in range(1,5):
        at=start+timedelta(days=7*event)
        for driver,position in [('A',1),('B',2),('C',3)]:
            row={name:None for name in FEATURE_COLUMNS}
            row.update(event_id=f'2025:{event}',driver_id=driver,team_id=driver,grid_position=position,
                       actual_position=position,event_time_utc=at,forecast_cutoff_utc=at-timedelta(hours=2),
                       outcome_available_at_utc=at+timedelta(hours=2),outcome_status='completed',status='Finished',eligible=True)
            result.append(row)
    return result


def protocol():
    return EvaluationProtocol(('2025:2','2025:3'),('2025:4',),datetime(2024,1,1,tzinfo=timezone.utc),min_train_events=1,min_development_events=2,min_final_events=1,bootstrap_samples=20)


def test_benchmark_fits_chronological_folds_and_keeps_final_locked():
    result=benchmark(rows(),protocol(),candidates=('ridge',),mode='post_qualifying')
    assert result['status']=='evaluated_development'
    assert result['cohort_event_ids']==['2025:2','2025:3']
    assert result['models']['ridge']['n_events']==2
    assert result['models']['grid']['metrics']['rank_spearman']==1
    assert result['promotion_allowed'] is False
    assert all('2025:4' not in fold['train_event_ids'] for fold in result['folds'])


def test_missing_cohort_is_insufficient_without_training():
    result=benchmark([],protocol(),candidates=('ridge','tabfm'),mode='post_qualifying')
    assert result['status']=='insufficient_data'
    assert result['models']=={}
    assert result['promotion_allowed'] is False


def test_missing_grid_uses_common_whole_event_coverage():
    data=rows();next(r for r in data if r['event_id']=='2025:2')['grid_position']=None
    result=benchmark(data,protocol(),candidates=('ridge',),mode='post_qualifying')
    assert result['status']=='insufficient_data'
    assert result['cohort_event_ids']==['2025:3']
    assert result['excluded_events']['2025:2']=='baseline_full_field_unavailable'


def lineage_rows():
    return [dict(row,source_ids=['source-a'],snapshot_id='snapshot-a',
                 audit={'provider':'fixture','received_at_utc':datetime(2025,1,1,tzinfo=timezone.utc)}) for row in rows()]


@pytest.mark.parametrize('change',[
    {'source_ids':['source-b']}, {'snapshot_id':'snapshot-b'}, {'audit':{'provider':'revised-fixture'}},
])
def test_benchmark_manifest_changes_for_source_snapshot_and_other_provenance(change):
    data=lineage_rows()
    first=benchmark(data,protocol(),candidates=('ridge',))
    changed=benchmark([dict(row,**change) for row in data],protocol(),candidates=('ridge',))
    assert first!=changed
    assert first['models']==changed['models']
    assert first['evaluation_manifest']['dataset']['sha256']!=changed['evaluation_manifest']['dataset']['sha256']


def test_benchmark_retains_repeatable_fold_local_metadata_without_fake_artifact_ids():
    first=benchmark(lineage_rows(),protocol(),candidates=('ridge',),seed=17)
    assert first==benchmark(lineage_rows(),protocol(),candidates=('ridge',),seed=17)
    manifest=first['evaluation_manifest']
    assert manifest['dataset']['source_ids']==['source-a']
    assert manifest['dataset']['snapshot_ids']==['snapshot-a']
    assert manifest['dataset']['row_count']==12
    assert manifest['feature_schema_version']=='features-v1'
    assert manifest['dataset_schema_version']=='dataset-v1'
    assert manifest['protocol_version']==protocol().version
    assert len(manifest['implementation']['source_sha256'])==64
    assert manifest['implementation']['git_revision'] is None or len(manifest['implementation']['git_revision'])==40
    for fold in first['folds']:
        model=fold['models']['ridge']
        assert model['artifact_status']=='metadata_only_not_persisted'
        assert model['model_version']=='1'
        assert model['training_config']['seed']==17
        assert model['metadata']['estimator_parameters']['alpha']==1.0
        assert model['metadata']['training_event_ids']==fold['train_event_ids']
        assert model['metadata']['sklearn_version']
        assert 'model_id' not in model and 'created_at_utc' not in model['metadata']
        assert len(fold['train_dataset']['sha256'])==64
        assert fold['test_dataset']['row_count']==3
        assert '2025:4' not in fold['train_event_ids']+fold['test_event_ids']
    assert first['evaluation_manifest']!=benchmark(lineage_rows(),protocol(),candidates=('ridge',),seed=18)['evaluation_manifest']


def test_benchmark_cli_saves_distinct_reproducible_evaluations_for_input_lineage(tmp_path,monkeypatch,capsys):
    from ml.cli import main
    from ml.protocols import freeze_protocol
    protocol_path=freeze_protocol(protocol(),tmp_path/'protocols')
    paths=[]
    monkeypatch.chdir(tmp_path)
    for source,snapshot in [('source-a','snapshot-a'),('source-b','snapshot-b'),('source-a','snapshot-a')]:
        dataset=tmp_path/'dataset.json'
        dataset.write_text(json.dumps([dict(row,source_ids=[source],snapshot_id=snapshot) for row in rows()],default=lambda t:t.isoformat()),encoding='utf-8')
        main(['benchmark','--dataset',str(dataset),'--protocol',str(protocol_path),'--mode','post_qualifying'])
        response=json.loads(capsys.readouterr().out)
        paths.append(response['path'])
        saved=json.loads(Path(response['path']).read_text(encoding='utf-8'))
        assert saved['evaluation_manifest']['dataset']['source_ids']==[source]
        assert saved['evaluation_manifest']['dataset']['snapshot_ids']==[snapshot]
        assert saved['final_holdout']=='locked' and saved['promotion_allowed'] is False
    assert paths[0]!=paths[1]
    assert paths[0]==paths[2]
    assert len(list((tmp_path/'.local/evaluations').glob('*.json')))==2
