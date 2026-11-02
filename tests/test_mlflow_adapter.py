from types import SimpleNamespace
from monitoring.mlflow_adapter import export_evaluation


class Client:
    def __init__(self): self.logged=[]
    def get_experiment_by_name(self,name): return SimpleNamespace(experiment_id='1')
    def search_runs(self,*args,**kwargs): return []
    def create_run(self,*args,**kwargs): return SimpleNamespace(info=SimpleNamespace(run_id='mlflow-run'))
    def log_metric(self,run_id,name,value,**kwargs): self.logged.append((name,value))
    def log_dict(self,run_id,payload,path): self.payload=payload
    def set_terminated(self,run_id,status): self.status=status


def test_adapter_disabled_without_explicit_target():
    assert export_evaluation({})=={'status':'disabled','reason':'No explicit tracking target configured'}


def test_adapter_logs_only_available_metrics_and_bounded_metadata():
    client=Client()
    result={'status':'evaluated_development','models':{'ridge':{'metrics':{'rank_spearman':0.7,'pace_mae_seconds':None},'n_events':12}},'protocol':{'version':'v1'},'raw_prompt':'secret'}
    exported=export_evaluation(result,client=client)
    assert exported['status']=='complete'
    assert client.logged==[('ridge.rank_spearman',0.7)]
    assert 'raw_prompt' not in client.payload
    assert client.status=='FINISHED'


def test_tracking_failure_does_not_invalidate_numeric_result():
    class Broken(Client):
        def create_run(self,*args,**kwargs): raise RuntimeError('credentialsecret')
    result={'status':'evaluated_development','models':{}}
    response=export_evaluation(result,client=Broken())
    assert response['status']=='unavailable'
    assert 'credentialsecret' not in str(response)
    assert result['status']=='evaluated_development'


def test_archive_projection_keeps_research_and_dataset_identity():
    client = Client()
    result = dict(status='evaluated_reconstruction', models={}, cohort_sha256='cohort',
        dataset_sha256='dataset', implementation_sha256='code', historical_publication_verified=False,
        holdout_locked=True, research_only=True, evidence_kind='historical_reconstruction',
        raw_prompt='must not leave the canonical store')
    assert export_evaluation(result, client=client)['status'] == 'complete'
    assert client.payload['cohort_sha256'] == 'cohort'
    assert client.payload['dataset_sha256'] == 'dataset'
    assert client.payload['historical_publication_verified'] is False
    assert client.payload['holdout_locked'] is True
    assert 'raw_prompt' not in client.payload


def test_export_retains_benchmark_lineage_and_changes_projection_digest():
    from test_benchmark import lineage_rows,protocol
    from ml.benchmark import benchmark
    class DigestClient(Client):
        def create_run(self,*args,**kwargs):
            self.tags=kwargs['tags']
            return super().create_run(*args,**kwargs)
    data=lineage_rows()
    first=benchmark(data,protocol(),candidates=('ridge',))
    changed=benchmark([dict(row,snapshot_id='snapshot-b') for row in data],protocol(),candidates=('ridge',))
    client_a,client_b=DigestClient(),DigestClient()
    assert export_evaluation(dict(first,raw_prompt='sensitive',rows=data),client=client_a)['status']=='complete'
    assert export_evaluation(changed,client=client_b)['status']=='complete'
    assert client_a.payload['evaluation_manifest']['dataset']['snapshot_ids']==['snapshot-a']
    assert client_a.payload['folds'][0]['models']['ridge']['metadata']['estimator_parameters']['alpha']==1.0
    assert client_a.tags['canonical_digest']!=client_b.tags['canonical_digest']
    assert 'sensitive' not in str(client_a.payload) and 'rows' not in client_a.payload
    assert 'fixture' not in str(client_a.payload) and 'received_at_utc' not in str(client_a.payload)
