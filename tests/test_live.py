from datetime import datetime,timedelta,timezone
import pytest
from data.live import LiveStore

T=datetime(2026,1,1,tzinfo=timezone.utc)


def event(i,offset=0):
    return dict(event_id=f'feed-{i}',session_id='2026:1:R',kind='position',driver_id='A',
                source_at_utc=T+timedelta(seconds=offset),received_at_utc=T+timedelta(seconds=offset+1),
                sequence=i,payload={'x':i,'y':i})


def test_single_collector_lease_and_reconnect_dedup(tmp_path):
    store=LiveStore(tmp_path/'feed.sqlite3')
    assert store.claim('2026:1:R','worker-a',now=T)
    assert not store.claim('2026:1:R','worker-b',now=T)
    assert store.append('worker-a',event(1),now=T)==True
    assert store.append('worker-a',event(1),now=T)==False
    assert store.count('2026:1:R')==1
    with pytest.raises(PermissionError):store.append('worker-b',event(2),now=T)


def test_disconnected_or_stale_never_displays_live(tmp_path):
    store=LiveStore(tmp_path/'feed.sqlite3');store.claim('2026:1:R','worker-a',now=T)
    store.append('worker-a',event(1),now=T)
    fresh=store.state('2026:1:R',now=T+timedelta(seconds=2),connected=True)
    assert fresh['status']=='live'
    assert store.state('2026:1:R',now=T+timedelta(seconds=2),connected=False)['status']=='disconnected'
    assert store.state('2026:1:R',now=T+timedelta(minutes=1),connected=True)['status']=='stale'


def test_received_cutoff_excludes_future_delivery_and_late_events(tmp_path):
    store=LiveStore(tmp_path/'feed.sqlite3');store.claim('2026:1:R','worker-a',now=T)
    store.append('worker-a',event(1,0),now=T)
    store.append('worker-a',event(3,10),now=T)
    records=store.prefix('2026:1:R',source_cutoff=T+timedelta(seconds=5),received_cutoff=T+timedelta(seconds=5))
    assert len(records)==1
    assert store.state('2026:1:R',now=T+timedelta(seconds=12),connected=True)['sequence_gaps']==[2]


def test_expired_worker_cannot_write_after_takeover(tmp_path):
    store=LiveStore(tmp_path/'feed.sqlite3',lease_seconds=5)
    store.claim('2026:1:R','worker-a',now=T)
    assert store.claim('2026:1:R','worker-b',now=T+timedelta(seconds=6))
    with pytest.raises(PermissionError):store.append('worker-a',event(1),now=T+timedelta(seconds=6))


def test_reconnect_redelivery_keeps_first_receipt_and_historical_prefix(tmp_path):
    store=LiveStore(tmp_path/'feed.sqlite3')
    store.claim('2026:1:R','worker-a',now=T)
    original=event(1)
    store.append('worker-a',original,now=T)
    cutoffs=dict(source_cutoff=T,received_cutoff=T+timedelta(seconds=1))
    prefix=store.prefix('2026:1:R',**cutoffs)
    store.disconnect('2026:1:R','worker-a')
    assert store.claim('2026:1:R','worker-b',now=T+timedelta(seconds=2))
    redelivery=dict(original,received_at_utc=T+timedelta(seconds=3))
    assert store.append('worker-b',redelivery,now=T+timedelta(seconds=3)) is False
    assert store.count('2026:1:R')==1
    assert store.prefix('2026:1:R',**cutoffs)==prefix
    assert prefix[0]['received_at_utc']==(T+timedelta(seconds=1)).isoformat()
    assert store.prefix('2026:1:R',source_cutoff=T,received_cutoff=T)==[]


@pytest.mark.parametrize('change',[
    {'payload':{'x':99,'y':1}}, {'source_at_utc':T+timedelta(seconds=1)}, {'sequence':2},
])
def test_redelivery_rejects_changed_source_content(tmp_path,change):
    store=LiveStore(tmp_path/'feed.sqlite3');store.claim('2026:1:R','worker-a',now=T)
    store.append('worker-a',event(1),now=T)
    changed=dict(event(1),received_at_utc=T+timedelta(seconds=3),**change)
    with pytest.raises(ValueError,match='conflicting duplicate'):
        store.append('worker-a',changed,now=T+timedelta(seconds=3))
    assert store.count('2026:1:R')==1
