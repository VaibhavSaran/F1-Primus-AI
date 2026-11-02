from dataclasses import replace
from datetime import datetime, timezone
import pytest
from ml.protocols import EvaluationProtocol, freeze_protocol, load_protocol, assess_readiness


def protocol():
    return EvaluationProtocol(development_events=tuple(f"2025:{i}" for i in range(1,13)), final_events=tuple(f"2026:{i}" for i in range(1,7)), frozen_at_utc=datetime(2026,1,1,tzinfo=timezone.utc))


def test_protocol_freeze_is_deterministic_and_refuses_edit(tmp_path):
    original=protocol()
    path=freeze_protocol(original,tmp_path)
    assert load_protocol(path)==original
    assert freeze_protocol(original,tmp_path)==path
    text=path.read_text().replace('rank_spearman','position_mae')
    path.write_text(text)
    with pytest.raises(ValueError,match='integrity'):
        load_protocol(path)


def test_overlap_and_retroactive_freeze_rejected():
    with pytest.raises(ValueError):
        replace(protocol(), final_events=('2025:1',))
    with pytest.raises(ValueError):
        replace(protocol(), frozen_at_utc=datetime(2026,1,1))


def test_missing_real_cohort_cannot_promote():
    result=assess_readiness(protocol(), folds=[], verified_event_ids=[])
    assert result['status']=='insufficient_data'
    assert result['promotion_allowed'] is False
    assert result['development_events_verified']==0
    assert result['final_events_verified']==0
    assert result['metrics'] is None


def test_cohort_gate_requires_full_fold_and_final_coverage():
    from types import SimpleNamespace
    p=protocol()
    folds=[SimpleNamespace(train_event_ids=tuple(f"2024:{i}" for i in range(1,13)), test_event_ids=(event,)) for event in p.development_events]
    ids=(*p.development_events,*p.final_events)
    result=assess_readiness(p,folds=folds,verified_event_ids=ids)
    assert result['status']=='ready_for_evaluation'
    assert result['promotion_allowed'] is False
    assert result['metrics'] is None
