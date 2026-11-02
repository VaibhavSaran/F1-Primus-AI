"""Frozen evaluation intent and explicit coverage gates; never a promotion shortcut."""
from __future__ import annotations
from dataclasses import asdict, dataclass
from datetime import datetime
from hashlib import sha256
import json
from pathlib import Path
import re
from data.contracts import require_utc


@dataclass(frozen=True)
class EvaluationProtocol:
    development_events: tuple[str, ...]
    final_events: tuple[str, ...]
    frozen_at_utc: datetime
    version: str = 'f1-ranking-protocol-v1'
    primary_metric: str = 'rank_spearman'
    baseline: str = 'mode_matched_grid_or_rolling_form'
    practical_improvement: float = 0.02
    min_train_events: int = 12
    min_development_events: int = 12
    min_final_events: int = 6
    bootstrap_samples: int = 2000

    def __post_init__(self):
        object.__setattr__(self,'frozen_at_utc',require_utc(self.frozen_at_utc))
        object.__setattr__(self,'development_events',tuple(self.development_events))
        object.__setattr__(self,'final_events',tuple(self.final_events))
        events=(*self.development_events,*self.final_events)
        if any(not isinstance(e,str) or not re.fullmatch(r'[0-9]{4}:[1-9][0-9]*',e) for e in events):
            raise ValueError('invalid protocol event identity')
        if len(set(events))!=len(events):
            raise ValueError('protocol cohorts must be unique and disjoint')
        for name in ('min_train_events','min_development_events','min_final_events','bootstrap_samples'):
            value=getattr(self,name)
            if type(value) is not int or value<1:
                raise ValueError(f'{name} must be positive')
        if self.primary_metric!='rank_spearman':
            raise ValueError('this protocol implements ranking Spearman only')
        if not 0 < self.practical_improvement <= 2:
            raise ValueError('invalid practical improvement')
        if self.final_events and (self.development_events and max(map(_event_order,self.development_events))>=min(map(_event_order,self.final_events))):
            raise ValueError('final cohort must follow development cohort')


def _event_order(event):
    return tuple(map(int,event.split(':')))


def _content(protocol):
    value=asdict(protocol)
    value['frozen_at_utc']=protocol.frozen_at_utc.isoformat()
    return value


def _encoded(value):
    return json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()


def freeze_protocol(protocol, directory):
    payload=_content(protocol)
    digest=sha256(_encoded(payload)).hexdigest()
    root=Path(directory);root.mkdir(parents=True,exist_ok=True)
    path=root/f'protocol-{digest}.json'
    if path.exists():
        if load_protocol(path)!=protocol:
            raise ValueError('existing protocol conflicts')
        return path
    with path.open('x',encoding='utf-8') as stream:
        json.dump(dict(sha256=digest,protocol=payload),stream,indent=2,allow_nan=False)
    return path


def load_protocol(path):
    envelope=json.loads(Path(path).read_text(encoding='utf-8'))
    payload=envelope.get('protocol')
    if not isinstance(payload,dict) or sha256(_encoded(payload)).hexdigest()!=envelope.get('sha256'):
        raise ValueError('protocol integrity failed')
    payload=dict(payload)
    payload['frozen_at_utc']=datetime.fromisoformat(payload['frozen_at_utc'])
    return EvaluationProtocol(**payload)


def assess_readiness(protocol, *, folds, verified_event_ids):
    """Coverage gate only. Readiness never implies measured accuracy or promotion."""
    verified=set(verified_event_ids)
    eligible_tests={event for fold in folds if len(set(fold.train_event_ids))>=protocol.min_train_events
                    for event in fold.test_event_ids}
    dev=set(protocol.development_events)&verified&eligible_tests
    final=set(protocol.final_events)&verified
    reasons=[]
    if len(dev)<protocol.min_development_events:
        reasons.append('insufficient_verified_development_folds')
    if len(final)<protocol.min_final_events:
        reasons.append('insufficient_verified_final_events')
    return dict(status='insufficient_data' if reasons else 'ready_for_evaluation',
                promotion_allowed=False,metrics=None,reasons=reasons,
                development_events_verified=len(dev),final_events_verified=len(final),
                protocol_version=protocol.version,
                warning='Coverage is not model performance; evaluate frozen cohorts before promotion.')
