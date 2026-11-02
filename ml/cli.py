"""Explicit operator-only offline model jobs; no service or provider side effects."""
from __future__ import annotations
import argparse
from datetime import datetime
from hashlib import sha256
import json
from pathlib import Path
from ml.artifacts import ArtifactStore
from ml.training import TrainingConfig,train_model
from ml.benchmark import benchmark
from ml.protocols import load_protocol


def main(argv=None):
    parser=argparse.ArgumentParser(description='Offline research training and frozen development evaluation')
    sub=parser.add_subparsers(dest='action',required=True)
    train=sub.add_parser('train')
    train.add_argument('--dataset',required=True)
    train.add_argument('--candidate',choices=['ridge','random_forest','gradient_boosting','xgboost'],default='ridge')
    train.add_argument('--target',choices=['finishing_position','clean_pace'],default='finishing_position')
    evaluate=sub.add_parser('benchmark')
    evaluate.add_argument('--dataset',required=True)
    evaluate.add_argument('--protocol',required=True)
    evaluate.add_argument('--mode',choices=['pre_weekend','post_qualifying'],required=True)
    args=parser.parse_args(argv)
    rows=json.loads(Path(args.dataset).read_text(encoding='utf-8'))
    if not isinstance(rows,list):raise ValueError('Dataset must be a JSON list of audited event-driver rows')
    if args.action=='train':
        artifact=train_model(rows,TrainingConfig(candidate=args.candidate,target=args.target))
        model_id=ArtifactStore('.local/models').save(artifact)
        print(json.dumps(dict(model_id=model_id,validation_status=artifact.metadata['validation_status'],
                training_availability_verified=artifact.metadata['training_availability_verified'],production_eligible=False)))
    else:
        protocol=load_protocol(args.protocol)
        result=benchmark(rows,protocol,mode=args.mode)
        def encode(value):
            if isinstance(value,datetime):return value.isoformat()
            raise TypeError('Unknown evaluation value')
        encoded=json.dumps(result,sort_keys=True,indent=2,default=encode,allow_nan=False)
        root=Path('.local/evaluations');root.mkdir(parents=True,exist_ok=True)
        path=root/f"evaluation-{sha256(encoded.encode()).hexdigest()}.json"
        if not path.exists():path.write_text(encoded,encoding='utf-8')
        print(json.dumps(dict(status=result['status'],path=str(path),promotion_allowed=False)))


if __name__=='__main__':main()
