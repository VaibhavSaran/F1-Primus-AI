"""Populate local retrospective reports from saved inputs and official results.

This is descriptive reconstruction, not walk-forward model validation. Fetching
outcomes is explicit, after forecast inputs are fixed, and never feeds features.
"""
from datetime import datetime, timezone
from dataclasses import asdict
from hashlib import sha256
import argparse
import json
from pathlib import Path

from data.contracts import PredictionRequest
from flows.prediction_service import generate_forecast
from ml.evaluation import PROTOCOL_VERSION, evaluate_predictions
from monitoring.store import TrackingStore


def populate(snapshots, outcomes, directory):
    # A repeated download is a revision, not a second scored race.
    selected = {}
    for snapshot in snapshots:
        if snapshot.payloads.get('historical_reconstruction') is not True:
            continue
        digest = sha256(json.dumps(asdict(snapshot), sort_keys=True, default=str).encode()).hexdigest()
        key = (snapshot.as_of_utc, max((source.fetched_at_utc for source in snapshot.sources),
                                      default=snapshot.as_of_utc), digest)
        previous = selected.get(snapshot.event.event_id)
        if previous is None or key > previous[0]:
            selected[snapshot.event.event_id] = (key, snapshot)
    snapshots = [selected[event][1] for event in sorted(selected)]
    root = Path(directory)
    store = TrackingStore(root / 'tracking.sqlite3')
    rows_by_model = {'grid': [], 'weekend_form': []}
    events = []
    lineage = []
    for snapshot in snapshots:
        if snapshot.payloads.get('historical_reconstruction') is not True:
            continue
        outcome = outcomes.get(snapshot.event.event_id)
        if outcome is None:
            continue
        inputs = set(row['driver_id'] for row in snapshot.payloads['entries'])
        labels = {row['driver_id']: row for row in outcome['entries']}
        if set(labels) != inputs:
            raise ValueError('official result and reconstructed forecast fields differ')
        store.append_outcome(outcome)
        for model in rows_by_model:
            result = generate_forecast(PredictionRequest(snapshot.event, snapshot.payloads['mode'], snapshot.as_of_utc),
                snapshot, model_id=model, schedule=snapshot.payloads['schedule'], store=store)
            store.score_forecast(result['run_id'], outcome['revision_id'])
            for row in result['ranking']:
                label = labels[row['driver_id']]
                rows_by_model[model].append(dict(event_id=snapshot.event.event_id, driver_id=row['driver_id'],
                    prediction_score=row['prediction_score'], actual_position=label['actual_position'],
                    status=label['status'], outcome_status='official'))
            lineage.append(dict(run_id=result['run_id'], snapshot_id=snapshot.snapshot_id,
                                revision_id=outcome['revision_id'], model_id=model))
        events.append(snapshot.event.event_id)
    if not events:
        raise ValueError('no historical reconstructions with official results')
    report = dict(status='evaluated_reconstruction', evidence_kind='historical_reconstruction',
                  cohort_event_ids=sorted(set(events)), folds=[], final_holdout='Not evaluated',
                  protocol=dict(version=PROTOCOL_VERSION, type='descriptive event-time reconstruction'),
                  revision_policy='Latest cutoff, source fetch and content hash per event',
                  models={model: evaluate_predictions(rows, bootstrap_samples=200, seed=42)
                          for model, rows in rows_by_model.items()}, lineage=lineage,
                  readiness={'reasons': ['Original publication timing is unverified. These are hindsight reconstructions, not a measured live forecasting record.',
                                        f'Only {len(set(events))} available race(s): no model selection, calibration, promotion or generalization claim.']})
    encoded = json.dumps(report, sort_keys=True, indent=2, allow_nan=False)
    path = root / 'evaluations' / ('evaluation-reconstruction-' + sha256(encoded.encode()).hexdigest()[:24] + '.json')
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(encoded, encoding='utf-8')
    return path


def fetch_outcome(event, directory):
    import requests
    url = f'https://api.jolpi.ca/ergast/f1/{event.season}/{event.round_number}/results.json?limit=100'
    response = requests.get(url, timeout=30)
    response.raise_for_status()
    races = response.json()['MRData']['RaceTable']['Races']
    if not races:
        raise ValueError('official race results unavailable')
    entries = []
    for row in races[0]['Results']:
        original = row['status']
        status = 'DSQ' if original == 'Disqualified' else 'DNS' if original in {'Did not start', 'Did not qualify', 'Withdrew'} else original
        entries.append(dict(driver_id=row['Driver']['driverId'], actual_position=int(row['position']), status=status))
    digest = sha256(json.dumps(entries, sort_keys=True).encode()).hexdigest()[:24]
    outcome = dict(event_id=event.event_id, revision_id='jolpica-' + digest, status='official',
                   published_at_utc=datetime.now(timezone.utc).isoformat(), entries=entries,
                   source_url=url, publication_basis='Fetch time only; original publication unknown')
    path = Path(directory) / 'outcomes' / f'{event.season}-{event.round_number}.json'
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_file():
        previous = json.loads(path.read_text(encoding='utf-8'))
        if previous['revision_id'] == outcome['revision_id']:
            return previous
    path.write_text(json.dumps(outcome, indent=2), encoding='utf-8')
    return outcome


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fetch', action='store_true', required=True)
    parser.parse_args()
    from streamlit_app import services
    snapshots = [row for row in services.snapshots()[0] if row.payloads.get('historical_reconstruction')]
    outcomes = {row.event.event_id: fetch_outcome(row.event, services.LOCAL) for row in snapshots}
    print(populate(snapshots, outcomes, services.LOCAL))


if __name__ == '__main__':
    main()
