"""Read boundaries for trusted authenticated identity, owner, and public classification."""
from __future__ import annotations

from collections import defaultdict
from hashlib import sha256
from math import ceil
from statistics import mean

from ml.evaluation import PROTOCOL_VERSION, evaluate_predictions
from monitoring.store import _identifier, _rows, _timestamp


class TrackingQueries:
    """Construct with server-established identity/role, never browser role claims."""

    def __init__(self, store, *, owner_id=None, role="viewer"):
        if role not in {"viewer", "ai_user", "operator"}:
            raise ValueError("unknown tracking role")
        if owner_id is not None:
            _identifier(owner_id, "owner_id")
        self.store, self.owner_id, self.role = store, owner_id, role

    def forecasts(self, *, model_id=None, mode=None, as_of_utc=None):
        cutoff = _timestamp(as_of_utc, "as_of_utc") if as_of_utc is not None else None
        return sorted([row for row in self.store._all("forecasts")
                       if (self.role == "operator" or row["is_public"] or
                           (self.owner_id is not None and row["owner_id"] == self.owner_id))
                       and (model_id is None or row["model_id"] == model_id)
                       and (mode is None or row["mode"] == mode)
                       and (cutoff is None or (row["as_of_utc"] <= cutoff and row["recorded_at_utc"] <= cutoff))],
                      key=lambda row: (row["as_of_utc"], row["run_id"]))

    def get_forecast(self, run_id):
        return next((row for row in self.forecasts() if row["run_id"] == run_id), None)

    def outcomes(self, event_id):
        if not any(row["event_id"] == event_id for row in self.forecasts()):
            return []
        return sorted([row for row in self.store._all("outcomes") if row["event_id"] == event_id],
                      key=lambda row: (row["published_at_utc"], row["revision_id"]))

    def score_history(self, run_id):
        if self.get_forecast(run_id) is None:
            return []
        return sorted([row for row in self.store._all("scores") if row["run_id"] == run_id],
                      key=lambda row: (row["scored_at_utc"], row["score_id"]))

    def summary(self, *, model_id=None, mode=None, window=None, as_of_utc=None, event_ids=None,
                bootstrap_samples=200, seed=0, evidence_kind="forecast"):
        """Present/historical race macros, latest cutoff per event/model/mode.

        Hindsight reconstructions are excluded unless explicitly requested as a
        separate evidence cohort. A newer published outcome makes prior scores stale until replayed. Window
        counts the latest event cutoffs, including pending events; missing metrics
        retain None and their own event denominators. Historical time uses outcome
        recording, publication and score times. Historical reconstruction never
        claims a forecast existed before it was saved.
        """
        if evidence_kind not in {"forecast", "historical_reconstruction"}:
            raise ValueError("evidence_kind must be forecast or historical_reconstruction")
        if window is not None and (type(window) is not int or window < 1):
            raise ValueError("window must be a positive integer")
        cutoff = _timestamp(as_of_utc, "as_of_utc") if as_of_utc is not None else None
        groups = defaultdict(list)
        for forecast in self.forecasts(model_id=model_id, mode=mode, as_of_utc=as_of_utc):
            retrospective = forecast.get("provenance", {}).get("validation_status") == "historical_reconstruction"
            if retrospective != (evidence_kind == "historical_reconstruction"):
                continue
            if event_ids is None or forecast["event_id"] in event_ids:
                groups[(forecast["model_id"], forecast["mode"])].append(forecast)
        outcomes = self.store._all("outcomes")
        scores = self.store._all("scores")
        result = []
        for (model, forecast_mode), forecasts in sorted(groups.items()):
            selected = {}
            for forecast in sorted(forecasts, key=lambda row: (row["as_of_utc"], row["recorded_at_utc"], row["run_id"])):
                selected[forecast["event_id"]] = forecast
            chosen = sorted(selected.values(), key=lambda row: (row["as_of_utc"], row["event_id"]))
            if window:
                chosen = chosen[-window:]
            rows, pending, revision_ids = [], 0, {}
            for forecast in chosen:
                revisions = [row for row in outcomes if row["event_id"] == forecast["event_id"] and
                             (cutoff is None or row["published_at_utc"] <= cutoff)]
                outcome = max(revisions, key=lambda row: (row["published_at_utc"], row["revision_id"])) if revisions else None
                score = next((row for row in scores if row["run_id"] == forecast["run_id"] and outcome and
                              row["revision_id"] == outcome["revision_id"] and
                              row["protocol_version"] == PROTOCOL_VERSION and
                              (cutoff is None or row["scored_at_utc"] <= cutoff)), None)
                available = score and score["evaluation"]["outcome_status"] == "completed"
                pending += not bool(available)
                revision_ids[forecast["event_id"]] = outcome["revision_id"] if outcome else None
                rows.extend(_rows(forecast, outcome if score else None))
            directions = {row.get("higher_score_better", False) for row in chosen}
            if len(directions) > 1:
                raise ValueError("model/mode summaries require a consistent score direction")
            evaluation = evaluate_predictions(rows, higher_score_better=next(iter(directions), False),
                                              bootstrap_samples=bootstrap_samples, seed=seed)
            result.append(dict(model_id=model, mode=forecast_mode, n_forecasts=len(forecasts),
                               n_pending_events=pending, n_scored_events=len(chosen) - pending,
                               selected_run_ids=[row["run_id"] for row in chosen],
                               outcome_revisions=revision_ids, **evaluation))
        return dict(groups=result, as_of_utc=cutoff, window=window, evidence_kind=evidence_kind,
                    cohort_policy="latest cutoff per event/model/mode; race macro")

    def compare_models(self, model_ids, *, mode=None, common_events=True, **summary_options):
        model_ids = list(dict.fromkeys(model_ids))
        if len(model_ids) < 2:
            raise ValueError("compare at least two model versions")
        own_summaries = {model: self.summary(model_id=model, mode=mode, **summary_options)["groups"]
                         for model in model_ids}
        modes = {group["mode"] for groups in own_summaries.values() for group in groups}
        if len(modes) > 1:
            raise ValueError("specify one forecast mode for model comparison")
        cohorts = [{race["event_id"] for group in own_summaries[model] for race in group["per_race"]
                    if race["outcome_status"] == "completed"} for model in model_ids]
        intersection = set.intersection(*cohorts)
        groups = []
        for model in model_ids:
            groups.extend(self.summary(model_id=model, mode=mode, event_ids=intersection,
                                       **summary_options)["groups"] if common_events else own_summaries[model])
        return dict(groups=groups, common_event_ids=sorted(intersection), common_events=common_events)

    def outcome_state(self, run_id):
        """Distinguish pending evidence from a known incompatible result field."""
        run = self.get_forecast(run_id)
        if run is None:
            return None
        outcomes = self.outcomes(run['event_id'])
        if not outcomes:
            return 'pending'
        latest = max(outcomes, key=lambda row: (row['published_at_utc'], row['revision_id']))
        if {row['driver_id'] for row in run['ranking']} != {row['driver_id'] for row in latest['entries']}:
            return 'roster_mismatch'
        completed = any(row['revision_id'] == latest['revision_id'] and row.get('evaluation', {}).get('outcome_status') == 'completed'
                        for row in self.score_history(run_id))
        return 'completed' if completed else 'unscored'

    def request_health(self):
        """Owner-only aggregate health; raw events and costs remain operator-only."""
        owners = {self.owner_id, 'chat-' + sha256(self.owner_id.encode()).hexdigest()[:24]} if self.owner_id else set()
        events = [row for row in self.store._all('operational_events')
                  if row['owner_id'] in owners]
        latencies = sorted(row['latency_ms'] for row in events
            if row['status'] in {'success', 'error', 'timeout'} and row.get('latency_ms') is not None)
        return dict(requests=len(events), failures=sum(row['status'] in {'error', 'timeout'} for row in events),
            denied=sum(row['status'] == 'denied' for row in events),
            p95_latency_ms=latencies[max(0, ceil(len(latencies) * .95) - 1)] if latencies else None)

    def _operator(self):
        if self.role != "operator":
            raise PermissionError("operational history and costs require operator access")

    def operational_events(self):
        self._operator()
        return sorted(self.store._all("operational_events"), key=lambda row: (row["occurred_at_utc"], row["event_id"]))

    def operational_summary(self):
        """Dispatch latency covers measured success/error/timeout events only."""
        events = self.operational_events()
        authorized = [row for row in events if row["status"] != "denied"]
        latencies = sorted(row["latency_ms"] for row in events if row["status"] in {"success", "error", "timeout"}
                           and row.get("latency_ms") is not None)
        denied_latencies = sorted(row["latency_ms"] for row in events if row["status"] == "denied"
                                  and row.get("latency_ms") is not None)
        def percentile(values, fraction):
            return values[max(0, ceil(len(values) * fraction) - 1)] if values else None
        def total(key):
            values = [row[key] for row in events if row.get(key) is not None]
            return sum(values) if values else None
        return dict(n_events=len(events), p50_latency_ms=percentile(latencies, .5), p95_latency_ms=percentile(latencies, .95),
                    latency_event_count=len(latencies),
                    denied_p50_latency_ms=percentile(denied_latencies, .5),
                    denied_p95_latency_ms=percentile(denied_latencies, .95),
                    denied_latency_event_count=len(denied_latencies),
                    failure_rate=mean(row["status"] in {"error", "timeout"} for row in authorized) if authorized else None,
                    timeout_count=sum(row["status"] == "timeout" for row in events),
                    denied_count=sum(row["status"] == "denied" for row in events),
                    input_tokens=total("input_tokens"), output_tokens=total("output_tokens"),
                    estimated_cost=total("estimated_cost"), billed_cost=total("billed_cost"),
                    estimated_cost_event_count=sum(row.get("estimated_cost") is not None for row in events),
                    billed_cost_event_count=sum(row.get("billed_cost") is not None for row in events))
