"""Offline native-router checks; browser/component performance is a separate gate."""
from datetime import datetime, timedelta, timezone
from hashlib import md5
import json
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from streamlit_app import services
from streamlit_app.screens import ROUTES

APP = Path(__file__).resolve().parents[1] / "streamlit_app" / "app.py"


@pytest.fixture(autouse=True)
def offline_ui(monkeypatch, tmp_path, request):
    # AppTest shares global config; earlier tests may first parse it outside the repo.
    import streamlit.config as config
    previous_address=config.get_option("server.address")
    config.set_option("server.address","localhost",where_defined="isolated test server")
    request.addfinalizer(lambda: config.set_option("server.address",previous_address))
    monkeypatch.setattr(services, "LOCAL", tmp_path)
    for name in ("ANTHROPIC_API_KEY", "GEMINI_API_KEY", "TAVILY_API_KEY", "OPENAI_API_KEY"):
        monkeypatch.delenv(name, raising=False)
    # Provider access from a render is a regression, including the empty state.
    import socket
    def forbidden(*args, **kwargs):
        raise AssertionError("UI rendering attempted a network connection")
    monkeypatch.setattr(socket, "create_connection", forbidden)
    monkeypatch.setattr(socket.socket, "connect", forbidden)


def page(slug="overview"):
    app = AppTest.from_file(str(APP))
    # AppTest 1.55 switch_page only accepts file pages; the router hashes explicit URL paths.
    app._page_hash = md5(slug.encode()).hexdigest()
    return app.run(timeout=20)


@pytest.mark.parametrize("group,label,slug,render", ROUTES)
def test_all_nine_native_routes_empty_without_credentials(group, label, slug, render):
    app = page(slug)
    assert not app.exception
    assert app.title[0].value == ('About Primus AI' if slug=='about' else label)
    if slug=='ask':
        assert any(item.label=='Additional context' for item in app.expander)
        assert not any(item.label=='Season' for item in app.selectbox)
    elif slug=='about':
        assert not app.selectbox
    else:
        assert [item.label for item in app.selectbox[:3]] == ["Season", "Event / round", "Session"]
    assert app.sidebar.title[0].value == "PRIMUS F1"
    assert not app.metric  # Absence of evidence must not turn into zero accuracy.


def test_grouped_navigation_contract():
    assert [item[0] for item in ROUTES] == ['Home'] + ["Weekend"] * 3 + ["Explore"] * 3 + ["Review"] * 2 + ["Preferences"]*2
    assert len({item[2] for item in ROUTES}) == 11


def test_forecast_models_and_no_snapshot_submission():
    app = page("predictions")
    model = app.selectbox(key="prediction_model")
    assert model.options == list(services.MODELS.values())
    assert 'tabfm' in services.MODELS and 'xgboost' in services.MODELS and 'timesfm' not in services.MODELS
    assert next(item for item in app.button if item.label == "Generate forecast").disabled
    model.select("ridge").run()
    assert not app.exception
    assert app.selectbox(key="artifact_ridge").options == []
    assert next(item for item in app.button if item.label == "Generate forecast").disabled
    assert not any("installed Python" in item.value for item in app.caption)


def test_context_change_invalidates_driver_and_run_selection():
    app = page()
    app.session_state["selected_drivers"] = ["driver-from-old-event"]
    app.session_state["active_run_id"] = "old-run"
    app.session_state["replay_commit"] = {"clock_ms": 100}
    app.selectbox(key="round_number").select(2).run()
    assert not app.exception
    assert app.session_state["workspace_context"]["round_number"] == 2
    for key in ("selected_drivers", "active_run_id", "replay_commit"):
        assert key not in app.session_state


def test_private_grounded_chat_enabled_with_external_providers_disabled():
    app = page("ask")
    assert not app.chat_input[0].disabled
    assert any("local evidence mode" in item.value.lower() for item in app.caption)
    assert not app.chat_message


def test_comparison_profiles_and_validation_views():
    comparison = page("compare")
    assert [item.label for item in comparison.tabs] == ["Performance", "Driver profiles", "Team profiles"]
    validation = page("validation")
    assert [item.label for item in validation.tabs] == ["Evaluation", "Performance Tracking", "Historical Race Studies"]
    assert any("never scored as zero" in item.value for item in validation.info)


def test_legacy_report_is_explicitly_unvalidated():
    folder=services.LOCAL/'legacy_reports'
    folder.mkdir()
    (folder/'historical-preview.md').write_text('# Earlier narrative',encoding='utf-8')
    app = page("reports")
    assert any("Historical narratives" in item.value for item in app.warning)
    assert next(item for item in app.selectbox if item.label == "Legacy report").options


def test_saved_snapshot_context_requires_exact_event_and_cutoff():
    from data.contracts import DataSnapshot, EventKey
    cutoff = datetime(2025, 7, 5, 18, tzinfo=timezone.utc)
    snapshot = DataSnapshot("test", EventKey(2025, 12), cutoff)
    selected = dict(season=2025, round_number=12, cutoff=cutoff)
    assert services.matches(snapshot, selected)
    assert not services.matches(snapshot, dict(selected, round_number=11))
    assert not services.matches(snapshot, dict(selected, cutoff=cutoff.replace(hour=17)))


def test_provider_readiness_never_exposes_secret(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "secret-value-that-must-not-render")
    app = page("settings")
    assert not app.exception
    assert "secret-value-that-must-not-render" not in str(app)
    assert "secret-value-that-must-not-render" not in str(services.dependency_readiness())


def saved_input(tmp_path):
    from data.contracts import DataSnapshot, DriverSessionRecord, EventKey, SourceRecord
    from data.sessions import EventInfo, SessionRecord
    from data.snapshots import save_snapshot
    cutoff = datetime(2026, 3, 7, 18, tzinfo=timezone.utc)
    event = EventKey(2026, 1, "albert_park")
    source = SourceRecord("fixture", "verified-grid", cutoff - timedelta(hours=1))
    schedule = EventInfo(event, "Fixture GP", "Test circuit", (
        SessionRecord(event, "Q", "Qualifying", cutoff - timedelta(hours=4), cutoff - timedelta(hours=3)),
        SessionRecord(event, "R", "Race", cutoff + timedelta(days=1))))
    snapshot = DataSnapshot("ui-verified", event, cutoff, payloads={
        "mode": "post_qualifying", "schedule": schedule, "entries_source": source,
        "entries": [{"driver_id": "driver-a", "team_id": "team-1"}, {"driver_id": "driver-b", "team_id": "team-2"}],
        "driver_sessions": [DriverSessionRecord(event, "GRID", driver, source.fetched_at_utc,
                                                {"grid_position": position}, source)
                            for driver, position in (("driver-a", 2), ("driver-b", 1))]}, sources=[source])
    save_snapshot(snapshot, tmp_path / "snapshots")
    return snapshot


def test_submit_persists_once_rerender_does_not_regenerate_and_exports_agree(monkeypatch, tmp_path):
    import streamlit as st
    snapshot = saved_input(tmp_path)
    calls, downloads = [], {}
    actual = services.generate
    def generate(*args, **kwargs):
        calls.append(args)
        return actual(*args, **kwargs)
    original_download = st.download_button
    def download(label, data, *args, **kwargs):
        downloads[label] = data
        return original_download(label, data, *args, **kwargs)
    monkeypatch.setattr(services, "generate", generate)
    monkeypatch.setattr(st, "download_button", download)
    app = page("predictions")
    assert not next(button for button in app.button if button.label == "Generate forecast").disabled
    next(button for button in app.button if button.label == "Use saved cutoff & mode").click().run()
    assert not app.exception
    submit = next(button for button in app.button if button.label == "Generate forecast")
    assert not submit.disabled
    submit.click().run()
    assert not app.exception
    assert len(calls) == 1
    run = app.session_state["visible_forecast"]
    assert run["snapshot_id"] == snapshot.snapshot_id
    assert [row["driver_id"] for row in run["ranking"]] == ["driver-b", "driver-a"]
    assert json.loads(downloads["Download JSON"]) == run
    import csv
    from io import StringIO
    assert [row["driver_id"] for row in csv.DictReader(StringIO(downloads["Download ranking CSV"]))] == ["driver-b", "driver-a"]
    assert downloads["Download briefing"] == run["briefing"]
    app.run()
    assert len(calls) == 1
    next(button for button in app.button if button.label == "Generate forecast").click().run()
    assert not app.exception
    assert len(services.tracking_queries().forecasts()) == 1
    assert len(calls) == 2
    reports = page("reports")
    assert not reports.exception
    assert next(x for x in reports.selectbox if x.label=='Saved forecast run').value == run["run_id"]
    validation = page("validation")
    assert not validation.exception
    summary = services.tracking_queries().summary()["groups"][0]
    assert summary["n_pending_events"] == 1
    assert summary["metrics"]["position_mae"] is None


def test_corrupt_snapshot_is_excluded_and_empty_state_usable(tmp_path):
    directory = tmp_path / "snapshots"
    directory.mkdir()
    (directory / "bad.json").write_text("not a valid snapshot", encoding="utf-8")
    app = page("predictions")
    assert not app.exception
    assert any("unreadable saved snapshot" in item.value for item in app.warning)
    assert next(button for button in app.button if button.label == "Generate forecast").disabled


def test_changed_controls_label_previous_run_and_no_implicit_prediction(tmp_path):
    saved_input(tmp_path)
    app = page("predictions")
    next(button for button in app.button if button.label == "Use saved cutoff & mode").click().run()
    next(button for button in app.button if button.label == "Generate forecast").click().run()
    assert not app.exception
    original_run = app.session_state["visible_forecast"]["run_id"]
    app.selectbox(key="prediction_model").select("rolling_form").run()
    assert not app.exception
    assert any("Previous saved run" in item.value for item in app.warning)
    assert app.session_state["visible_forecast"]["run_id"] == original_run
    assert len(services.tracking_queries().forecasts()) == 1


def test_invalid_tracking_store_retains_other_pages(tmp_path):
    (tmp_path / "tracking.sqlite3").write_text("invalid database", encoding="utf-8")
    app = page("validation")
    assert not app.exception
    assert any("tracking store is unavailable" in item.value for item in app.warning)


def test_profiles_exclude_future_or_unsourced_evidence(tmp_path):
    from dataclasses import replace
    from data.snapshots import save_snapshot
    snapshot = saved_input(tmp_path)
    payload = dict(snapshot.payloads, driver_profiles=[
        {"driver_id": "visible", "available_at_utc": snapshot.as_of_utc.isoformat(), "source_ids": ["profile-source"]},
        {"driver_id": "future", "available_at_utc": (snapshot.as_of_utc + timedelta(days=1)).isoformat(), "source_ids": ["future-source"]},
        {"driver_id": "unsourced", "available_at_utc": snapshot.as_of_utc.isoformat()}])
    save_snapshot(replace(snapshot, snapshot_id="ui-profiles", payloads=payload), tmp_path / "snapshots")
    app = page("compare")
    app.selectbox(key="comparison_snapshot").select("ui-profiles").run()
    assert not app.exception
    profiles = [item.value for item in app.dataframe if "driver_id" in item.value.columns]
    assert len(profiles) == 1
    assert profiles[0]["driver_id"].tolist() == ["visible"]


def test_remote_binding_stops_before_private_workspace_reads(monkeypatch):
    import streamlit.config as config
    config.set_option('server.address','0.0.0.0',where_defined='test remote binding')
    monkeypatch.setattr(services,'snapshots',lambda:pytest.fail('remote private workspace read'))
    app=page('overview')
    assert not app.exception
    assert any('verified access' in item.value.lower() for item in app.error)
    assert not app.selectbox


def test_ask_f1_local_pilot_responds_without_provider_or_raw_prompt_logging():
    app=page('ask')
    app.chat_input[0].set_value('Explain this F1 model forecast').run()
    assert not app.exception
    assert len(app.chat_message)==2
    assert any('sufficient saved evidence' in item.value.lower() for item in app.markdown)
    query=services.tracking_queries()
    from monitoring.queries import TrackingQueries
    operator=TrackingQueries(query.store,owner_id='local',role='operator')
    assert operator.operational_summary()['n_events']==1


def test_strategy_route_connects_observed_and_scenario_views():
    app=page('strategy')
    assert not app.exception
    assert any('No normalized historical source matches' in item.value for item in app.info)


def test_forecast_cutoff_and_mode_survive_navigation_cleanup(tmp_path):
    snapshot=saved_input(tmp_path)
    app=page('predictions')
    next(button for button in app.button if button.label=='Use saved cutoff & mode').click().run()
    assert app.session_state['forecast_request_context']['cutoff']==snapshot.as_of_utc
    app._page_hash=md5(b'reports').hexdigest();app.run()
    app._page_hash=md5(b'predictions').hexdigest();app.run()
    assert not app.exception
    assert app.date_input(key='cutoff_date').value==snapshot.as_of_utc.date()
    assert app.selectbox(key='forecast_mode').value=='post_qualifying'


def test_comparison_builds_sourced_profiles_from_snapshot_history(tmp_path):
    from dataclasses import replace
    from data.snapshots import save_snapshot
    snapshot=saved_input(tmp_path)
    history=[dict(event_id='2025:1',driver_id='driver-a',team_id='team-1',actual_position=3,status='FINISHED',
                  event_time_utc='2025-03-01T00:00:00Z',outcome_available_at_utc='2025-03-02T00:00:00Z',source_ids=['verified-result'])]
    save_snapshot(replace(snapshot,snapshot_id='ui-history',payloads=dict(snapshot.payloads,history=history)),tmp_path/'snapshots')
    app=page('compare')
    app.selectbox(key='comparison_snapshot').select('ui-history').run()
    assert not app.exception
    profiles=[item.value for item in app.dataframe if 'driver_id' in item.value.columns and 'recent_form' in item.value.columns]
    assert profiles and profiles[0]['driver_id'].tolist()==['driver-a']


def test_ui_service_uses_saved_history_and_excludes_future_results():
    from test_prediction_service import history_inputs
    request,snapshot,schedule=history_inputs()
    result=services.generate(dict(season=2025,round_number=12,cutoff=request.as_of_utc,mode=request.mode.value),snapshot,'rolling_form')
    assert [(r['driver_id'],r['prediction_score'],r['baseline_source']) for r in result['ranking']]==[('B',1.0,'driver_form'),('A',2.0,'driver_form')]
