"""Nine native screens consuming saved, versioned local evidence."""
from __future__ import annotations

import csv
from datetime import datetime
from io import StringIO
import sqlite3
from zoneinfo import ZoneInfo
import streamlit as st
from streamlit_app import services
from streamlit_app.components.context import forecast_context


def context():
    return st.session_state.get("workspace_context", {"season": 2026, "round_number": 1, "session": "R"})


def heading(title, description, *, require_context=True):
    from data.catalog import session_label
    title_area = st.session_state.get('_page_heading_area')
    selected = context()
    session_title = selected.get('session_name') or session_label(selected.get('session'))
    event_title = f"{selected['event_name']} · {session_title}" if selected.get('event_name') and selected.get('session') else None
    with title_area if title_area is not None else st.container():
        st.title(event_title if title == 'Race Overview' and event_title else title)
        st.caption(description)
    if require_context and (not selected.get('round_number') or not selected.get('session')):
        st.info('Select a race and session to continue.')
        st.stop()
    if event_title and title != 'Race Overview':
        st.subheader(event_title)


def event_snapshots():
    records, errors = services.snapshots_for_event(context())
    if errors:
        st.warning(f"{errors} unreadable saved snapshot(s) excluded. Other evidence remains usable.")
    selected = context()
    return sorted([item for item in records if (item.event.season, item.event.round_number) ==
            (selected["season"], selected["round_number"])], key=lambda item: (item.snapshot_id.startswith('pending-'),item.as_of_utc,
              max((source.fetched_at_utc for source in item.sources), default=item.as_of_utc)), reverse=True)


def choose_snapshot(key):
    by_id = {}
    for item in event_snapshots():
        by_id.setdefault(item.snapshot_id, item)
    selected = st.selectbox("Saved input snapshot", list(by_id), index=0 if by_id else None,
                            placeholder="No snapshot saved for this event", key=key,
                            format_func=lambda value: f"{'Prepared pre-race inputs · forecasts ready' if value.startswith('pending-') else 'Historical reconstruction' if by_id[value].payloads.get('historical_reconstruction') else 'Saved inputs'} · {by_id[value].as_of_utc:%d %b %Y %H:%M} UTC")
    return by_id.get(selected)


def source_details(snapshot):
    zone = ZoneInfo(st.session_state.get("display_timezone", "UTC"))
    st.caption(f"Saved input cutoff: {snapshot.as_of_utc.astimezone(zone):%d %b %Y %H:%M %Z} · {len(snapshot.sources)} source records")
    for flag in snapshot.quality_flags:
        st.warning(flag.replace("_", " "))
    if snapshot.sources:
        st.dataframe([{"Provider": item.provider,
                       "Fetched (UTC)": item.fetched_at_utc, "Available (UTC)": item.availability_time_utc}
                      for item in snapshot.sources], hide_index=True, width="stretch")


def tracking():
    try:
        return services.tracking_queries()
    except (ImportError, OSError, ValueError, sqlite3.Error):
        st.warning("The saved tracking store is unavailable. No quality scores were substituted.")
        return None


def render_run(run, prefix):
    labels = services.driver_labels(run.get('event_id', ''))
    model = str(run.get('model_id', 'Unknown model')).split('-finishing_position-')[0]
    st.subheader("Saved forecast")
    st.caption(f"{services.MODELS.get(model, model)} · {str(run.get('status', 'Research result')).replace('_', ' ').title()}")
    with st.expander("Original context & provenance"):
        st.write(f"Input cutoff: {run.get('as_of_utc', 'Unavailable')}")
        st.write(f"Mode: {str(run.get('mode', 'Unavailable')).replace('_', ' ').title()}")
        st.write(f"Evidence: {str(run.get('provenance', {}).get('validation_status', 'Research')).replace('_', ' ').title()}")
        st.caption('The downloadable record retains the exact run, input and model references.')
    warnings = run.get('warnings', [])
    if warnings:
        st.caption('Research output · model and input limitations apply.')
        with st.expander('Data quality & model limitations'):
            for warning in warnings:
                st.write('• ' + str(warning).replace('_', ' '))
    ranking = run.get("ranking") or []
    tabs = st.tabs(["Finishing order", "Evidence", "Briefing"])
    with tabs[0]:
        if ranking:
            st.dataframe([{'Predicted position': row.get('predicted_rank'),
                'Driver': labels.get(row.get('driver_id'), row.get('driver_name') or row.get('driver_id')),
                'Ranking score': row.get('prediction_score')} for row in ranking], hide_index=True, width="stretch")
            ids = [row["driver_id"] for row in ranking if row.get("driver_id")]
            focus = st.selectbox("Focus driver", ids, format_func=lambda value: labels.get(value, value), key=f"{prefix}_driver")
            focused = next(row for row in ranking if row['driver_id'] == focus)
            st.write(f"{labels.get(focus, focus)} · estimated finishing position {focused.get('predicted_rank', 'unavailable')}")
            missing = run.get('provenance', {}).get('missing_inputs', {}).get(focus, [])
            if missing:
                st.caption('Unavailable inputs for this driver: ' + ', '.join(str(value).replace('_', ' ') for value in missing))
            st.caption("Scores follow the saved model target. Probabilities and intervals are unavailable unless explicitly supplied.")
        else:
            st.info("No eligible ranking is saved for this run.")
    with tabs[1]:
        missing = run.get('provenance', {}).get('missing_inputs', {})
        if isinstance(missing, dict) and missing:
            st.dataframe([{'Driver': labels.get(driver, driver),
                'Missing inputs': ', '.join(str(value).replace('_', ' ') for value in fields) or 'None'}
                for driver, fields in missing.items()], hide_index=True, width='stretch')
        else:
            st.info('Detailed input coverage is retained in the downloadable record when available.')
        st.caption(f"{len(run.get('provenance', {}).get('source_ids', []))} source records underpin this result. Missing inputs are not filled with invented facts.")
    with tabs[2]:
        if run.get("briefing"):
            st.markdown(str(run["briefing"]))
        else:
            st.info("No grounded briefing is saved. Numeric results remain available above.")
    st.download_button("Download JSON", services.json_export(run), file_name=f"{run.get('run_id', 'forecast')}.json",
                       mime="application/json", key=f"{prefix}_json")
    if ranking:
        output = StringIO()
        writer = csv.DictWriter(output, list(dict.fromkeys(key for row in ranking for key in row)))
        writer.writeheader()
        writer.writerows(ranking)
        st.download_button("Download ranking CSV", output.getvalue(), file_name="ranking.csv",
                           mime="text/csv", key=f"{prefix}_csv")
    if run.get("briefing"):
        st.download_button("Download briefing", str(run["briefing"]), file_name="briefing.md",
                           mime="text/markdown", key=f"{prefix}_md")


def overview():
    heading("Race Overview", "Classification, lap timing and driver entries for the selected session.")
    if context().get('archive_session_id'):
        from streamlit_app.archive_views import overview as archived
        return archived(context())
    records = event_snapshots()
    snapshot = records[0] if records else None
    panels = [("Session evidence", "Saved" if snapshot else "Unavailable"),
              ("Driver coverage", str(len(snapshot.payloads.get("entries", []))) + " saved entries" if snapshot else "Unknown"),
              ("Weather / tyres / pit loss", "Inspect snapshot" if snapshot else "Missing evidence"),
              ("Model validation", "No promotion claimed")]
    columns = st.columns(1 if st.session_state.get("compact_view") else 4)
    for index, (label, value) in enumerate(panels):
        column = columns[index % len(columns)]
        with column.container(border=True):
            st.markdown(f"**{label}**")
            st.write(value)
    if snapshot:
        source_details(snapshot)
        if snapshot.payloads.get("entries"):
            st.subheader("Saved event entries")
            st.dataframe(snapshot.payloads["entries"], hide_index=True, width="stretch")
        else:
            st.info("An explicit event entry list is required before forecasting.")
    else:
        st.info("No versioned input snapshot is saved for this event. Import verified data through the local data pipeline before forecasting.")
    inventory = [row for row in services.session_inventory() if (row['season'], row['round_number']) ==
                 (context()['season'], context()['round_number'])]
    if inventory:
        from data.catalog import session_label
        st.subheader('Historical session coverage')
        st.dataframe([{'Session': session_label(row['session_id']), 'Drivers': row['drivers'],
                       'Recorded laps': row['laps'], 'Position samples': row['positions']} for row in inventory],
                     hide_index=True, width='stretch')
    if context().get('event_name'):
        st.caption(f"{context()['event_name']} · {context().get('race_date')} · calendar supplied by FastF1")
    st.page_link(st.Page(compare_drivers, url_path="compare"), label="Compare Drivers")


def predictions():
    heading("Predictions", "Forecast the final Grand Prix finishing order using earlier weekend sessions. Sprint is an input, never the target.")
    snapshot = choose_snapshot("prediction_snapshot")
    prepared = services.pending_snapshot_for(context())
    if prepared and snapshot and prepared.snapshot_id != snapshot.snapshot_id:
        st.info('Forecasts are ready under a separate prepared pre-race input snapshot. This historical snapshot has a different cutoff.')
        def use_prepared():
            st.session_state['prediction_snapshot'] = prepared.snapshot_id
            st.session_state.pop('forecast_input_snapshot', None)
        st.button('Use forecast-ready inputs',on_click=use_prepared)
    if snapshot and st.session_state.get('forecast_input_snapshot') != snapshot.snapshot_id:
        st.session_state['forecast_input_snapshot'] = snapshot.snapshot_id
        st.session_state['forecast_request_context'] = dict(mode=snapshot.payloads.get('mode', 'post_qualifying'), cutoff=snapshot.as_of_utc)
        st.session_state['forecast_mode'] = snapshot.payloads.get('mode', 'post_qualifying')
        st.session_state['cutoff_date'] = snapshot.as_of_utc.date()
        st.session_state['cutoff_time'] = snapshot.as_of_utc.time().replace(tzinfo=None)
    selected = services.bind_saved_cutoff(snapshot, forecast_context(context()))
    pending_model_panel(snapshot,'predictions')
    default_model = 'qualifying' if snapshot and snapshot.payloads.get('historical_reconstruction') else 'grid'
    model = st.selectbox("Prediction model", list(services.MODELS), index=list(services.MODELS).index(default_model), format_func=services.MODELS.get, key="prediction_model")
    foundation = model == 'tabfm'
    saved_forecast = services.cached_forecast(snapshot, model)
    artifact = None
    if saved_forecast:
        st.success('A saved forecast is ready for these exact inputs. Generate forecast opens it without repeating model inference.')
    elif foundation:
        receipts = services.foundation_receipts(snapshot,model)
        if not receipts:
            st.info('TabFM has no saved result for these inputs. Choose forecast-ready inputs above when available, or select a baseline model.')
        artifact = st.selectbox('Completed research inference',[name for name,_ in receipts],
            index=0 if receipts else None,placeholder='No matching completed inference',
            format_func=lambda value:'Version '+value[:8],key=f'foundation_receipt_{model}')
    if not saved_forecast and model in {"ridge", "random_forest", "gradient_boosting",'xgboost'}:
        ids = services.eligible_artifact_ids(model, snapshot)
        artifact = st.selectbox("Fitted local artifact", ids, index=0 if ids else None,
                                placeholder="No eligible fitted version", key=f"artifact_{model}",
                                format_func=lambda value: 'Version ' + value[-8:])
        if not ids:
            st.info("This candidate needs an eligible fitted local artifact. Training is an operator action outside this page.")
    if snapshot:
        if snapshot.snapshot_id.startswith('pending-'):
            st.info('Inputs were saved before the Grand Prix. These forecasts await the actual race result and have not been scored.')
        elif snapshot.payloads.get('historical_reconstruction'):
            st.warning('Historical reconstruction · uses completed sessions before the Grand Prix. Original publication timing is unverified; this is not a validated backtest.')
        sessions = sorted({row.session_id for row in snapshot.payloads.get('driver_sessions', [])})
        from data.catalog import session_label
        st.caption('Target: Grand Prix Race · Inputs: ' + ', '.join(session_label(value) for value in sessions))
        st.caption(f"Field: {len(snapshot.payloads.get('entries', []))} recorded input entrants. Final starters and changes after the cutoff may differ.")
        if model == 'weekend_form':
            st.caption('Research heuristic: qualifying 60%, practice pace rank 30%, sprint finish 10%. Available weights are renormalized. No winner probabilities or measured accuracy claimed.')
        with st.expander('Input coverage & provenance'):
            source_details(snapshot)
        def use_saved_cutoff():
            st.session_state["cutoff_date"] = snapshot.as_of_utc.date()
            st.session_state["cutoff_time"] = snapshot.as_of_utc.time().replace(tzinfo=None)
            if snapshot.payloads.get("mode") in {"pre_weekend", "post_qualifying"}:
                st.session_state["forecast_mode"] = snapshot.payloads["mode"]
        st.button("Use saved cutoff & mode", on_click=use_saved_cutoff)
    matching = snapshot is not None and services.matches(snapshot, selected) and selected['mode'] == snapshot.payloads.get('mode')
    if snapshot and not matching:
        st.warning("Snapshot event/cutoff differs from these controls. Match its original UTC cutoff before submitting.")
    supported = bool(saved_forecast) or (bool(artifact) if foundation else (not snapshot or not snapshot.payloads.get('historical_reconstruction') or model in {'qualifying','grid','rolling_form','weekend_form'} or bool(artifact)))
    if snapshot and snapshot.payloads.get('historical_reconstruction') and not supported and not foundation:
        st.info('This event has no matching learned model trained only on earlier races. Baselines remain available.')
    enabled = matching and supported and (bool(saved_forecast) or model in {"qualifying", "grid", "rolling_form", "weekend_form"} or bool(artifact))
    with st.form("forecast_submission"):
        st.caption("Data → input checks → prediction → saved result. Only actual service completion is reported.")
        submitted = st.form_submit_button("Generate forecast", disabled=not enabled, type="primary")
    if submitted:
        try:
            with st.spinner("Checking saved inputs and generating the forecast…"):
                run = services.generate(selected, snapshot, model, artifact)
            st.session_state["visible_forecast"] = run
            st.session_state["visible_forecast_controls"] = (selected, model, artifact, snapshot.snapshot_id)
        except (ImportError, OSError, ValueError, RuntimeError, sqlite3.Error):
            st.error("Forecast unavailable. Saved input checks or the local service failed; no replacement result was created.")
    if st.session_state.get("visible_forecast"):
        current = (selected, model, artifact, snapshot.snapshot_id if snapshot else None)
        if current != st.session_state.get("visible_forecast_controls"):
            st.warning("Previous saved run: the current controls differ from its original request.")
        render_run(st.session_state["visible_forecast"], "prediction")
    else:
        st.info("No forecast selected. Unsupported pace, uncertainty and accuracy stay unavailable.")
    with st.expander("Model access & input groups"):
        model_table()
        st.write('Ridge, Random Forest, Gradient Boosting and XGBoost use qualifying position/gap, clean practice pace/variation/lap count, available tyre age/compound, and prior driver/team/circuit form. TabFM estimates finishing position from these same tabular fields using earlier labelled races as context. The weekend heuristic also uses sprint finishing order when a sprint exists.')
        st.write("Weather requires an issued pre-race forecast; pit loss and penalties require sourced estimates known before cutoff. Actual Grand Prix weather, laps, pit stops and results are never prediction inputs. Missing groups remain missing; we do not invent tyre temperatures or sprint sessions.")


def pending_model_panel(snapshot=None,prefix='validation'):
    if snapshot is None:
        snapshot=services.pending_snapshot_for(context())
    if snapshot is None or not snapshot.snapshot_id.startswith('pending-'):
        return
    with st.expander('Forecast this upcoming Grand Prix with all models',expanded=False):
        st.caption(f"Frozen before the Grand Prix at {snapshot.as_of_utc:%d %b %Y %H:%M} UTC. These ranks are estimates; no actual result has been used.")
        if st.button('Forecast with all active models',key=f'{prefix}_all_models'):
            try:
                st.session_state[f'{prefix}_model_comparison']=(snapshot.snapshot_id,services.pending_models(snapshot))
            except (OSError,ValueError,sqlite3.Error):
                st.warning('The prepared forecast comparison could not be read.')
        saved=st.session_state.get(f'{prefix}_model_comparison')
        if saved and saved[0]==snapshot.snapshot_id:
            results=saved[1]
            labels=services.driver_labels(snapshot.event.event_id,snapshot)
            ready={m:r for m,r in results.items() if r is not None}
            st.dataframe([{'Model':services.model_label(m),'Predicted winner':labels.get(r['ranking'][0]['driver_id'],r['ranking'][0]['driver_id']),
                'Drivers':len(r['ranking']),'Result comparison':'Awaiting actual Grand Prix result'} for m,r in ready.items()],hide_index=True,width='stretch')
            order=next(iter(ready.values()))['ranking'] if ready else []
            values={m:{r['driver_id']:r['predicted_rank'] for r in run['ranking']} for m,run in ready.items()}
            if order:
                st.dataframe([dict(Driver=labels.get(r['driver_id'],r['driver_id']),**{services.model_label(m):ranks.get(r['driver_id']) for m,ranks in values.items()}) for r in order],hide_index=True,width='stretch')
            for model,result in results.items():
                if result is None:st.caption(f'{services.model_label(model)}: no eligible prepared run.')
            st.caption('Matching saved inputs reuse immutable numeric forecasts. The model weights are not reloaded on a page rerun. Actual results can be appended later for a separate forecast-performance comparison.')


def compare_drivers():
    heading("Compare Drivers", "Compare lap times, sectors and car telemetry from this session.")
    if context().get('archive_session_id'):
        from streamlit_app.archive_views import compare
        return compare(context())
    snapshot = choose_snapshot("comparison_snapshot")
    tabs = st.tabs(["Performance", "Driver profiles", "Team profiles"])
    with tabs[0]:
        entries = snapshot.payloads.get("entries", []) if snapshot else []
        ids = [item["driver_id"] for item in entries if item.get("driver_id")]
        selected = st.multiselect("Drivers (up to four)", ids, default=ids[:2], max_selections=4, key="selected_drivers")
        records = [item for item in snapshot.payloads.get("driver_sessions", [])
                   if item.driver_id in selected and item.session_id == context()["session"]
                   and item.available_at_utc <= snapshot.as_of_utc and
                   (item.source.available_by(snapshot.as_of_utc) or snapshot.payloads.get('historical_reconstruction'))] if snapshot else []
        if records:
            st.dataframe([dict(Driver=item.driver_id, Session=item.session_id, **item.values) for item in records],
                         hide_index=True, width="stretch")
            st.caption("Saved same-session summaries; no cross-circuit lap comparison. An ideal lap is derived, never driven.")
        else:
            from streamlit_app.components.replay_ui import _sources
            historical, _ = _sources(context())
            if historical:
                from statistics import mean
                source = historical[0]
                summaries=[]
                for driver in selected:
                    laps=[lap for lap in source.timed_laps if lap['driver_id']==driver]
                    if laps:
                        summaries.append({'Driver':driver,'Timed laps':len(laps),
                                          'Fastest recorded lap (s)':min(lap['lap_time_s'] for lap in laps),
                                          'Mean recorded lap (s)':mean(lap['lap_time_s'] for lap in laps)})
                if summaries:
                    st.dataframe(summaries, hide_index=True, width='stretch')
                    st.caption('Historical session summary, including pit/flag laps. This full-session hindsight view is not used as a pre-race forecast input.')
                else:
                    st.info('No timed laps recorded for these drivers in this session.')
            else:
                st.info("Comparable session evidence is unavailable. Telemetry and sector values are not inferred.")
    with tabs[1]:
        profile_records(snapshot, "driver_profiles")
    with tabs[2]:
        profile_records(snapshot, "team_profiles")
    if snapshot:
        source_details(snapshot)


def profile_records(snapshot, key):
    profiles = snapshot.payloads.get(key) if snapshot else None
    if not profiles and snapshot and snapshot.payloads.get('history'):
        from data.profiles import build_profiles
        try:
            history=[row for row in snapshot.payloads['history'] if row.get('source_ids') or row.get('source')]
            bundle=build_profiles(history,snapshot.as_of_utc)
            profiles=[dict(row,available_at_utc=row['freshness_utc']) for row in
                      bundle['drivers' if key=='driver_profiles' else 'teams']]
        except (ValueError,TypeError):
            profiles=[]
    if profiles:
        try:
            profiles = [row for row in profiles if isinstance(row, dict) and row.get("source_ids")
                        and datetime.fromisoformat(row["available_at_utc"].replace("Z", "+00:00")) <= snapshot.as_of_utc]
        except (KeyError, ValueError, TypeError):
            profiles = []
    if profiles:
        st.dataframe(profiles, hide_index=True, width="stretch")
        st.caption("Profiles belong to this saved snapshot and cutoff. Samples and sources must accompany quantitative claims.")
    else:
        st.info("No sourced, versioned profiles are saved at this cutoff. Dated affiliations, form, reliability and pit-operation samples remain unavailable.")


def race_replay():
    heading("Race Replay", "Play recorded track positions and inspect timing, tyres and telemetry.")
    st.radio("Analysis mode", ["Causal prefix", "Hindsight"], horizontal=True, key="replay_mode")
    st.checkbox("Reduced motion / static view", value=not bool(context().get('archive_session_id')), key="reduced_motion")
    try:
        from streamlit_app.components.replay_ui import render_replay
    except ImportError:
        st.info("Replay data/component is not ready. Track positions, timing and playback are unavailable until a versioned frame source is loaded.")
    else:
        render_replay(context())
    st.caption("Live feed: not enabled. Missing coordinates are never invented.")


def strategy_lab():
    heading("Strategy Lab", "Explore recorded tyre stints, pit visits and lap pace.")
    if context().get('archive_session_id'):
        from streamlit_app.archive_views import strategy
        return strategy(context())
    from streamlit_app.components.strategy_ui import render_strategy
    render_strategy(context())


def weather_context():
    heading('Weather & Context', 'Recorded session conditions, track status and race control.')
    if context().get('archive_session_id'):
        from streamlit_app.archive_views import weather
        return weather(context())
    st.info('Select a downloaded archive session to inspect its weather and control messages.')


def model_validation():
    heading("Model Validation", "Evaluate race forecasts and follow their performance as official results arrive.")
    pending_model_panel()
    tabs = st.tabs(["Evaluation", "Performance Tracking", "Historical Race Studies"])
    queries = tracking()
    race_names = services.event_labels()
    metric_labels = {"rank_spearman": "Race order correlation", "position_mae": "Mean position error",
                     "ndcg_at_3": "Podium ranking quality", "podium_recall": "Podium recall",
                     "top10_overlap": "Top 10 overlap", "winner_accuracy": "Winner accuracy",
                     "winner_top3_hit": "Winner in predicted top 3", "winner_log_loss": "Winner log loss",
                     "winner_brier": "Winner Brier score", "pace_mae_seconds": "Mean pace error (s)",
                     "pace_rmse_seconds": "Pace RMSE (s)"}

    def metric_rows(models):
        rows = []
        for model, evaluation in models.items():
            for metric, value in evaluation.get("metrics", {}).items():
                if value is None and not st.session_state.get('evaluation_show_unavailable',False):
                    continue
                interval = evaluation.get("confidence_intervals", {}).get(metric)
                rows.append({"Model": services.model_label(model),
                             "Metric": metric_labels.get(metric, metric.replace("_", " ").title()),
                             "Result": f"{value:.3f}" if isinstance(value, (int, float)) else "Unavailable",
                             "Scored races": evaluation.get("metric_event_counts", {}).get(metric, 0),
                             "95% interval": f"{interval[0]:.3f} to {interval[1]:.3f}" if interval else "Unavailable"})
        return rows

    with tabs[0]:
        summaries = services.evaluations()
        if summaries:
            if summaries[0][0]=='Current active model comparison' and len(summaries)>1:
                if not st.checkbox('Include earlier research evaluations',key='evaluation_include_legacy'):
                    summaries=summaries[:1]
            labels = {name: ('Historical reconstruction' if report.get('status')=='evaluated_reconstruction' else
                             'Historical coverage audit' if report.get('status')=='insufficient_data' else 'Development evaluation')
                      + f" · {len(report.get('cohort_event_ids', []))} evaluated races" for name, report in summaries}
            chosen = st.selectbox("Evaluation report", [name for name, _ in summaries], key="evaluation_report",
                                  format_func=labels.get)
            summary = dict(summaries)[chosen]
            subsets=summary.get('benchmark_subsets',{})
            if subsets:
                size=st.selectbox('Benchmark size',[5,10,50,100],index=3,
                    format_func=lambda n:f'{n} races',key='evaluation_size')
                selected_subset=subsets[str(size)]
                ids=set(selected_subset['cohort_event_ids'])
                summary=dict(summary,**selected_subset,
                    folds=[f for f in summary.get('folds',[]) if set(f.get('test_event_ids',[]))&ids])
                st.caption('Nested chronological benchmarks: the latest 5 races are inside the latest 10, 50 and 100. Every model uses the same races; training always uses earlier outcomes.')
            models = summary.get("models", {})
            selected_model = st.selectbox("Evaluation model", [None, *models],
                format_func=lambda value: services.MODELS.get(value, value) if value else "All models", key="evaluation_model")
            st.caption('Qualifying baseline: predict the qualifying order as the final finishing order. It provides a strong, simple reference; it does not retrospectively apply race-grid penalties.')
            st.checkbox('Show metrics this model cannot yet provide',key='evaluation_show_unavailable')
            if summary.get("status") == "insufficient_data":
                st.warning("Not enough verified historical races to measure forecast accuracy. Imported laps are coverage, not an accuracy score.")
            elif summary.get('status') == 'evaluated_reconstruction':
                st.info('Research comparison from archived weekends. Original publication times were not recorded; live forecast accuracy must be measured separately on newly saved forecasts.')
            else:
                st.info("Development evaluation only. The final holdout remains locked; no production promotion is claimed.")
            coverage = summary.get("coverage", {})
            cards = [("Evaluated races", len(summary.get("cohort_event_ids", [])))]
            if summary.get("folds"):
                cards.append(("Walk-forward folds", len(summary["folds"])))
            if coverage:
                cards.extend([("Loaded sessions", coverage.get("sessions_loaded", "Unavailable")),
                              ("Observed laps", coverage.get("total_laps", "Unavailable"))])
            columns = st.columns(len(cards))
            for column, (label, value) in zip(columns, cards):
                column.metric(label, value)
            selected = {key: value for key, value in models.items() if selected_model is None or key == selected_model}
            rows = metric_rows(selected)
            if rows:
                st.subheader("Forecast quality")
                st.dataframe(rows, hide_index=True, width="stretch")
            else:
                st.info("Accuracy metrics will appear after eligible races have been evaluated. Unavailable scores remain empty.")
            reasons = summary.get("readiness", {}).get("reasons", [])
            unavailable = summary.get("unavailable_candidates", {})
            with st.expander("Data readiness & evaluation protocol"):
                protocol = summary.get("protocol", {})
                st.write(f"Protocol: {protocol.get('version') or summary.get('protocol_version') or summary.get('version','Legacy research protocol')} · Final holdout: {summary.get('final_holdout', 'Separate future-race evaluation')}")
                st.write('The protocol fixes which races are tested, how earlier races train each model, how missing/retired entrants are handled, and which metrics are calculated. The 2026 holdout is kept out of this historical benchmark; saved pending forecasts can be compared separately when actual results arrive.')
                st.write('These limitations do not mean the downloaded data is corrupt. They make the research claims precise and support a reproducible résumé project; they do not establish live winner-prediction accuracy.')
                for reason in reasons:
                    st.write("• " + str(reason).replace("_", " ").capitalize())
                for model, reason in unavailable.items():
                    st.write(f"{services.MODELS.get(model, model)}: {reason}")
                if coverage.get("unavailable_features"):
                    st.write("Missing evidence: " + ", ".join(str(item).replace("_", " ") for item in coverage["unavailable_features"]))
                st.download_button("Download evaluation audit", services.json_export(summary), file_name=chosen,
                                   mime="application/json", key="evaluation_audit")
        else:
            st.info("No evaluation report is saved yet. Forecast quality appears here once historical races have been evaluated.")
        with st.expander("How to read the metrics"):
            st.write("Race order correlation compares the whole finishing order (higher is better). Mean position error measures places missed (lower is better). Podium and Top 10 metrics measure the most relevant finishing positions.")
            st.write("Winner accuracy measures first-place picks. Brier score and log loss require a coherent, calibrated winner probability distribution; they remain unavailable for ranking-only models. Pace errors use seconds.")
            st.caption("Each metric shows its own race count. Intervals resample whole races; training error is never presented as forecast accuracy.")
    with tabs[1]:
        evidence_kind = "forecast"
        if queries:
            try:
                all_runs = queries.forecasts()
                if all_runs:
                    historical_only = all(row.get("provenance", {}).get("validation_status") == "historical_reconstruction" for row in all_runs)
                    evidence_kind = st.selectbox("Performance evidence", ["forecast", "historical_reconstruction"],
                        index=1 if historical_only else 0, key="tracking_evidence_kind",
                        format_func=lambda value: "Historical reconstructions" if value == "historical_reconstruction" else "Saved pre-race forecasts")
                ledger = [row for row in all_runs if
                          (row.get("provenance", {}).get("validation_status") == "historical_reconstruction") ==
                          (evidence_kind == "historical_reconstruction")]
                if evidence_kind == "historical_reconstruction":
                    st.warning("Historical reconstruction uses hindsight downloads of earlier-weekend sessions with assumed availability. These scores describe retrospective research, not verified pre-race prediction accuracy.")
                if ledger:
                    model_ids = sorted({row["model_id"] for row in ledger})
                    model_id = st.selectbox("Tracked model", [None, *model_ids],
                        format_func=lambda value: services.model_label(value) if value else "All models", key="tracked_model")
                    outcome_filter = st.selectbox("Outcome status", ["All outcomes", "Scored", "Awaiting outcome", "Incomplete matching field"], key="tracked_outcome")
                    summary = queries.summary(model_id=model_id, evidence_kind=evidence_kind)
                    groups = summary.get("groups", [])
                    scored = sum(group.get("n_scored_events", 0) for group in groups)
                    pending = sum(group.get("n_pending_events", 0) for group in groups)
                    left, right = st.columns(2)
                    left.metric("Scored model / race pairs", scored)
                    right.metric("Unscored model / race pairs", pending)
                    quality = metric_rows({group["model_id"] + " · " + group["mode"].replace("_", " "): group for group in groups})
                    if quality and outcome_filter != "Awaiting outcome":
                        st.subheader("Performance to date")
                        st.dataframe(quality, hide_index=True, width="stretch")
                    selected_run_ids = {run_id for group in groups for run_id in group.get("selected_run_ids", [])}
                    statuses = {(group["model_id"], group["mode"], race["event_id"]): race.get("outcome_status")
                                for group in groups for race in group.get("per_race", [])}
                    incompatible = {(row['model_id'], row['mode'], row['event_id']) for row in ledger
                        if row['run_id'] in selected_run_ids and queries.outcome_state(row['run_id']) == 'roster_mismatch'}
                    history = []
                    for group in groups:
                        for race in group.get("per_race", []):
                            state = ('Incomplete matching field' if (group['model_id'], group['mode'], race['event_id']) in incompatible else
                                     "Scored" if race.get("outcome_status") == "completed" else "Awaiting outcome")
                            if outcome_filter != "All outcomes" and state != outcome_filter:
                                continue
                            history.append({"Race": race_names.get(race['event_id'], race['event_id']), "Model": services.model_label(group['model_id']),
                                            "Forecast context": group["mode"].replace("_", " ").title(), "Outcome": state,
                                            "Position error": race.get("metrics", {}).get("position_mae"),
                                            "Order correlation": race.get("metrics", {}).get("rank_spearman")})
                    if history:
                        st.subheader("Race-by-race history")
                        st.dataframe(history, hide_index=True, width="stretch")
                    else:
                        st.info("No races match this outcome filter yet.")
                    with st.expander("Saved forecast runs"):
                        rows = []
                        for row in ledger:
                            if model_id is not None and row["model_id"] != model_id:
                                continue
                            # The summary scores only its selected latest cutoff; older saved runs do not inherit that score.
                            state = ('Incomplete matching field' if (row['model_id'], row['mode'], row['event_id']) in incompatible else "Scored" if statuses.get((row["model_id"], row["mode"], row["event_id"])) == "completed" else "Awaiting outcome") if row["run_id"] in selected_run_ids else "Earlier cutoff · outside current summary"
                            if outcome_filter != "All outcomes" and state != outcome_filter:
                                continue
                            rows.append({"Run": row["run_id"], "Race": race_names.get(row['event_id'], row['event_id']), "Model": services.model_label(row['model_id']),
                                         "Forecast cutoff (UTC)": row["as_of_utc"], "Outcome": state})
                        if rows:
                            st.dataframe(rows, hide_index=True, width="stretch")
                        else:
                            st.info("No saved runs match these filters.")
                else:
                    st.info("No saved pre-race forecasts in this cohort yet. Outcomes are pending, never scored as zero." if evidence_kind == "forecast" else
                            "No historical reconstruction runs in this cohort yet. Missing metrics remain unavailable.")
            except (ValueError, OSError, TypeError, KeyError, sqlite3.Error):
                st.warning("Tracking records could not be read. Missing metrics remain unavailable.")
        else:
            st.info("No saved tracking store. Forecasts awaiting official outcomes are pending, never scored as zero.")
        st.caption("Performance uses the latest saved cutoff per race, model and forecast context within the selected evidence cohort. Historical reconstruction and pre-race forecast scores are never combined. Different model cohorts are not a fair head-to-head comparison.")
        st.caption('A known result with a different recorded field is labelled incomplete and receives no score; it is not a future race awaiting a result.')
        if queries:
            health = queries.request_health()
            if health['requests']:
                st.subheader('Request health')
                columns = st.columns(3)
                columns[0].metric('Recorded requests', health['requests'])
                columns[1].metric('Failed requests', health['failures'])
                columns[2].metric('95th percentile runtime', f"{health['p95_latency_ms']:.0f} ms" if health['p95_latency_ms'] is not None else 'Unavailable')
                st.caption('Runtime and request failures describe the service, separately from prediction quality. Only your own request aggregates are shown.')


    with tabs[2]:
        historical_studies()


def historical_studies():
    studies = services.model_studies()
    if not studies:
        st.info('Historical race comparisons appear after the offline evaluation completes.')
        return
    chosen = st.selectbox('Historical study', [name for name,_ in studies],
        format_func=lambda name: f"{len(dict(studies)[name]['models'])} models · {len(dict(studies)[name]['cohort_event_ids'])} development races · {name[:8]}",
        key='historical_study')
    report = dict(studies)[chosen]
    st.caption('Pre-Grand-Prix inputs compared with independently joined race results. Historical publication times are reconstructed; 2026 stays locked.')
    cohorts={(r['n_events'],tuple(sorted(p['event_id'] for p in r.get('per_race',[])))) for r in report['models'].values()}
    if len(cohorts)>1:
        st.warning('Model coverage differs. Overall scores below are not a direct head-to-head comparison; use the matched qualifying comparisons for each challenger.')
    rows=[]
    for candidate,result in report['models'].items():
        m=result['metrics']
        rows.append({'Model':services.model_label(candidate),'Races':result['n_events'],
            'Order correlation':m.get('rank_spearman'),'Mean position error':m.get('position_mae'),
            'Winner accuracy':m.get('winner_accuracy')})
    st.dataframe(rows,hide_index=True,width='stretch')
    paired=[]
    baseline={p['event_id']:p['metrics'].get('rank_spearman') for p in report['models'].get('qualifying',{}).get('per_race',[])}
    for candidate,metrics in report.get('paired_qualifying_differences',{}).items():
        m=metrics.get('rank_spearman',{})
        values=[baseline[e] for e in m.get('event_ids',[]) if baseline.get(e) is not None]
        if values:
            paired.append({'Model':services.model_label(candidate),'Matched races':m['n_events'],
                'Qualifying correlation on same races':sum(values)/len(values),
                'Model minus qualifying correlation':m['mean_delta']})
    if paired:
        st.dataframe(paired,hide_index=True,width='stretch')
    choices=list(report.get('races',{})) or [report['case_study']['event_id']]
    case_event=st.selectbox('Historical race',choices,index=choices.index(report['case_study']['event_id']),
        format_func=lambda event:f"{report.get('races',{}).get(event,report['case_study'])['event_name']} · {event.split(':')[0]} · round {event.split(':')[1]}",
        key='historical_case_event')
    case=services.study_race(chosen,case_event)
    st.subheader(f"{case['event_name']} · {case['event_id'].split(':')[0]}")
    from data.catalog import session_label
    st.caption('Available earlier sessions: '+', '.join(session_label(s) for s in case['input_sessions'])+' · UTC cutoff: '+case['cutoff_utc'])
    model=st.selectbox('Case-study model',list(case['models']),format_func=services.model_label,key='case_study_model')
    values=case['models'][model]
    if values:
        st.dataframe([{'Driver':r['driver_name'],'Predicted':r['predicted_rank'],
            'Actual':r['actual_position'],'Position error':r['position_error'],'Race status':r['status']}
            for r in sorted(values,key=lambda r:r['predicted_rank'])],hide_index=True,width='stretch')
        st.caption('Position error = predicted minus actual place. Positive means the driver finished better than predicted. Retirements remain actual outcomes, not missing telemetry.')
    else:
        st.info('This model has no eligible prediction for the case race.')
    with st.expander('Matched comparisons & model limitations'):
        for candidate,metrics in report.get('paired_qualifying_differences',{}).items():
            m=metrics.get('rank_spearman',{})
            interval=m.get('interval_95')
            if interval:
                st.write(f"{services.model_label(candidate)}: correlation change {m['mean_delta']:+.3f} versus qualifying over {m['n_events']} matched races; 95% interval {interval[0]:+.3f} to {interval[1]:+.3f}.")
        st.write('Learned v1 models use practice, qualifying and prior driver/team/circuit form. The weekend heuristic also uses sprint finishes. Actual race weather and pit losses are excluded.')
        st.write('TabFM uses the same prior-only tabular fields as the learned regressors. Foundation pretraining overlap is unknown. No active model supplies calibrated winner probabilities; TimesFM is retained only in historical evidence.')
    st.download_button('Download historical study',services.json_export(report),file_name='historical-study.json',mime='application/json',key='study_download')


def reports():
    heading("Reports", "Download saved conversations, session analysis and model forecasts.",require_context=False)
    saved_chat_reports()
    st.info('Download session analysis for the selected race/session, or choose an earlier saved forecast to inspect its original inputs, model version and rankings. Session exports contain classification, laps, stints, pit visits and weather; forecast exports contain the saved ranking and briefing.')
    if context().get('archive_session_id'):
        from streamlit_app import archive_services
        from flows.analysis_report import analysis_bundle
        try:
            selected_data = archive_services.tables(context())
            st.subheader('Selected session analysis')
            st.write('Export the recorded classification, laps, stints, pit visits and weather with their source version. Large telemetry streams stay in the data library.')
            st.download_button('Download session analysis', analysis_bundle(selected_data),
                file_name=context()['archive_session_id'].replace(':', '-') + '-analysis.zip', mime='application/zip', key='session_analysis_export')
        except (OSError, ValueError, sqlite3.Error):
            st.warning('This session analysis is temporarily unavailable. Refresh the selection and try again.')
    queries = tracking()
    if queries:
        try:
            rows = queries.forecasts()
            ids = [services.plain(row)["run_id"] for row in rows]
            races = services.event_labels()
            labels = {row['run_id']: f"{races.get(row['event_id'], row['event_id'])} · {services.model_label(row['model_id'])}" for row in rows}
            selected = st.selectbox("Saved forecast run", ids, index=0 if ids else None, format_func=labels.get)
            if selected:
                render_run(services.plain(queries.get_forecast(selected)), "report")
        except (ValueError, OSError, TypeError, KeyError, sqlite3.Error):
            st.warning("A saved run could not be opened. Other reports remain available.")
    else:
        st.info("No immutable forecast run is saved in the local tracking store.")
    with st.expander("Legacy Markdown reports"):
        st.caption('These are earlier project-generated Markdown previews, retained for reference. They are separate from your saved chats and current model runs. Their narrative predictions were not recorded with the model/input references needed to validate them.')
        st.warning('Historical narratives · do not treat their claims as measured accuracy or current forecasts.')
        names = {path.name: path for path in services.reports()}
        selected = st.selectbox("Legacy report", list(names), index=0 if names else None)
        if selected:
            try:
                text = names[selected].read_text(encoding="utf-8")
                st.markdown(text)
                st.download_button("Download legacy Markdown", text, file_name=selected, mime="text/markdown")
            except (OSError, UnicodeError):
                st.error("This legacy report cannot be read. Choose another report.")
        else:
            st.info("No legacy reports are available.")


def archive_profile_panel(selected):
    from datetime import timezone
    adapter = services.archive_context()
    if adapter is None:
        st.info('Season profiles appear when recorded race results are available.')
        return
    try:
        profiles = adapter.profiles(selected['season'], datetime.now(timezone.utc))
    except (ValueError, OSError, sqlite3.Error):
        st.info('Season profiles are temporarily unavailable.')
        return
    tabs = st.tabs(['Driver profiles', 'Team profiles'])
    team_names = {p['team_id']: p.get('display_name', p['team_id']) for p in profiles['teams']}
    races = services.event_labels()
    for tab, kind, label in zip(tabs, ('drivers', 'teams'), ('Driver', 'Team')):
        with tab:
            by_id = {p['entity_id']: p for p in profiles[kind]}
            chosen = st.selectbox(label, list(by_id), index=0 if by_id else None,
                format_func=lambda value, profiles=by_id: profiles.get(value, {}).get('display_name', value), key='archive_profile_' + kind)
            if chosen is None:
                st.info('No sourced results are available for this season.')
                continue
            profile = by_id[chosen]
            form = profile['recent_form']
            reliability = profile['reliability']
            columns = st.columns(3)
            mean = form.get('mean_position')
            columns[0].metric('Recent mean position', f'{mean:.1f}' if mean is not None else 'Unavailable')
            columns[1].metric('Recent races', form.get('event_count', 0))
            columns[2].metric('Recorded retirements', reliability.get('classified_dnf_count', 0))
            affiliation = profile.get('affiliation')
            if affiliation:
                st.caption('Last recorded team this season: ' + team_names.get(affiliation['team_id'], affiliation['team_id']))
            st.caption('Recent form covers up to five races. Recorded status counts do not establish retirement causes or current contracts.')
            st.dataframe([{'Race': races.get(row['event_id'], row['event_id']), 'Classified position': row['actual_position'],
                'Recorded status': row['status']} for row in profile['lineage'][-10:]], hide_index=True, width='stretch')
            st.caption(f"{len(profile.get('source_ids', []))} source records · {selected['season']} season only")
            for url in profile.get('source_urls', [])[:3]:
                st.link_button('Open official source', url)


def ask_f1():
    from datetime import timezone
    from uuid import uuid4
    from agents.chat import safe_markdown
    from agents.claude_provider import vetted_url
    from security.access import AccessDenied
    from security.limits import LimitRejected
    area=st.session_state.get('_page_heading_area')
    with area if area is not None else st.container():
        st.title('Primus AI')
        st.caption('Ask about F1, explore race evidence, or compare saved model forecasts. Web answers include their sources.')
    selected=context() if st.session_state.get('chat_bind_context',False) else {}
    bound=bool(st.session_state.get('chat_bind_context',False))
    valid=bool(selected.get('round_number') and selected.get('session'))
    snapshot=None;run_id=None;driver_id=None
    extra=st.session_state.get('_chat_context_area')
    with extra if extra is not None else st.expander('Additional context'):
        if bound and valid:
            with st.expander('Driver & team profiles'):
                archive_profile_panel(selected)
            snapshot=choose_snapshot('chat_snapshot')
            queries=tracking()
            event_id=f"{selected['season']}:{selected['round_number']}"
            runs=[r for r in queries.forecasts() if r['event_id']==event_id] if queries else []
            run_id=st.selectbox('Attach saved forecast',[None,*[r['run_id'] for r in runs]],
                format_func=lambda value:value or 'No forecast attached',key='chat_attached_run')
            entries=snapshot.payloads.get('entries',()) if snapshot else ()
            if not entries and selected.get('archive_session_id'):
                from streamlit_app import archive_services
                entries=archive_services.tables(selected)['entries'].to_dict('records')
            labels=services.driver_labels(event_id,snapshot)
            driver_id=st.selectbox('Driver focus',[None,*[r['driver_id'] for r in entries]],
                format_func=lambda value:labels.get(value,value) if value else 'All available drivers',key='chat_focus_driver')
            st.caption('Binding applies to this race/session. A conflicting question will ask you to refine it or remove binding.')
        elif bound:
            st.info('Finish selecting a race and session, or turn off race binding.')
        st.caption('F1 only · read-only tools · shared request and spending limits. Forecast numbers stay in local tools; web facts are linked to their sources.')
    scope=(bound,selected.get('season'),selected.get('round_number'),selected.get('session'),
           snapshot.snapshot_id if snapshot else None,run_id,driver_id)
    if st.session_state.get('chat_scope')!=scope or 'chat_service' not in st.session_state:
        st.session_state['chat_scope']=scope
        st.session_state['chat_service']=services.chat_service(snapshot)
    service=st.session_state['chat_service']
    provider=getattr(service,'provider',None)
    st.caption(('Claude Sonnet 5.5 · internal evidence and cited web search' if provider else
                'Local evidence mode · configure Claude locally to enable general F1 reasoning and cited web search'))
    if st.button('New conversation',key='chat_new_conversation'):
        st.session_state['chat_messages']=[]
        st.session_state.pop('chat_conversation_id',None)
    if not st.session_state.get('chat_messages'):
        st.subheader('What would you like to understand?')
        st.write('Try “Who won the 2025 US Grand Prix?”, “Explain tyre graining with sources”, or “Compare model forecasts for Mexico 2026”.')
    def render_message(record):
        with st.chat_message(record['role']):
            st.markdown(safe_markdown(record['text']))
            local=False
            for citation in record.get('citations',[]):
                if citation.get('kind')=='web':
                    url=vetted_url(citation.get('url'))
                    if url:
                        label=citation.get('title') or 'Web source'
                        category='Media reporting' if citation.get('source_type')=='media' else 'Official source'
                        st.link_button(f"[{citation.get('id','W')}] {category} · {label[:100]}",url)
                else:
                    local=True
            if local:st.caption('Sources: Internal data')
    for record in st.session_state.get('chat_messages',[]):render_message(record)
    message=st.chat_input('Ask Primus AI about F1…',max_chars=4000,disabled=bound and not valid)
    if message:
        try:
            kwargs={}
            if bound and valid:
                kwargs=dict(run_id=run_id,driver_id=driver_id,event_id=f"{selected['season']}:{selected['round_number']}",
                    session_id=selected.get('archive_session_id',selected['session']),snapshot_id=snapshot.snapshot_id if snapshot else None)
            history=[dict(role=r['role'],content=r['text'][:6000]) for r in st.session_state.get('chat_messages',[])[-8:]]
            with st.spinner('Checking F1 evidence and sources…'):
                reply=service.ask(services.local_identity(),message,request_id='chat-'+uuid4().hex,
                    as_of_utc=datetime.now(timezone.utc),conversation=history,**kwargs)
            new=[dict(role='user',text=message),dict(role='assistant',text=reply['answer'],citations=reply.get('citations',[]))]
            st.session_state['chat_messages']=(st.session_state.get('chat_messages',[])+new)[-16:]
            for record in new:render_message(record)
            st.session_state['chat_last_reply']=reply
            try:
                st.session_state['chat_conversation_id']=services.chat_history().append_turn(
                    services.local_identity(),st.session_state.get('chat_conversation_id'),message,reply)
            except (OSError,ValueError,sqlite3.Error,AccessDenied):
                st.warning('This answer is visible, but could not be saved to conversation history. Keep a copy before leaving this page.')
        except (AccessDenied,LimitRejected) as error:
            st.warning('Request paused by access, rate or spending controls. '+str(error).replace('_',' '))
        except (OSError,ValueError,RuntimeError,sqlite3.Error):
            st.warning('The answer could not be completed. Try a narrower F1 question; no unsupported answer was substituted.')


def model_table():
    from ml.foundation_models import foundation_registry
    foundation = foundation_registry()
    st.dataframe([{"Model": name, "Access": foundation[key]['reason'] if key in foundation else
                  ('Ready · research baseline' if key in {"qualifying", "grid", "rolling_form", "weekend_form"} else "Fitted artifact required"),
                   "Validation": "No production eligibility claimed"} for key, name in services.MODELS.items()],
                 hide_index=True, width="stretch")


def settings_data():
    from streamlit_app.components.tables import display_table
    heading("Settings & Data", "Browse downloaded sessions and check the selected session’s coverage.")
    st.caption('Use this page to see which seasons, races and sessions are available, inspect missing/partial data, and check model or AI integration readiness. It helps explain why an option or chart may be unavailable; it does not start downloads.')
    with st.expander('AI and integration readiness'):
        st.dataframe(services.dependency_readiness(),hide_index=True,width='stretch')
        st.caption('Only configuration presence is shown. API keys stay local and are never displayed.')
    from streamlit_app import archive_services
    from data.catalog import session_label
    try:
        sessions = archive_services.inventory()
    except (OSError, ValueError, sqlite3.Error):
        st.warning('The local archive is temporarily unavailable. Refresh after collection finishes writing.')
        return
    if sessions:
        st.subheader('Local data library')
        display_table([{'Season': row['season'], 'Race': row['event_name'],
                       'Race date': row['local_race_date'],
                       'Session': session_label(row['session_code']) if row['session_code'] != 'SQ' else row['full_name'] or session_label('SQ')}
                      for row in sessions], hide_index=True, width='stretch')
        st.caption('Selectors show downloaded sessions only. Missing sessions and future races are not offered. This page never starts downloads.')
        from streamlit_app.archive_views import load
        data = load(context())
        if data is not None:
            st.subheader('Selected session coverage')
            labels = {'car': 'Car telemetry', 'positions': 'Track positions', 'laps': 'Lap timing',
                      'results': 'Classification', 'weather': 'Track weather', 'entries': 'Driver entries',
                      'track_status': 'Track status', 'race_control': 'Race control',
                      'session_results': 'Classification', 'session_entries': 'Driver entries'}
            coverage = data['coverage'].to_dict('records')
            if context()['session'] in ('R', 'S'):
                timing = archive_services.timing(str(archive_services.root().resolve()),
                    context()['archive_session_id'], context()['archive_version_id'])
                availability = {'timing_stream': 'Recorded timing', 'lap_end_position': 'Lap-end checkpoints'}
                coverage = [row for row in coverage if row['dataset_kind'] != 'timing']
                coverage.append(dict(dataset_kind='timing', status=availability.get(timing.attrs.get('timing_basis'), 'Unavailable'), rows=len(timing)))
                labels['timing'] = 'Replay standings'
            display_table([{'Data': labels.get(row['dataset_kind'], row['dataset_kind'].replace('_', ' ').title()),
                           'Coverage': str(row['status']).replace('_', ' ').title(), 'Records': row['rows']}
                          for row in coverage], hide_index=True, width='stretch')
            st.caption('Partial coverage means some samples or metadata are missing. Charts omit identified parser-generated samples; gaps remain visible. Tyre temperatures and stationary pit-stop times are not supplied by these FastF1 streams.')
        return
    st.info('No published local sessions are available. Run a collection notebook, then refresh this page.')


def saved_chat_reports():
    st.subheader('Saved conversations')
    st.caption('Completed questions and answers are saved automatically on this computer. New conversation starts a separate record. Downloads retain the original context and sources; conversations from before chat saving was added may not be available.')
    try:
        principal=services.local_identity()
        history=services.chat_history()
        records=history.list(principal)
        by_id={r['chat_id']:r for r in records}
        selected=st.selectbox('Conversation',list(by_id),index=0 if by_id else None,
            format_func=lambda value:f"{by_id[value]['title']} · {by_id[value]['updated_at'][:16].replace('T',' ')} UTC",key='report_chat')
        if selected:
            text=history.markdown(principal,selected)
            st.download_button('Download conversation Markdown',text,file_name=f'primus-chat-{selected[:8]}.md',mime='text/markdown')
            with st.expander('Conversation preview'):
                st.markdown(text)
        else:
            st.info('No conversations saved yet. Ask Primus AI a question to start one.')
    except (OSError,ValueError,sqlite3.Error):
        st.warning('Saved conversations are temporarily unavailable.')


def about():
    st.title('About Primus AI')
    st.write('Explore Formula 1 weekends through recorded data, race replays, pre-race model comparisons and sourced answers. Primus AI is a personal data science project built to make that evidence easier to understand.')
    start, guide, credits = st.tabs(['Getting started', 'Page guide', 'Credits'])
    with start:
        st.subheader('Start with a question or a weekend')
        st.markdown('''1. Open **Primus AI** to ask an F1 question. Include a year and race, or bind a session under **Additional context**.
2. For telemetry pages, choose **Season → Event / round → Session**. Only available sessions are offered; select again after changing season or race.
3. Open a page from the sidebar to inspect results, compare drivers or replay the session. Charts and tables follow that selection.
4. In **Predictions**, use matching saved inputs and choose a model. **Model Validation** explains how those models performed; **Reports** holds exports and saved chats.

**Completed weekend:** choose the 2025 United States Grand Prix and Grand Prix Race to inspect recorded results and compare them with a saved pre-race forecast.

**Mexico 2026:** practice and qualifying are available, but the Grand Prix recording is not yet saved. Explore those sessions and use the prepared pre-race forecasts; there is no actual race result to replay or score yet.''')
        st.caption('Missing observations stay unknown. A prediction is an estimate; sources and data coverage help you judge it.')
    with guide:
        st.subheader('What each page does')
        pages = [
            ('Primus AI', 'Ask F1 questions and inspect the evidence behind an answer. Use Additional context to bind a race/session; remove binding if you change topics. Saved conversations can be downloaded from Reports.'),
            ('Race Overview', 'See the selected session’s classification, participants and lap summary. Start here to understand what data is available for the weekend.'),
            ('Predictions', 'Estimate the final Grand Prix order from earlier weekend inputs. Select a saved snapshot and model, restore its saved cutoff/mode if needed, then Generate forecast. Forecast-ready inputs open existing matching results.'),
            ('Compare Drivers', 'Select one or more drivers, lap/tyre filters and a telemetry lap. Compare lap trends, sectors, speed, throttle and braking. Clean laps exclude identified pit and flagged laps.'),
            ('Race Replay', 'Select drivers and play or seek through recorded positions. Weather and event markers follow the clock. Recorded race positions/gaps appear when available; lap-completion order is clearly labelled otherwise.'),
            ('Strategy Lab', 'Read observed tyre stints, pit visits and lap time against tyre age. Initial garage departures are not necessarily missing stops. Unknown entry/exit durations remain unknown.'),
            ('Weather & Context', 'Inspect measured session conditions and track messages. Independent historical weather and sourced incident context are distinguished from track measurements.'),
            ('Model Validation', 'Evaluation compares models on shared 5/10/50/100-race research cohorts. Performance Tracking follows saved forecasts and later outcomes. Historical Race Studies shows past predicted-versus-actual orders; pending races are not scored.'),
            ('Reports', 'Download selected-session analysis, saved forecast rankings/briefings and saved conversation Markdown. A saved forecast keeps its original race and cutoff even if the global selection differs.'),
            ('Settings & Data', 'Check which races/sessions exist and inspect coverage or integration readiness. This page explains missing data; it does not start downloads.'),
            ('About', 'Find this usage guide and the authors, data sources and libraries credited by the project.'),
        ]
        for name, description in pages:
            with st.expander(name):
                st.write(description)
    with credits:
        st.subheader('Credits and acknowledgements')
        st.markdown('''- [FastF1](https://docs.fastf1.dev/): timing, telemetry and session data.
- [Jolpica F1](https://github.com/jolpica/jolpica-f1): historical race results.
- [Open-Meteo](https://open-meteo.com/): independent historical weather data.
- [Streamlit](https://streamlit.io/), Plotly, pandas and the Python open-source ecosystem: UI and analysis.
- [Google Research TabFM](https://github.com/google-research/tabfm): tabular foundation model research.
- [XGBoost](https://xgboost.readthedocs.io/), scikit-learn and PyTorch: prediction tooling.
- Anthropic Claude: F1 reasoning and cited web search; saved numeric forecasts come from local model tools.
- [Tom Shaw (@IAmTomShaw)](https://github.com/IAmTomShaw): [F1 race replay](https://github.com/IAmTomShaw/f1-race-replay), for visualization inspiration.
- [Mariana Antaya (@mar-antaya)](https://github.com/mar-antaya): [2025 F1 predictions](https://github.com/mar-antaya/2025_f1_predictions) and [2026 F1 predictions](https://github.com/mar-antaya/2026_f1_predictions), for weekend prediction research.

An independent project, not an official Formula 1 service. Data and software remain attributed to their respective sources.''')


ROUTES = [("Home", "Primus AI", "ask", ask_f1),
          ("Weekend", "Race Overview", "overview", overview),
          ("Weekend", "Predictions", "predictions", predictions),
          ("Weekend", "Compare Drivers", "compare", compare_drivers),
          ("Explore", "Race Replay", "replay", race_replay),
          ("Explore", "Strategy Lab", "strategy", strategy_lab),
          ("Explore", "Weather & Context", "weather", weather_context),
          ("Review", "Model Validation", "validation", model_validation),
          ("Review", "Reports", "reports", reports),
          ("Preferences", "About", "about", about),
          ("Preferences", "Settings & Data", "settings", settings_data)]
