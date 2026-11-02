"""One shared selection context, distinct from immutable saved-run context."""
from datetime import datetime, timezone
import sqlite3

import streamlit as st


def _reset_analysis():
    for key in ('selected_drivers', 'selected_driver', 'active_run_id', 'replay_commit',
                'pilot_compare', 'pilot_replay', 'pilot_telemetry_driver', 'pilot_telemetry_lap',
                'pilot_telemetry_loaded', 'replay_native_driver', 'replay_native_lap', 'replay_native_seconds'):
        st.session_state.pop(key, None)


def _race_changed():
    st.session_state['session'] = None
    _reset_analysis()


def _season_changed():
    st.session_state['round_number'] = None
    _race_changed()


def render_context():
    from streamlit_app import services
    from data.catalog import event_label, session_label
    st.session_state.setdefault("cutoff_initial", datetime.now(timezone.utc).replace(microsecond=0))
    from streamlit_app import archive_services
    try:
        archived = archive_services.inventory()
    except (OSError, ValueError, sqlite3.Error):
        st.warning('The local archive could not be read. Check its database before analysis.')
        archived = []
    if archived:
        return _archive_context(archived)
    calendar = services.calendar()
    inventory = services.session_inventory()
    saved, _ = services.snapshots()
    if 'season' not in st.session_state:
        preferred = max(saved, key=lambda item: item.as_of_utc) if saved else None
        st.session_state['season'] = preferred.event.season if preferred else (inventory[0]['season'] if inventory else 2026)
        st.session_state['round_number'] = preferred.event.round_number if preferred else (inventory[0]['round_number'] if inventory else 1)
    seasons = sorted({item['season'] for item in calendar} | {item.event.season for item in saved} |
                     {st.session_state['season']}, reverse=True) if calendar else list(range(2026, 2017, -1))
    columns = st.columns(3)
    season = columns[0].selectbox("Season", seasons, key="season", on_change=_season_changed)
    events = {row['round_number']: row for row in calendar if row['season'] == season}
    rounds = sorted(events) if events else list(range(1, 25))
    if 'round_number' not in st.session_state:
        st.session_state['round_number'] = rounds[0]
    elif st.session_state['round_number'] is not None and st.session_state['round_number'] not in rounds:
        st.session_state['round_number'] = None
    round_number = columns[1].selectbox("Event / round", rounds, key="round_number",
        index=None, placeholder='Select a race', on_change=_race_changed,
        format_func=lambda value: event_label(events[value]) if value in events else f'Round {value} · schedule not loaded')
    event = events.get(round_number, {})
    session_ids = [row['session_id'] for row in event.get('sessions', [])]
    options = [value for value in ['R', 'Q', 'FP1', 'FP2', 'FP3', 'SQ', 'S'] if not session_ids or value in session_ids] if round_number is not None else []
    if 'session' not in st.session_state:
        st.session_state['session'] = options[0]
    elif st.session_state['session'] is not None and st.session_state['session'] not in options:
        st.session_state['session'] = None
    session = columns[2].selectbox("Session", options, key="session", format_func=session_label,
                                 index=None, placeholder='Select a session', disabled=round_number is None)
    identity = (season, round_number, session)
    if st.session_state.get("context_identity") != identity:
        for key in ("selected_drivers", "selected_driver", "active_run_id", "replay_commit"):
            st.session_state.pop(key, None)
        st.session_state["context_identity"] = identity
    return {"season": season, "round_number": round_number, "session": session,
            'event_name': event.get('event_name'), 'race_date': event.get('race_date')}


def _archive_context(rows):
    """Offer downloaded, published sessions only, with provider names and dates."""
    from data.catalog import session_label
    columns = st.columns(3)
    seasons = sorted({row['season'] for row in rows}, reverse=True)
    if st.session_state.get('season') not in seasons:
        st.session_state['season'] = seasons[0]
    season = columns[0].selectbox('Season', seasons, key='season', on_change=_season_changed)
    events = {row['round_number']: row for row in rows if row['season'] == season}
    if 'round_number' not in st.session_state:
        st.session_state['round_number'] = min(events)
    elif st.session_state['round_number'] is not None and st.session_state['round_number'] not in events:
        st.session_state['round_number'] = None
    number = columns[1].selectbox('Event / round', sorted(events), key='round_number',
        index=None, placeholder='Select a race', on_change=_race_changed,
        format_func=lambda value: f"{events[value]['event_name']} · {events[value]['local_race_date']}")
    sessions = {row['session_code']: row for row in rows
                if row['season'] == season and row['round_number'] == number}
    options = [code for code in ('R', 'Q', 'SQ', 'S', 'FP1', 'FP2', 'FP3') if code in sessions]
    options.extend(code for code in sessions if code not in options)
    if 'session' not in st.session_state:
        st.session_state['session'] = options[0] if options else None
    elif st.session_state['session'] is not None and st.session_state['session'] not in sessions:
        st.session_state['session'] = None
    def label(value):
        # Preserve historical Sprint Shootout names while expanding practice codes.
        return session_label(value) if value in {'FP1', 'FP2', 'FP3', 'R'} else sessions[value]['full_name'] or session_label(value)
    code = columns[2].selectbox('Session', options, key='session', format_func=label,
                               index=None, placeholder='Select a session', disabled=number is None)
    if number is None or code is None:
        return dict(season=season, round_number=number, session=code, event_name=None, race_date=None)
    row = sessions[code]
    identity = (season, number, code, row['version_id'])
    if st.session_state.get('context_identity') != identity:
        for key in ('selected_drivers', 'selected_driver', 'active_run_id', 'replay_commit',
                    'pilot_compare', 'pilot_replay', 'pilot_telemetry_driver', 'pilot_telemetry_lap', 'pilot_telemetry_loaded',
                    'replay_native_driver', 'replay_native_lap', 'replay_native_seconds'):
            st.session_state.pop(key, None)
        st.session_state['context_identity'] = identity
    date = datetime.fromtimestamp(row['scheduled_start_utc_ns'] / 1e9, timezone.utc).strftime('%d %b %Y') if row['scheduled_start_utc_ns'] else 'Date unavailable'
    st.caption(f"{label(code)} · {date} UTC")
    return dict(season=season, round_number=number, session=code, event_name=row['event_name'],
                race_date=row['local_race_date'], archive_session_id=row['session_id'],
                archive_version_id=row['version_id'], session_name=label(code))


def forecast_context(context):
    saved=st.session_state.get('forecast_request_context',{})
    if saved:
        st.session_state.setdefault('forecast_mode',saved['mode'])
        st.session_state.setdefault('cutoff_date',saved['cutoff'].date())
        st.session_state.setdefault('cutoff_time',saved['cutoff'].time().replace(tzinfo=None))
    columns = st.columns(3)
    mode = columns[0].selectbox("Forecast mode", ["pre_weekend", "post_qualifying"], key="forecast_mode",
                                format_func=lambda value: value.replace("_", " ").title())
    initial = st.session_state["cutoff_initial"]
    st.session_state.setdefault('cutoff_date', initial.date())
    st.session_state.setdefault('cutoff_time', initial.time().replace(tzinfo=None))
    day = columns[1].date_input("Cutoff date (UTC)", key="cutoff_date")
    clock = columns[2].time_input("Cutoff time (UTC)", key="cutoff_time")
    result=dict(context, mode=mode, cutoff=datetime.combine(day, clock, tzinfo=timezone.utc))
    st.session_state['forecast_request_context']=dict(mode=result['mode'],cutoff=result['cutoff'])
    return result
