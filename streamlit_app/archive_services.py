"""Cached, read-only access to the notebook's published local sessions."""
import streamlit as st
from data.archive.reader import ArchiveReader
from streamlit_app import services


def root():
    return services.release_root() / 'f1_2025_2026'


@st.cache_data(show_spinner=False, max_entries=8)
def _inventory(directory, stamp):
    return ArchiveReader(directory).available_sessions()


def inventory():
    directory = root()
    db = directory / 'f1.sqlite3'
    if not db.is_file():
        return []
    stat = db.stat()
    return _inventory(str(directory.resolve()), (stat.st_mtime_ns, stat.st_size))


@st.cache_data(show_spinner=False, max_entries=8)
def _tables(directory, session_id, version_id):
    data = ArchiveReader(directory).tables(session_id)
    if data['meta']['version_id'] != version_id:
        raise ValueError('The published session changed; refresh the page.')
    return data


def tables(context):
    return _tables(str(root().resolve()), context['archive_session_id'], context['archive_version_id'])


@st.cache_data(show_spinner=False, max_entries=8)
def timing(directory, session_id, version_id):
    return ArchiveReader(directory).timing(session_id, version_id=version_id)


@st.cache_data(show_spinner=False, max_entries=8)
def dense(directory, session_id, version_id, kind, drivers):
    reader = ArchiveReader(directory)
    if reader.tables(session_id)['meta']['version_id'] != version_id:
        raise ValueError('The published session changed; refresh the page.')
    return reader.dense(session_id, kind, driver_ids=list(drivers))


@st.cache_data(show_spinner=False, max_entries=8)
def replay(directory, session_id, version_id, drivers):
    from data.archive.replay_adapter import build_replay
    reader = ArchiveReader(directory)
    if reader.tables(session_id)['meta']['version_id'] != version_id:
        raise ValueError('The published session changed; refresh the page.')
    return build_replay(reader, session_id, list(drivers))
