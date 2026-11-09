"""PRIMUS F1: a native shell for the saved telemetry workspace."""
import os
import sys
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from streamlit_app.screens import ROUTES
from streamlit_app.components.context import render_context

st.set_page_config(page_title="PRIMUS F1", page_icon="🏁", layout="wide")
archive_preview = os.environ.get("F1_ACCESS_MODE") == "private_archive_preview"
if archive_preview and not os.environ.get("F1_SERVING_RELEASE_ROOT"):
    st.error("The archive preview requires a configured serving release.")
    st.stop()
# The archive preview is deployed with Cloud Run IAM and serves only archive views.
if not archive_preview and st.get_option("server.address") not in {"localhost", "127.0.0.1", "::1"}:
    st.error("This private workspace requires verified access before using a non-loopback server binding.")
    st.stop()
with st.sidebar:
    st.title("PRIMUS F1")
    st.caption("Race telemetry · models · grounded analysis")

pages = {}
for group, label, slug, render in ROUTES:
    if archive_preview:
        if slug in {"overview", "compare", "replay", "weather", "about"}:
            pages.setdefault(group, []).append(st.Page(
                render, title=label, url_path=slug, default=slug == "overview"
            ))
        continue
    if slug == 'ask':
        pages.setdefault('Home', []).append(st.Page(render,title=label,url_path='home',default=True))
        pages['Home'].append(st.Page(render,title='Primus AI direct link',url_path='ask',visibility='hidden'))
    else:
        pages.setdefault(group, []).append(st.Page(render, title=label, url_path=slug))
page = st.navigation(pages)
# Pages without selectors must not let widget cleanup discard the shared choice.
for key in ('season', 'round_number', 'session'):
    if key in st.session_state:
        st.session_state[key] = st.session_state[key]
# Keep shared selectors in the entrypoint so their values survive page navigation.
st.session_state['_page_heading_area'] = st.container()
if not archive_preview and page.url_path in {'','ask'}:
    with st.expander('Additional context',expanded=False):
        bound=st.checkbox('Bind this question to a race',value=False,key='chat_bind_context')
        if bound:
            st.session_state['workspace_context']=render_context()
        else:
            st.session_state['workspace_context']={}
            st.caption('Optional: bind a race/session when your question is specific to it. Without binding, include a race and year or ask a general F1 question.')
        st.session_state['_chat_context_area']=st.container()
elif page.url_path != 'about':
    st.session_state['workspace_context'] = render_context()
page.run()
