"""
F1 Primus AI — Streamlit app entry point.
"""
import streamlit as st

st.set_page_config(
    page_title="F1 Primus AI",
    page_icon="🏎️",
    layout="wide",
)

st.title("🏎️ F1 Primus AI")
st.caption("Autonomous pre-race prediction pipeline — 2026 F1 Season")
st.divider()

st.markdown("""
**Navigate using the sidebar:**

- **Run Agent** — trigger the full pre-race analysis for any round
- **Model Performance** — view MLflow experiment metrics
- **Race History** — browse previously generated reports
""")

st.info("Select a page from the sidebar to get started.")