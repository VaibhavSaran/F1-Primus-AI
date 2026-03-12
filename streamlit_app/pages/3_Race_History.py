"""
Page 3 — Browse previously generated race reports.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import streamlit as st

st.title("Race History")
st.caption("Browse previously generated pre-race analysis reports.")
st.divider()

reports_dir = Path("data/reports")

if not reports_dir.exists() or not any(reports_dir.glob("*.md")):
    st.info("No reports found yet. Run the agent on the Run Agent page first.")
else:
    report_files = sorted(reports_dir.glob("*.md"), reverse=True)

    report_labels = [f.stem for f in report_files]
    selected_label = st.selectbox("Select a report", options=report_labels)

    selected_file = reports_dir / f"{selected_label}.md"
    st.divider()

    content = selected_file.read_text(encoding="utf-8")
    st.markdown(content)

    st.divider()
    st.download_button(
        label="Download report",
        data=content,
        file_name=f"{selected_label}.md",
        mime="text/markdown",
    )