"""
Page 1 — Run the pre-race agent for a selected round.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import streamlit as st
from config import F1_CALENDAR
from agents.graph import run_pre_race_agent

st.title("Run Agent")
st.caption("Trigger the full pre-race intelligence pipeline for any 2026 round.")
st.divider()

# Round selector
round_options = {
    f"Round {r} — {info[0]}": r
    for r, info in F1_CALENDAR.items()
}

selected_label = st.selectbox("Select a race round", options=list(round_options.keys()))
selected_round = round_options[selected_label]

race_info = F1_CALENDAR[selected_round]
race_name, location, country, race_date = race_info

st.markdown(f"""
| | |
|---|---|
| **Circuit** | {location}, {country} |
| **Date** | {race_date} |
| **Round** | {selected_round} of 24 |
""")

st.divider()

if st.button("Run Analysis"):
    with st.spinner(f"Running pipeline for {race_name}... this takes 2-3 minutes."):
        result = run_pre_race_agent(round_number=selected_round)

    if "error" in result:
        st.error(f"Pipeline failed: {result['error']}")
    else:
        st.success(f"Analysis complete — {result['message_count']} messages exchanged.")
        st.divider()
        st.markdown(result["report"])

        # Save report to disk
        from pathlib import Path
        import json
        from datetime import datetime

        reports_dir = Path("data/reports")
        reports_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"Round_{selected_round}_{race_name.replace(' ', '_')}_{timestamp}"

        (reports_dir / f"{filename}.md").write_text(result["report"], encoding="utf-8")
        (reports_dir / f"{filename}_tools.json").write_text(
            json.dumps(result["tool_outputs"], indent=2, default=str),
            encoding="utf-8"
        )

        st.caption(f"Report saved to data/reports/{filename}.md")