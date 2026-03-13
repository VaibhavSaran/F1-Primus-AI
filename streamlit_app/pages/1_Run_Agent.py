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
    progress = st.progress(0, text="Starting pipeline...")
    status   = st.empty()

    status.info("⏳ Fetching weather data...")
    progress.progress(10)

    import threading
    result_container = {}

    def run_agent():
        result_container["result"] = run_pre_race_agent(round_number=selected_round)

    thread = threading.Thread(target=run_agent)
    thread.start()

    import time
    steps = [
        (25,  "⏳ Loading FP2 practice data..."),
        (50,  "⏳ Searching news and penalties..."),
        (70,  "⏳ Running ML prediction model..."),
        (85,  "⏳ Claude is writing the report..."),
    ]
    step_interval = 30  # seconds between fake progress updates

    for i, (pct, msg) in enumerate(steps):
        thread.join(timeout=step_interval)
        if not thread.is_alive():
            break
        progress.progress(pct, text=msg)
        status.info(msg)

    thread.join()  # wait for full completion
    progress.progress(100, text="✅ Complete!")
    status.empty()

    result = result_container.get("result", {})

    if "error" in result:
        st.error(f"Pipeline failed: {result['error']}")
    else:
        st.success(f"Analysis complete — {result['message_count']} messages exchanged.")
        st.divider()
        st.markdown(result["report"])

        from pathlib import Path
        import json
        from datetime import datetime

        reports_dir = Path("data/reports")
        reports_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename  = f"Round_{selected_round}_{race_name.replace(' ', '_')}_{timestamp}"

        (reports_dir / f"{filename}.md").write_text(result["report"], encoding="utf-8")
        (reports_dir / f"{filename}_tools.json").write_text(
            json.dumps(result["tool_outputs"], indent=2, default=str),
            encoding="utf-8"
        )
        st.caption(f"Report saved to data/reports/{filename}.md")