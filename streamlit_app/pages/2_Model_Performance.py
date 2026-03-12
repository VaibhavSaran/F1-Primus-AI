"""
Page 2 — MLflow experiment metrics viewer.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import streamlit as st

st.title("Model Performance")
st.caption("MLflow experiment tracking for the Gradient Boosting race predictor.")
st.divider()

try:
    import mlflow
    from config import MLFLOW_TRACKING_URI

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    client = mlflow.tracking.MlflowClient()

    experiments = client.search_experiments()
    if not experiments:
        st.warning("No MLflow experiments found. Run the agent at least once first.")
    else:
        exp = experiments[0]
        st.markdown(f"**Experiment:** {exp.name}")
        st.divider()

        runs = client.search_runs(
            experiment_ids=[exp.experiment_id],
            order_by=["start_time DESC"],
            max_results=20,
        )

        if not runs:
            st.warning("No runs found yet.")
        else:
            import pandas as pd

            rows = []
            for run in runs:
                rows.append({
                    "Run Name":  run.data.tags.get("mlflow.runName", run.info.run_id[:8]),
                    "MAE (s)":   round(run.data.metrics.get("mae", 0), 4),
                    "Drivers":   int(run.data.metrics.get("num_drivers", 0)),
                    "Status":    run.info.status,
                    "Run ID":    run.info.run_id[:8],
                })

            df = pd.DataFrame(rows)
            st.dataframe(df, width="stretch", hide_index=True)

            st.divider()
            st.caption(f"Showing {len(runs)} most recent runs · MLflow: {MLFLOW_TRACKING_URI}")

except Exception as e:
    st.error(f"Could not connect to MLflow: {e}")
    st.info("Make sure MLflow server is running: `mlflow server --port 5001 --workers 1`")