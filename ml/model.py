"""
Machine Learning model
Gradient Boosting Regressor for F1 race time prediction.
Tracks every experiment run with MLflow.
"""

from __future__ import annotations
import warnings
import numpy as np
import pandas as pd
import os
import tempfile
import fastf1
import mlflow
import mlflow.sklearn
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error
from config import (
    CACHE_DIR, CURRENT_SEASON, MLFLOW_TRACKING_URI,
    MLFLOW_EXPERIMENT, DRIVERS_2026,
)

warnings.filterwarnings("ignore", category=FutureWarning)
fastf1.Cache.enable_cache(CACHE_DIR)
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

def run_prediction(round_number: int, race_name: str) -> dict:
    """
    Train on historical data and predict race times.
    Logs parameters, metrics, and model artifact to MLflow.

    Args:
        round_number: Race round number on the 2026 calendar (1-24).
        race_name:    Human readable race name e.g. 'Australian GP'.

    Returns:
        dict with predicted podium, top 10, MAE, and MLflow run ID.
    """
    mlflow.set_experiment(MLFLOW_EXPERIMENT)

    run_name = f"Round_{round_number}_{race_name.replace(' ', '_')}"

    with mlflow.start_run(run_name=run_name) as run:

        # Load training data 
        print(f"Loading training data for Round {round_number}")
        train_df = _load_historical_data(round_number)

        if train_df.empty:
            return {"error": "Could not load sufficient historical training data."}

        # Feature engineering 
        le      = LabelEncoder()
        X_train = _build_features(train_df, le, fit=True)
        y_train = train_df["mean_lap_time"].values

        # Train model 
        params = {
            "n_estimators":  200,
            "learning_rate": 0.05,
            "max_depth":     4,
            "random_state":  42,
        }
        print(f"Training Gradient Boosting model")
        model = GradientBoostingRegressor(**params)
        model.fit(X_train, y_train)

        train_preds = model.predict(X_train)
        mae         = mean_absolute_error(y_train, train_preds)

        # Build 2026 prediction input 
        print(f"Generating predictions")
        pred_df = _build_2026_input(round_number)
        X_pred = _build_features(pred_df, le, fit=False)
        predicted_times = model.predict(X_pred)

        pred_df["predicted_time"] = predicted_times
        results = (
            pred_df[["driver", "predicted_time"]]
            .sort_values("predicted_time")
            .reset_index(drop=True)
        )
        results["position"]          = results.index + 1
        results["predicted_time_fmt"] = results["predicted_time"].apply(_fmt_time)
        results["gap_to_leader"]      = (
            results["predicted_time"] - results["predicted_time"].iloc[0]
        ).round(3)

        # Log to MLflow 
        print(f"Logging to MLflow")
        mlflow.log_params(params)
        mlflow.log_param("round_number", round_number)
        mlflow.log_param("race_name", race_name)
        mlflow.log_param("season", CURRENT_SEASON)
        mlflow.log_metric("train_mae_seconds", round(mae, 4))
        mlflow.sklearn.log_model(model, "gradient_boosting_model")

        # Save predictions CSV as artifact
        results_path = os.path.join(tempfile.gettempdir(), f"predictions_round_{round_number}.csv")
        results.to_csv(results_path, index=False)
        mlflow.log_artifact(results_path)

        # Build return payload 
        podium = (
            results.head(3)[["position", "driver", "predicted_time_fmt"]]
            .to_dict("records")
        )
        top_10 = (
            results.head(10)[["position", "driver", "predicted_time_fmt", "gap_to_leader"]]
            .to_dict("records")
        )

        print(f" MAE: {mae:.3f}s | MLflow Run: {run.info.run_id[:8]}")

        return {
            "round":        round_number,
            "race_name":    race_name,
            "podium":       podium,
            "top_10":       top_10,
            "model_mae_s":  round(mae, 3),
            "mlflow_run_id": run.info.run_id,
            "summary":      _summarise(race_name, podium, mae),
        }


# Data loaders 

def _load_historical_data(round_number: int) -> pd.DataFrame:
    """Load previous season race data as training data."""
    rows = []
    for attempt_round in [round_number, max(1, round_number - 1), round_number + 1]:
        try:
            sess = fastf1.get_session(2025, attempt_round, "R")
            sess.load(telemetry=False, weather=False, messages=False)
            laps = sess.laps.pick_quicklaps()
            if laps.empty:
                continue
            for driver in laps["Driver"].unique():
                drv_laps = laps[laps["Driver"] == driver]["LapTime"].dt.total_seconds()
                if len(drv_laps) >= 3:
                    rows.append({
                        "driver":        driver,
                        "mean_lap_time": drv_laps.mean(),
                        "best_lap_time": drv_laps.min(),
                        "std_lap_time":  drv_laps.std(),
                        "lap_count":     len(drv_laps),
                    })
            if rows:
                print(f"Training data loaded from 2025 Round {attempt_round}")
                break
        except Exception:
            continue
    return pd.DataFrame(rows)


def _build_2026_input(round_number: int) -> pd.DataFrame:
    """
    Build prediction input using 2026 FP2 data.
    For drivers missing from FP2, estimates based on
    the slowest FP2 time + a small penalty offset.
    """
    rows = []
    fp2_times = {}

    try:
        sess = fastf1.get_session(CURRENT_SEASON, round_number, "FP2")
        sess.load(telemetry=False, weather=False, messages=False)
        laps = sess.laps.pick_quicklaps()

        # Build from actual FP2 data first
        for driver in laps["Driver"].unique():
            drv_laps = laps[laps["Driver"] == driver]["LapTime"].dt.total_seconds()
            if len(drv_laps) >= 1:
                best = drv_laps.min()
                fp2_times[driver] = best
                rows.append({
                    "driver":        driver,
                    "mean_lap_time": drv_laps.mean(),
                    "best_lap_time": best,
                    "std_lap_time":  drv_laps.std() if len(drv_laps) > 1 else 0.5,
                    "lap_count":     len(drv_laps),
                })

    except Exception:
        print(f"FP2 data unavailable — using tier estimates for all drivers")

    # For drivers missing from FP2, use slowest FP2 time + offset
    if fp2_times:
        slowest_fp2  = max(fp2_times.values())
        fastest_fp2  = min(fp2_times.values())
        fp2_range    = slowest_fp2 - fastest_fp2

        # Tier offsets beyond slowest FP2 time
        missing_tiers = {
            # New team drivers (Cadillac) — expect ~1.5s off pace
            "BOT": slowest_fp2 + 1.5,
            "PER": slowest_fp2 + 1.5,
            # Drivers who didn't set a FP2 time
            "STR": slowest_fp2 + 0.8,
            "SAI": slowest_fp2 + 0.3,
        }

        for driver in DRIVERS_2026:
            if driver not in fp2_times:
                base = missing_tiers.get(driver, slowest_fp2 + 1.0)
                rows.append({
                    "driver":        driver,
                    "mean_lap_time": base,
                    "best_lap_time": base - 0.3,
                    "std_lap_time":  0.5,
                    "lap_count":     5,
                })
                print(f"{driver} missing from FP2 — using estimate: {base:.3f}s")
    else:
        # Full fallback if no FP2 data at all
        for driver in DRIVERS_2026:
            rows.append(_placeholder_row(driver))

    return pd.DataFrame(rows)


def _placeholder_row(driver: str) -> dict:
    """Fallback estimates when live session data isn't available."""
    tier_map = {
        # Top tier
        "NOR": 89.5, "VER": 89.6, "LEC": 89.7, "RUS": 89.8,
        "PIA": 89.9, "HAM": 90.0, "ANT": 90.1, "HAD": 90.4,
        # Midfield
        "ALO": 90.5, "SAI": 90.6, "GAS": 90.7, "COL": 90.9,
        "ALB": 91.0, "STR": 91.1, "LAW": 91.2, "LIN": 91.3,
        # Lower midfield
        "OCO": 91.2, "BEA": 91.3, "HUL": 91.1, "BOR": 91.4,
        # Cadillac (new team)
        "BOT": 91.6, "PER": 91.5,
    }
    base = tier_map.get(driver, 91.5)
    return {
        "driver":        driver,
        "mean_lap_time": base,
        "best_lap_time": base - 0.5,
        "std_lap_time":  0.4,
        "lap_count":     20,
    }


def _build_features(df: pd.DataFrame, le: LabelEncoder, fit: bool) -> np.ndarray:
    """Encode drivers and assemble feature matrix."""
    df = df.copy()
    if fit:
        df["driver_enc"] = le.fit_transform(df["driver"])
    else:
        known = set(le.classes_)
        df["driver"] = df["driver"].apply(
            lambda d: d if d in known else le.classes_[0]
        )
        df["driver_enc"] = le.transform(df["driver"])

    features = ["driver_enc", "best_lap_time", "std_lap_time", "lap_count"]
    for col in features:
        if col not in df.columns:
            df[col] = 0.0

    return df[features].fillna(0).values


# Formatters 

def _fmt_time(seconds: float) -> str:
    mins = int(seconds // 60)
    secs = seconds % 60
    return f"{mins}:{secs:06.3f}"


def _summarise(race: str, podium: list, mae: float) -> str:
    if not podium:
        return "Prediction failed."
    p1 = podium[0]
    p2 = podium[1] if len(podium) > 1 else {}
    p3 = podium[2] if len(podium) > 2 else {}
    return (
        f"Predicted {race} podium: "
        f"1. {p1.get('driver','?')} | "
        f"2. {p2.get('driver','?')} | "
        f"3. {p3.get('driver','?')} "
        f"(Model MAE: {mae:.2f}s)"
    )