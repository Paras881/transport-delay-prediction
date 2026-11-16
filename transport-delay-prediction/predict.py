"""Predict delay for new trips with the saved model.

    python predict.py                       # demo: a clear vs a stormy, event-heavy trip
    python predict.py --csv new_trips.csv   # same columns as the training CSV (target not needed)
"""
import argparse
import joblib
import pandas as pd

from src.data_prep import load_raw, to_snake
from src.features import add_features

MODEL_PATH = "outputs/models/best_model.joblib"


def prepare(df, roles):
    df = df.copy()
    if "date" in roles and roles["date"] in df:
        df[roles["date"]] = pd.to_datetime(df[roles["date"]], errors="coerce")
    for r in ("time", "scheduled_arrival"):
        if r in roles and roles[r] in df:
            df[roles[r]] = pd.to_datetime(df[roles[r]].astype(str), format="mixed", errors="coerce")
    for r in ("transport_type", "route", "origin", "destination", "weather", "event_type", "weekday", "season"):
        if r in roles and roles[r] in df:
            df[roles[r]] = df[roles[r]].astype("string")
    return add_features(df, dict(roles))


def predict(df):
    bundle = joblib.load(MODEL_PATH)
    roles = bundle["roles"]
    feats = prepare(df, roles)
    cols = bundle["numeric_features"] + bundle["categorical_features"]
    for c in cols:
        if c not in feats:
            feats[c] = pd.NA
    return bundle["pipeline"].predict(feats[cols]), bundle["model_name"]


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv")
    a = ap.parse_args()
    if a.csv:
        df = load_raw(a.csv)
    else:
        base = dict(date="2024-11-15", time="17:30:00", transport_type="Bus", route_id="Route_7",
                    origin_station="Station_3", destination_station="Station_12",
                    scheduled_departure="17:30:00", scheduled_arrival="18:05:00",
                    temperature_C=14.0, humidity_percent=60, wind_speed_kmh=12, precipitation_mm=0.0,
                    traffic_congestion_index=55, holiday=0, peak_hour=1, weekday="Friday", season="Autumn",
                    weather_condition="Clear", event_type="None", event_attendance_est=0)
        storm = {**base, "weather_condition": "Storm", "precipitation_mm": 14.0, "wind_speed_kmh": 45,
                 "event_type": "Sports", "event_attendance_est": 45000, "traffic_congestion_index": 85}
        df = pd.DataFrame([base, storm])
        df.columns = [to_snake(c) for c in df.columns]
    preds, name = predict(df)
    print(f"Model: {name}")
    for i, p in enumerate(preds):
        print(f"  trip {i}: predicted delay = {p:.1f} min")
