"""Feature engineering: time features, weather severity flags, event features."""
import numpy as np
import pandas as pd

EXTREME_COLD_C = 0
EXTREME_HEAT_C = 32


def add_features(df: pd.DataFrame, roles: dict) -> pd.DataFrame:
    df = df.copy()

    # ---------- Time-based ----------
    if "date" in roles:
        d = df[roles["date"]]
        df["month"] = d.dt.month
        df["day_of_week"] = d.dt.dayofweek
        df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
    if "time" in roles:
        t = df[roles["time"]]
        df["hour"] = t.dt.hour
        mins = t.dt.hour * 60 + t.dt.minute
        # cyclical encoding so 23:00 is close to 00:00
        df["time_sin"] = np.sin(2 * np.pi * mins / 1440)
        df["time_cos"] = np.cos(2 * np.pi * mins / 1440)
        df["time_band"] = pd.cut(df["hour"], [-1, 5, 9, 15, 19, 23],
                                 labels=["Night", "AM peak", "Midday", "PM peak", "Evening"]).astype(str)
        if "peak_hour" not in roles:
            df["peak_hour"] = df["hour"].isin([7, 8, 9, 16, 17, 18, 19]).astype(int)
            roles["peak_hour"] = "peak_hour"
        if "scheduled_arrival" in roles:
            dur = (df[roles["scheduled_arrival"]] - t).dt.total_seconds() / 60
            df["scheduled_duration_min"] = dur.where(dur >= 0, dur + 1440)

    # ---------- Weather ----------
    if "weather" in roles:
        w = df[roles["weather"]].str.lower()
        df["is_rain"] = w.str.contains("rain|storm|drizzle|shower", na=False).astype(int)
        df["is_snow"] = w.str.contains("snow|sleet|ice|hail", na=False).astype(int)
        df["is_fog"] = w.str.contains("fog|mist|haze", na=False).astype(int)
        df["bad_weather"] = df[["is_rain", "is_snow", "is_fog"]].max(axis=1)
    if "temperature" in roles:
        temp = df[roles["temperature"]]
        df["extreme_temp"] = ((temp <= EXTREME_COLD_C) | (temp >= EXTREME_HEAT_C)).astype(int)
        df["temp_dev_from_comfort"] = (temp - 18).abs()
    if "precipitation" in roles:
        p = df[roles["precipitation"]]
        df["heavy_precip"] = (p >= p.quantile(0.9)).astype(int) if p.max() > 0 else 0
    if "wind" in roles:
        wd = df[roles["wind"]]
        df["high_wind"] = (wd >= wd.quantile(0.9)).astype(int)

    # ---------- City events ----------
    if "event_type" in roles:
        e = df[roles["event_type"]].str.lower()
        df["has_event"] = (~e.isin(["none", "no event", "unknown", "nan", ""])).astype(int)
    if "attendance" in roles:
        df["log_attendance"] = np.log1p(df[roles["attendance"]])
        if "has_event" not in df:
            df["has_event"] = (df[roles["attendance"]] > 0).astype(int)

    # ---------- Interactions ----------
    if "has_event" in df and "peak_hour" in roles:
        df["event_x_peak"] = df["has_event"] * df[roles["peak_hour"]]
    if "bad_weather" in df and "congestion" in roles:
        df["badweather_x_congestion"] = df["bad_weather"] * df[roles["congestion"]]
    if "bad_weather" in df and "peak_hour" in roles:
        df["badweather_x_peak"] = df["bad_weather"] * df[roles["peak_hour"]]

    return df


def model_columns(df: pd.DataFrame, roles: dict, include_departure_delay=False):
    """Split columns into numeric / categorical model inputs, excluding leakage."""
    exclude = {roles.get("target"), roles.get("delayed_flag"), roles.get("trip_id"),
               roles.get("date"), roles.get("time"), roles.get("scheduled_arrival")}
    if not include_departure_delay:
        # departure delay is only known once the trip has started -> leakage
        exclude.add(roles.get("departure_delay"))
    exclude.discard(None)

    feats = [c for c in df.columns if c not in exclude]
    cat = [c for c in feats if df[c].dtype == object or str(df[c].dtype) in ("string", "category")]
    # drop ID-like / free-text / raw timestamp strings (e.g. "08:15:00")
    cat = [c for c in cat if df[c].nunique() <= 60]
    num = [c for c in feats if c not in cat and pd.api.types.is_numeric_dtype(df[c])
           and not pd.api.types.is_bool_dtype(df[c]) and df[c].nunique() > 1]
    return num, cat
