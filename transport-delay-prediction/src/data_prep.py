"""Loading and cleaning the Public Transport Delays dataset.

Column names are normalised to snake_case and then mapped to standard
"roles" (weather, event_type, target ...) by keyword matching, so the code
keeps working even if the CSV's exact headers differ slightly.
"""
import re
import numpy as np
import pandas as pd

# role -> list of keyword patterns (first matching column wins)
ROLE_PATTERNS = {
    "trip_id":        [r"trip_id", r"^id$"],
    "date":           [r"^date$", r"trip_date", r"date"],
    "time":           [r"^time$", r"scheduled_departure", r"departure_time", r"time"],
    "scheduled_arrival": [r"scheduled_arrival", r"arrival_time"],
    "transport_type": [r"transport_type", r"mode", r"vehicle_type"],
    "route":          [r"route"],
    "origin":         [r"origin"],
    "destination":    [r"destination"],
    "weather":        [r"weather_condition", r"weather"],
    "temperature":    [r"temp"],
    "humidity":       [r"humid"],
    "wind":           [r"wind"],
    "precipitation":  [r"precip", r"rain_mm"],
    "event_type":     [r"event_type", r"^event$"],
    "attendance":     [r"attendance"],
    "congestion":     [r"congestion_index", r"traffic_congestion", r"congestion"],
    "holiday":        [r"holiday"],
    "peak_hour":      [r"peak"],
    "weekday":        [r"weekday", r"day_of_week"],
    "season":         [r"season"],
    "departure_delay": [r"departure_delay"],
    "target":         [r"arrival_delay", r"delay_min", r"delay_minutes", r"^delay$"],
    "delayed_flag":   [r"^delayed$", r"is_delayed", r"delay_flag", r"delayed"],
}


def to_snake(name: str) -> str:
    name = re.sub(r"[^0-9a-zA-Z]+", "_", str(name)).strip("_")
    name = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", name)
    return name.lower()


def detect_roles(columns):
    """Return {role: column_name} for every role we can find."""
    roles, used = {}, set()
    for role, patterns in ROLE_PATTERNS.items():
        for pat in patterns:
            hit = next((c for c in columns if c not in used and re.search(pat, c)), None)
            if hit:
                roles[role] = hit
                used.add(hit)
                break
    return roles


def load_raw(path):
    # keep the literal string "None" (= no event) instead of treating it as missing
    df = pd.read_csv(path, keep_default_na=False,
                     na_values=["", "NA", "N/A", "NaN", "nan", "null", "NULL", "?"])
    df.columns = [to_snake(c) for c in df.columns]
    return df


def _to_bool_int(s: pd.Series) -> pd.Series:
    mapping = {"yes": 1, "no": 0, "true": 1, "false": 0, "1": 1, "0": 0, "y": 1, "n": 0}
    return s.astype(str).str.strip().str.lower().map(mapping).astype("float")


def clean(df: pd.DataFrame, roles: dict, verbose=True) -> pd.DataFrame:
    """Standard cleaning: dedupe, types, missing values, outliers."""
    log = []
    df = df.copy()
    n0 = len(df)

    df = df.drop_duplicates()
    log.append(f"Dropped {n0 - len(df)} duplicate rows")

    tgt = roles["target"]
    df[tgt] = pd.to_numeric(df[tgt], errors="coerce")
    n1 = len(df)
    df = df.dropna(subset=[tgt])
    log.append(f"Dropped {n1 - len(df)} rows with missing target ({tgt})")

    # Parse date/time
    if "date" in roles:
        df[roles["date"]] = pd.to_datetime(df[roles["date"]], errors="coerce")
    for r in ("time", "scheduled_arrival"):
        if r in roles:
            df[roles[r]] = pd.to_datetime(df[roles[r]].astype(str), format="mixed", errors="coerce")

    # Binary flags -> 0/1
    for r in ("holiday", "peak_hour", "delayed_flag"):
        if r in roles and df[roles[r]].dtype == object:
            df[roles[r]] = _to_bool_int(df[roles[r]])

    # Numeric columns: coerce, clip impossible values, impute median
    numeric_roles = ["temperature", "humidity", "wind", "precipitation",
                     "attendance", "congestion", "departure_delay"]
    for r in numeric_roles:
        if r in roles:
            c = roles[r]
            df[c] = pd.to_numeric(df[c], errors="coerce")
            if r in ("humidity",):
                df[c] = df[c].clip(0, 100)
            if r in ("wind", "precipitation", "attendance"):
                df[c] = df[c].clip(lower=0)
            miss = df[c].isna().sum()
            if miss:
                df[c] = df[c].fillna(df[c].median())
                log.append(f"Imputed {miss} missing values in {c} with median")

    # Categoricals: tidy text, fill missing
    for r in ("transport_type", "route", "origin", "destination", "weather",
              "event_type", "weekday", "season"):
        if r in roles:
            c = roles[r]
            df[c] = df[c].astype("string").str.strip()
            fill = "None" if r == "event_type" else "Unknown"
            miss = df[c].isna().sum()
            df[c] = df[c].fillna(fill).replace({"": fill, "nan": fill})
            if miss:
                log.append(f"Filled {miss} missing values in {c} with '{fill}'")

    # Cap extreme target outliers (keeps early arrivals, i.e. negatives)
    lo, hi = df[tgt].quantile([0.005, 0.995])
    n_out = ((df[tgt] < lo) | (df[tgt] > hi)).sum()
    df[tgt] = df[tgt].clip(lo, hi)
    log.append(f"Winsorised {n_out} extreme target values to [{lo:.1f}, {hi:.1f}] min")

    if verbose:
        print("\n".join("  - " + l for l in log))
    return df.reset_index(drop=True)
