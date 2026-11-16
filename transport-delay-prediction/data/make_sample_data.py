"""Generate a synthetic stand-in for public_transport_delays.csv.

Only used so the project runs before you download the real Kaggle file.
Same structure as the Kaggle dataset (2,000 trips; timetable, weather,
events, congestion, calendar flags). Results on this file are NOT real findings.

    python data/make_sample_data.py
"""
import os
import numpy as np
import pandas as pd

rng = np.random.default_rng(7)
N = 2000

dates = pd.to_datetime("2024-01-01") + pd.to_timedelta(rng.integers(0, 365, N), unit="D")
hours = rng.choice(range(5, 24), N, p=np.array([1, 3, 6, 7, 5, 4, 4, 4, 4, 4, 5, 7, 7, 6, 4, 3, 2, 2, 1]) / 79)
mins = rng.integers(0, 60, N)
dep = dates + pd.to_timedelta(hours, unit="h") + pd.to_timedelta(mins, unit="m")
transport = rng.choice(["Bus", "Train", "Tram", "Metro"], N, p=[.4, .25, .2, .15])
dur = np.where(transport == "Train", rng.integers(25, 90, N), rng.integers(10, 45, N))
arr = dep + pd.to_timedelta(dur, unit="m")

month = dates.month
season = np.select([np.isin(month, [12, 1, 2]), np.isin(month, [3, 4, 5]), np.isin(month, [6, 7, 8])],
                   ["Winter", "Spring", "Summer"], "Autumn")
base_temp = {"Winter": 1, "Spring": 12, "Summer": 26, "Autumn": 13}
temp = np.array([base_temp[s] for s in season]) + rng.normal(0, 6, N)

weather = []
for s, t in zip(season, temp):
    if t < 1 and rng.random() < .45:
        weather.append("Snow")
    else:
        weather.append(rng.choice(["Clear", "Cloudy", "Rain", "Fog", "Storm"], p=[.4, .27, .2, .07, .06]))
weather = np.array(weather)
precip = np.where(np.isin(weather, ["Rain", "Storm", "Snow"]), rng.gamma(2, 3, N), rng.exponential(.2, N))
wind = rng.gamma(3, 5, N) + np.where(weather == "Storm", 25, 0)
humidity = np.clip(55 + np.where(np.isin(weather, ["Rain", "Fog", "Storm"]), 25, 0) + rng.normal(0, 10, N), 15, 100)

event = rng.choice(["None", "Concert", "Sports", "Festival", "Protest", "Parade"], N,
                   p=[.7, .08, .09, .05, .04, .04])
attendance = np.where(event == "None", 0, rng.integers(500, 60000, N))
weekday_name = dates.day_name()
holiday = (rng.random(N) < .04).astype(int)
peak = np.isin(hours, [7, 8, 9, 16, 17, 18, 19]).astype(int)
congestion = np.clip(35 + 25 * peak + 10 * (event != "None") + 8 * np.isin(weather, ["Rain", "Snow", "Storm"])
                     + rng.normal(0, 12, N), 0, 100)

w_eff = pd.Series(weather).map({"Clear": 0, "Cloudy": .5, "Rain": 3.5, "Fog": 2.5, "Storm": 7, "Snow": 9}).values
mode_eff = pd.Series(transport).map({"Bus": 2.5, "Tram": 1.5, "Train": 0, "Metro": -1}).values
surface = np.isin(transport, ["Bus", "Tram"])
ev_eff = np.where(event != "None", 2 + 3.5 * np.log10(attendance + 1) / 4.8, 0) * np.where(surface, 1.4, .8)
delay = (-4 + w_eff + mode_eff + ev_eff + .09 * congestion * surface + .04 * congestion
         + 2.5 * peak + 2.5 * ((temp < 0) | (temp > 32)) + .25 * precip - 1.5 * holiday
         + rng.normal(0, 4, N))
arr_delay = np.round(delay, 0).astype(int)
dep_delay = np.round(arr_delay * .7 + rng.normal(0, 2, N)).astype(int)

df = pd.DataFrame({
    "trip_id": [f"T{i:05d}" for i in range(N)],
    "date": dates.strftime("%Y-%m-%d"),
    "time": dep.strftime("%H:%M:%S"),
    "transport_type": transport,
    "route_id": [f"Route_{r}" for r in rng.integers(1, 21, N)],
    "origin_station": [f"Station_{s}" for s in rng.integers(1, 31, N)],
    "destination_station": [f"Station_{s}" for s in rng.integers(1, 31, N)],
    "scheduled_departure": dep.strftime("%H:%M:%S"),
    "scheduled_arrival": arr.strftime("%H:%M:%S"),
    "actual_departure_delay_min": dep_delay,
    "actual_arrival_delay_min": arr_delay,
    "weather_condition": weather,
    "temperature_C": temp.round(1),
    "humidity_percent": humidity.round(0),
    "wind_speed_kmh": wind.round(1),
    "precipitation_mm": precip.round(1),
    "event_type": event,
    "event_attendance_est": attendance,
    "traffic_congestion_index": congestion.round(0),
    "holiday": holiday,
    "peak_hour": peak,
    "weekday": weekday_name,
    "season": season,
    "delayed": (arr_delay > 5).astype(int),
})
# sprinkle realistic dirt: missing values & duplicates
for c in ["temperature_C", "humidity_percent", "wind_speed_kmh", "weather_condition"]:
    df.loc[rng.choice(N, 15, replace=False), c] = np.nan
df = pd.concat([df, df.sample(10, random_state=1)], ignore_index=True)

out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sample_public_transport_delays.csv")
df.to_csv(out, index=False)
print(f"Wrote {len(df)} rows -> {out}")
