# 🚌 Predicting Public Transport Delays from Weather & City Events

A complete data-science project that predicts a trip's **arrival delay (minutes)** from external factors (weather, city events, traffic congestion and time) using the Kaggle dataset
[Public Transport Delays with Weather & Events](https://www.kaggle.com/datasets/khushikyad001/public-transport-delays-with-weather-and-events).

**Skills covered:** data cleaning · feature engineering · time-based analysis · regression models (Linear, Random Forest, XGBoost) · hypothesis testing · model interpretation

---

## Quick start

```bash
pip install -r requirements.txt

# 1. Download public_transport_delays.csv from Kaggle and put it in data/
#    (or: kaggle datasets download -d khushikyad001/public-transport-delays-with-weather-and-events -p data --unzip)

# 2. Run everything (cleaning → EDA → models → report), about 20 seconds
python run_pipeline.py

# 3. Predict delays for new trips with the saved model
python predict.py
```

Or open **`notebooks/transport_delay_prediction.ipynb`** for the step-by-step walkthrough with explanations. That's the best format for a portfolio or a submission.

> If the real CSV isn't in `data/` yet, everything runs on `data/sample_public_transport_delays.csv`, a **synthetic** file with the same structure (made by `data/make_sample_data.py`). The numbers it produces are only a demo, so rerun once you have the real file.

## Project structure

```
transport-delay-prediction/
├── data/
│   ├── public_transport_delays.csv        ← put the Kaggle file here
│   └── make_sample_data.py                ← synthetic stand-in generator
├── notebooks/transport_delay_prediction.ipynb
├── src/
│   ├── data_prep.py    # loading, column detection, cleaning
│   ├── features.py     # time / weather / event / interaction features
│   ├── eda.py          # Q1 weather & Q2 events analysis + stats tests + plots
│   └── modeling.py     # models, CV, tuning, permutation importance
├── run_pipeline.py     # one-command end-to-end run → outputs/report.md
├── predict.py          # score new trips with the saved model
└── outputs/            # figures/, models/best_model.joblib, report.md, CSVs
```

## Approach

### 1. Data cleaning (`src/data_prep.py`)
- Headers normalised to snake_case and **auto-mapped to roles** (weather, event, target…) by keyword, so small header differences don't break the code
- Duplicates removed; dates/times parsed; Yes/No flags converted to 0/1
- Impossible values clipped (humidity 0–100, non-negative rain/wind); numeric gaps filled with the median, categorical gaps with `"Unknown"`/`"None"`
- Extreme 1% of delays winsorised

### 2. Feature engineering (`src/features.py`)
| Group | Features |
|---|---|
| Time | hour, cyclical sin/cos time-of-day, day of week, weekend, month, time band, scheduled duration |
| Weather | `is_rain`, `is_snow`, `is_fog`, `bad_weather`, `extreme_temp` (≤0 °C / ≥32 °C), distance from 18 °C, `heavy_precip`, `high_wind` |
| Events | `has_event`, `log_attendance` |
| Interactions | event × peak, bad weather × congestion, bad weather × peak |

**Leakage prevention:** `actual_departure_delay_min` and the `delayed` flag are excluded from the inputs. Both are only known after the trip starts (or come from the target). Add `--include-departure-delay` to see how much they inflate scores.

### 3. Analysis questions (`src/eda.py`)
- **Q1: weather:** delay by condition (Kruskal–Wallis), rain/snow/fog/extreme-temperature effects (Mann–Whitney), temperature bands, precipitation trend
- **Q2: events:** P(delay > 5 min) with vs without an event (chi-square, relative risk), delay by event type and attendance size
- **Time:** hour-of-day profile, weekday × hour heatmap, peak vs off-peak, season, transport mode

### 4. Models (`src/modeling.py`)
- **Chronological 80/20 split:** train on past trips, test on the most recent ones (realistic forecasting), plus 5-fold CV on the training set
- Mean baseline → Linear Regression → Ridge → Random Forest → **XGBoost** (uses sklearn `HistGradientBoostingRegressor` automatically if `xgboost` isn't installed)
- Randomized hyper-parameter search on the best tree model
- Metrics: MAE, RMSE, R²; permutation feature importance; predicted-vs-actual and residual plots

## Outputs
After `python run_pipeline.py`:
- `outputs/report.md`: all findings, tables and charts in one auto-generated report
- `outputs/figures/01…07_*.png`: distribution, weather, events, time, correlations, model results, feature importance
- `outputs/model_comparison.csv`, `outputs/feature_importance.csv`
- `outputs/models/best_model.joblib`

## Ideas to extend
- Classification: predict P(delay > 5 min) with logistic regression / XGBoost classifier + ROC-AUC
- Quantile regression or conformal prediction for delay *ranges*
- SHAP values for per-trip explanations
- A Streamlit app on top of `predict.py`
- Validate on real data (GTFS-realtime + a weather API), since the Kaggle data is synthetic
