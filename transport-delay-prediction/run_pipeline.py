"""End-to-end pipeline: clean -> engineer features -> EDA -> train & compare models -> report.

Usage:
    python run_pipeline.py                                   # uses data/public_transport_delays.csv
    python run_pipeline.py --data path/to/file.csv
    python run_pipeline.py --include-departure-delay         # see note on leakage in README
    python run_pipeline.py --no-tune                         # skip hyper-parameter search (faster)
"""
import argparse
import json
import os
import sys
import warnings

import pandas as pd

from src.data_prep import load_raw, detect_roles, clean
from src.features import add_features, model_columns
from src import eda
from src.modeling import (build_models, evaluate_models, time_split, tune, importance,
                          diagnostic_plots, save_model, metrics)

warnings.filterwarnings("ignore")
ROOT = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(ROOT, "outputs", "figures")
OUT = os.path.join(ROOT, "outputs")


def pick_data(path):
    if path:
        return path
    real = os.path.join(ROOT, "data", "public_transport_delays.csv")
    if os.path.exists(real):
        return real
    sample = os.path.join(ROOT, "data", "sample_public_transport_delays.csv")
    if not os.path.exists(sample):
        os.system(f'"{sys.executable}" "{os.path.join(ROOT, "data", "make_sample_data.py")}"')
    print("\n!! Real Kaggle CSV not found in data/ — running on SYNTHETIC sample data. !!\n")
    return sample


def df_md(d):
    try:
        return d.to_markdown()
    except ImportError:  # tabulate not installed
        return "```\n" + d.to_string() + "\n```"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data")
    ap.add_argument("--include-departure-delay", action="store_true")
    ap.add_argument("--no-tune", action="store_true")
    args = ap.parse_args()

    path = pick_data(args.data)
    print(f"[1/6] Loading {path}")
    raw = load_raw(path)
    roles = detect_roles(list(raw.columns))
    print(f"      {raw.shape[0]} rows x {raw.shape[1]} cols")
    print("      Detected columns:", json.dumps(roles, indent=None))
    if "target" not in roles:
        sys.exit("Could not find a delay column (e.g. 'actual_arrival_delay_min'). "
                 "Rename your target column or edit ROLE_PATTERNS in src/data_prep.py.")

    print("[2/6] Cleaning")
    df = clean(raw, roles)

    print("[3/6] Feature engineering")
    df = add_features(df, roles)
    num, cat = model_columns(df, roles, args.include_departure_delay)
    print(f"      {len(num)} numeric + {len(cat)} categorical model features")

    print("[4/6] Exploratory analysis")
    desc = eda.target_distribution(df, roles, FIG)
    wx = eda.weather_effect(df, roles, FIG)
    ev = eda.event_effect(df, roles, FIG)
    tp = eda.time_patterns(df, roles, FIG)
    corr = eda.correlations(df, roles, FIG)

    print("[5/6] Training models (chronological 80/20 split + 5-fold CV on train)")
    tr, te = time_split(df, roles)
    X, y = df[num + cat], df[roles["target"]]
    X_tr, X_te, y_tr, y_te = X.loc[tr], X.loc[te], y.loc[tr], y.loc[te]
    results, fitted = evaluate_models(build_models(num, cat), X_tr, y_tr, X_te, y_te)

    best_name = results.drop("Baseline (mean)").sort_values("CV_MAE").index[0]
    best, best_params = fitted[best_name], None
    if not args.no_tune and best_name not in ("Linear Regression", "Ridge Regression"):
        print(f"      Tuning {best_name} ...")
        tuned, best_params = tune(best, X_tr, y_tr)
        tm = metrics(y_te, tuned.predict(X_te))
        results.loc[f"{best_name} (tuned)"] = [None, None, None, *[round(v, 3) for v in tm.values()]]
        if tm["MAE"] <= results.loc[best_name, "Test_MAE"]:
            best, best_name = tuned, f"{best_name} (tuned)"
        print(f"      tuned test MAE {tm['MAE']:.2f}")

    y_pred = best.predict(X_te)
    diagnostic_plots(y_te, y_pred, results, FIG, best_name)
    imp = importance(best, X_te, y_te, FIG)

    print("[6/6] Saving model and report")
    save_model(best, dict(model_name=best_name, numeric_features=num, categorical_features=cat,
                          roles=roles), os.path.join(OUT, "models", "best_model.joblib"))
    results.to_csv(os.path.join(OUT, "model_comparison.csv"))
    imp.to_csv(os.path.join(OUT, "feature_importance.csv"), index=False)

    # --------------- Markdown report ---------------
    bm = metrics(y_te, y_pred)
    base = results.loc["Baseline (mean)"]
    L = [f"# Public Transport Delay Prediction — Results\n",
         f"Data: `{os.path.basename(path)}` — {len(df)} trips after cleaning. "
         f"Target: `{roles['target']}` (minutes).\n",
         "## 1. Delay overview\n", df_md(desc.to_frame("delay_min")), "\n![](figures/01_target_distribution.png)\n",
         "## 2. Weather effects\n"]
    if "weather_table" in wx:
        L += [df_md(wx["weather_table"]),
              f"\nKruskal–Wallis test across weather conditions: p = {wx['weather_kruskal_p']:.2g}\n"]
    if len(wx.get("weather_flags", [])):
        L += ["\nEffect of each adverse condition (mean delay with vs without, Mann–Whitney p-value):\n",
              df_md(wx["weather_flags"].set_index("condition"))]
    if "temperature_table" in wx:
        L += ["\nMean delay by temperature band:\n", df_md(wx["temperature_table"])]
    L += ["\n![](figures/02_weather_effect.png)\n", "## 3. City events\n"]
    if "delay_prob_event" in ev:
        L += [f"- P(delay > {eda.DELAY_THRESHOLD} min) **with** an event nearby: **{ev['delay_prob_event']:.1f}%**",
              f"- P(delay > {eda.DELAY_THRESHOLD} min) **without** an event: **{ev['delay_prob_no_event']:.1f}%**",
              f"- Relative risk: **{ev['relative_risk']:.2f}×** (chi-square p = {ev['event_chi2_p']:.2g})\n"]
    if "event_table" in ev:
        L += [df_md(ev["event_table"])]
    L += ["\n![](figures/03_event_effect.png)\n", "## 4. Time patterns\n"]
    for k, title in (("peak", "Peak hour (1) vs off-peak (0)"), ("season", "Season"), ("transport", "Transport type")):
        if k in tp:
            L += [f"**{title}** — mean delay:\n", df_md(tp[k].to_frame("mean_delay_min")), ""]
    L += ["![](figures/04_time_patterns.png)\n", "![](figures/05_correlations.png)\n",
          "## 5. Models\n",
          "Chronological split (train on earliest 80% of trips, test on latest 20%); "
          "CV = 5-fold on the training set.\n", df_md(results),
          f"\n**Best model: {best_name}** — test MAE {bm['MAE']:.2f} min, RMSE {bm['RMSE']:.2f}, "
          f"R² {bm['R2']:.3f} (baseline MAE {base['Test_MAE']:.2f}, "
          f"{(1 - bm['MAE'] / base['Test_MAE']) * 100:.0f}% better).\n"]
    if best_params:
        L += [f"Tuned parameters: `{best_params}`\n"]
    L += ["![](figures/06_model_results.png)\n", "### Top features (permutation importance)\n",
          df_md(imp.head(15).set_index("feature")), "\n![](figures/07_feature_importance.png)\n"]
    if not args.include_departure_delay and "departure_delay" in roles:
        L += [f"\n_Note: `{roles['departure_delay']}` was excluded — it is only known after the trip starts, "
              "so using it would leak the answer. Run with `--include-departure-delay` to see its effect._\n"]
    with open(os.path.join(OUT, "report.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(L))

    print("\nDone. Model comparison:\n", results.to_string())
    print(f"\nReport: outputs/report.md | Figures: outputs/figures/ | Model: outputs/models/best_model.joblib")


if __name__ == "__main__":
    main()
