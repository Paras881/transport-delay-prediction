"""Exploratory analysis: how weather, events and time affect delays.

Every function saves a figure and returns a small summary table / test result
so the findings can be quoted in the report.
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

sns.set_theme(style="whitegrid", context="notebook")
BLUE, ORANGE, GREY = "#2E6FBA", "#E07B39", "#8A8F98"
DELAY_THRESHOLD = 5  # minutes: a trip counts as "delayed" above this


def _save(fig, out_dir, name):
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, name)
    fig.tight_layout()
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return path


def target_distribution(df, roles, out_dir):
    y = df[roles["target"]]
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.histplot(y, bins=40, color=BLUE, ax=ax)
    ax.axvline(y.mean(), color=ORANGE, ls="--", label=f"mean = {y.mean():.1f} min")
    ax.axvline(y.median(), color=GREY, ls=":", label=f"median = {y.median():.1f} min")
    ax.set(title="Distribution of arrival delay", xlabel="Delay (minutes)", ylabel="Trips")
    ax.legend()
    _save(fig, out_dir, "01_target_distribution.png")
    return y.describe().round(2)


def weather_effect(df, roles, out_dir):
    """Q1: how do rain, snow and extreme temperatures affect delays?"""
    tgt, res = roles["target"], {}
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))

    if "weather" in roles:
        w = roles["weather"]
        order = df.groupby(w)[tgt].median().sort_values().index
        sns.boxplot(data=df, x=w, y=tgt, order=order, color=BLUE, ax=axes[0], fliersize=2)
        axes[0].set(title="Delay by weather condition", xlabel="", ylabel="Delay (min)")
        axes[0].tick_params(axis="x", rotation=30)
        groups = [g[tgt].values for _, g in df.groupby(w)]
        h, p = stats.kruskal(*groups)
        res["weather_table"] = (df.groupby(w)[tgt]
                                .agg(trips="count", mean_delay="mean", median_delay="median",
                                     pct_delayed=lambda s: (s > DELAY_THRESHOLD).mean() * 100)
                                .sort_values("mean_delay", ascending=False).round(2))
        res["weather_kruskal_p"] = p

    if "temperature" in roles:
        t = roles["temperature"]
        bins = pd.cut(df[t], bins=[-50, 0, 10, 20, 30, 60],
                      labels=["≤0°C", "0–10", "10–20", "20–30", ">30°C"])
        tb = df.groupby(bins, observed=True)[tgt].agg(["mean", "count"])
        axes[1].bar(tb.index.astype(str), tb["mean"], color=BLUE)
        axes[1].set(title="Mean delay by temperature band", ylabel="Mean delay (min)")
        res["temperature_table"] = tb.round(2)
        res["temp_spearman"] = stats.spearmanr(df[t], df[tgt])

    if "precipitation" in roles:
        p = roles["precipitation"]
        sns.regplot(data=df, x=p, y=tgt, ax=axes[2], scatter_kws=dict(s=8, alpha=.35, color=GREY),
                    line_kws=dict(color=ORANGE), lowess=False)
        axes[2].set(title="Delay vs precipitation", xlabel="Precipitation (mm)", ylabel="Delay (min)")
        res["precip_spearman"] = stats.spearmanr(df[p], df[tgt])

    _save(fig, out_dir, "02_weather_effect.png")

    # Rain / snow / extreme temp vs normal - effect sizes
    rows = []
    for flag in ("is_rain", "is_snow", "is_fog", "extreme_temp", "heavy_precip", "high_wind"):
        if flag in df and df[flag].nunique() == 2:
            a, b = df.loc[df[flag] == 1, tgt], df.loc[df[flag] == 0, tgt]
            _, pval = stats.mannwhitneyu(a, b)
            rows.append(dict(condition=flag, trips=len(a), mean_delay_with=a.mean(),
                             mean_delay_without=b.mean(), diff_min=a.mean() - b.mean(), p_value=pval))
    res["weather_flags"] = pd.DataFrame(rows).round(3)
    return res


def event_effect(df, roles, out_dir):
    """Q2: do major city events increase delay probability?"""
    tgt, res = roles["target"], {}
    df = df.assign(is_delayed=(df[roles["target"]] > DELAY_THRESHOLD).astype(int))
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))

    if "has_event" in df:
        ct = pd.crosstab(df["has_event"], df["is_delayed"])
        chi2, p, *_ = stats.chi2_contingency(ct)
        prob = df.groupby("has_event")["is_delayed"].mean() * 100
        axes[0].bar(["No event", "Event nearby"], prob.reindex([0, 1]).values, color=[GREY, ORANGE])
        axes[0].set(title=f"P(delay > {DELAY_THRESHOLD} min)", ylabel="% of trips delayed")
        for i, v in enumerate(prob.reindex([0, 1]).values):
            axes[0].text(i, v + 1, f"{v:.1f}%", ha="center")
        p_ev, p_no = prob.get(1, np.nan), prob.get(0, np.nan)
        res.update(delay_prob_event=p_ev, delay_prob_no_event=p_no, event_chi2_p=p,
                   relative_risk=(p_ev / p_no) if p_no else np.nan)

    if "event_type" in roles:
        e = roles["event_type"]
        tab = (df.groupby(e).agg(trips=(tgt, "count"), mean_delay=(tgt, "mean"),
                                 pct_delayed=("is_delayed", lambda s: s.mean() * 100))
               .sort_values("mean_delay", ascending=False).round(2))
        axes[1].barh(tab.index.astype(str), tab["mean_delay"], color=BLUE)
        axes[1].invert_yaxis()
        axes[1].set(title="Mean delay by event type", xlabel="Mean delay (min)")
        res["event_table"] = tab

    if "attendance" in roles:
        a = roles["attendance"]
        ev = df[df[a] > 0]
        if len(ev) > 10:
            q = pd.qcut(ev[a], 4, duplicates="drop")
            tb = ev.groupby(q, observed=True)[tgt].mean()
            axes[2].plot(range(len(tb)), tb.values, marker="o", color=ORANGE)
            axes[2].set_xticks(range(len(tb)))
            axes[2].set_xticklabels([f"Q{i+1}" for i in range(len(tb))])
            axes[2].set(title="Mean delay by event size (attendance quartile)",
                        ylabel="Mean delay (min)", xlabel="Attendance quartile")
            res["attendance_spearman"] = stats.spearmanr(ev[a], ev[tgt])

    _save(fig, out_dir, "03_event_effect.png")
    return res


def time_patterns(df, roles, out_dir):
    tgt, res = roles["target"], {}
    fig, axes = plt.subplots(1, 2, figsize=(15, 4.5))
    if "hour" in df:
        hb = df.groupby("hour")[tgt].mean()
        axes[0].plot(hb.index, hb.values, marker="o", color=BLUE)
        axes[0].set(title="Mean delay by hour of day", xlabel="Hour", ylabel="Mean delay (min)")
        res["hourly"] = hb.round(2)
    if "hour" in df and "day_of_week" in df:
        pv = df.pivot_table(index="day_of_week", columns="hour", values=tgt, aggfunc="mean")
        pv.index = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"][: len(pv)] if len(pv) == 7 else pv.index
        sns.heatmap(pv, cmap="Blues", ax=axes[1], cbar_kws=dict(label="Mean delay (min)"))
        axes[1].set(title="Delay heatmap: weekday × hour", xlabel="Hour", ylabel="")
    _save(fig, out_dir, "04_time_patterns.png")

    if "season" in roles:
        res["season"] = df.groupby(roles["season"])[tgt].mean().round(2)
    if "peak_hour" in roles:
        res["peak"] = df.groupby(roles["peak_hour"])[tgt].mean().round(2)
    if "transport_type" in roles:
        res["transport"] = df.groupby(roles["transport_type"])[tgt].mean().sort_values().round(2)
    return res


def correlations(df, roles, out_dir):
    tgt = roles["target"]
    num = df.select_dtypes("number").drop(columns=[roles.get("delayed_flag")], errors="ignore")
    num = num.loc[:, num.nunique() > 1]
    corr = num.corr(method="spearman")[tgt].drop(tgt).sort_values()
    fig, ax = plt.subplots(figsize=(7, max(4, 0.28 * len(corr))))
    ax.barh(corr.index, corr.values, color=[ORANGE if v > 0 else BLUE for v in corr.values])
    ax.set(title="Spearman correlation with delay", xlabel="ρ")
    _save(fig, out_dir, "05_correlations.png")
    return corr.round(3)
