"""
Step 6 - Temporal completeness refinement.

scripts/03_eda.py's b-value analysis computes a single Mc per zone over the
entire 2007-2026 span, but the temporal EDA (figures/03_temporal_distribution.png)
already shows TMD's event rate changing a lot over time (network build-out).
This checks whether Mc is actually stationary, by recomputing it in rolling
time windows per zone, reusing (not modifying) max_curvature_mc() from
scripts/03_eda.py.

Produces:
  figures/11_mc_over_time.png
  reports/completeness_over_time.csv
"""
import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _modutil import load_script

_eda_mod = load_script("03_eda")
max_curvature_mc = _eda_mod.max_curvature_mc
ZONE_COLOR = _eda_mod.ZONE_COLOR
ZONES = [z for z in _eda_mod.ZONE_ORDER if z != "Unclassified"]

IN_PATH = "data/processed/catalog_with_zones.csv"
FIG_DIR = "figures"
REPORT_DIR = "reports"

ROLLING_WINDOW_YEARS = 5
ROLLING_STEP_YEARS = 1
MIN_EVENTS_PER_WINDOW = 30


def rolling_mc(df, window_years=ROLLING_WINDOW_YEARS, step_years=ROLLING_STEP_YEARS,
                min_events=MIN_EVENTS_PER_WINDOW):
    df = df.copy()
    df["year"] = df["time_utc"].dt.year + (df["time_utc"].dt.dayofyear - 1) / 365.25

    results = []
    for zone in ZONES:
        sub = df[df["zone_code"] == zone]
        if len(sub) < min_events:
            continue
        y_min, y_max = sub["year"].min(), sub["year"].max()
        start = y_min
        while start < y_max:
            end = start + window_years
            window = sub[(sub["year"] >= start) & (sub["year"] < end)]
            if len(window) >= min_events:
                mc = max_curvature_mc(window["mag"].dropna().values)
                results.append({"zone_code": zone, "window_start": round(start, 2),
                                 "window_end": round(end, 2), "window_center": round((start + end) / 2, 2),
                                 "n_events": len(window), "mc": mc})
            start += step_years
    return pd.DataFrame(results)


def plot_mc_over_time(mc_df):
    fig, ax = plt.subplots(figsize=(12, 6))
    for zone in ZONES:
        sub = mc_df[mc_df["zone_code"] == zone]
        if len(sub) == 0:
            continue
        ax.plot(sub["window_center"], sub["mc"], marker="o", ms=3, color=ZONE_COLOR[zone], label=zone)
    ax.axvline(2012, color="k", ls=":", lw=1, label="USGS catalog starts (2012)")
    ax.set_xlabel("Year (rolling window center)")
    ax.set_ylabel("Mc (completeness magnitude, max-curvature)")
    ax.set_title(f"Rolling-window magnitude of completeness by zone\n"
                 f"({ROLLING_WINDOW_YEARS}-year window, {ROLLING_STEP_YEARS}-year step, "
                 f">= {MIN_EVENTS_PER_WINDOW} events/window)")
    ax.legend(fontsize=8, ncol=2)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(f"{FIG_DIR}/11_mc_over_time.png", dpi=150)
    plt.close(fig)


def main():
    os.makedirs(FIG_DIR, exist_ok=True)
    os.makedirs(REPORT_DIR, exist_ok=True)

    df = pd.read_csv(IN_PATH, parse_dates=["time_utc"])
    df = df[df["zone_code"] != "Unclassified"]

    print(f"Computing rolling Mc (window={ROLLING_WINDOW_YEARS}y, step={ROLLING_STEP_YEARS}y)...")
    mc_df = rolling_mc(df)
    mc_df.to_csv(f"{REPORT_DIR}/completeness_over_time.csv", index=False)
    print(mc_df.to_string(index=False))

    plot_mc_over_time(mc_df)
    print(f"\nSaved {FIG_DIR}/11_mc_over_time.png and {REPORT_DIR}/completeness_over_time.csv")


if __name__ == "__main__":
    main()
