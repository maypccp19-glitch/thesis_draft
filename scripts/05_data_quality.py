"""
Step 5 - Data-quality diagnostics.

Digs into four things the cleaning/zoning pipeline (scripts/01, 02) glosses
over, all of which matter for how comparable the per-zone b-values and
mainshock thresholds actually are:

  (a) Magnitude-type composition by network and by zone (TMD's ML vs. USGS's
      mb/mww/ml/... mix).
  (b) Dedup offset diagnostics: how close TMD-USGS matched pairs actually
      are in time/distance, to check the 60s/50km cutoffs used by dedupe().
  (c) Missing-data audit: rows dropped at each cleaning filter step.
  (d) Unclassified-event audit: where the ~7.7% "Unclassified" events are.

Does not modify load_tmd(), load_usgs(), or dedupe() in 01_clean_data.py -
uses wrapper functions and the additive dedupe_with_diagnostics() instead.

Produces:
  figures/09_magtype_composition.png
  figures/10_dedup_offset_diagnostics.png
  reports/magtype_composition.csv
  reports/missing_data_audit.csv
  reports/unclassified_audit.csv
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

_clean_mod = load_script("01_clean_data")
_eda_mod = load_script("03_eda")
ZONE_ORDER = _eda_mod.ZONE_ORDER  # includes "Unclassified"

IN_PATH_ZONES = "data/processed/catalog_with_zones.csv"
FIG_DIR = "figures"
REPORT_DIR = "reports"

DEDUP_TIME_WINDOW_S = 60
DEDUP_DIST_WINDOW_KM = 50


# --- (a) magnitude-type composition -----------------------------------------
def magtype_composition(df):
    net_ct = df.groupby(["network", "mag_type"]).size().reset_index(name="count")
    net_ct.insert(0, "group_type", "network")
    net_ct = net_ct.rename(columns={"network": "group_value"})

    zone_ct = df.groupby(["zone_code", "mag_type"]).size().reset_index(name="count")
    zone_ct.insert(0, "group_type", "zone")
    zone_ct = zone_ct.rename(columns={"zone_code": "group_value"})

    combined = pd.concat([net_ct, zone_ct], ignore_index=True)
    combined.to_csv(f"{REPORT_DIR}/magtype_composition.csv", index=False)
    print("\nMagnitude-type composition by network:")
    print(net_ct.pivot(index="group_value", columns="mag_type", values="count").fillna(0).astype(int).to_string())
    return net_ct, zone_ct


def plot_magtype_composition(net_ct, zone_ct):
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    net_pivot = net_ct.pivot(index="group_value", columns="mag_type", values="count").fillna(0)
    net_pivot.plot(kind="bar", stacked=True, ax=axes[0], colormap="tab20")
    axes[0].set_title("Magnitude-type composition by network")
    axes[0].set_xlabel("Network")
    axes[0].set_ylabel("Number of events")
    axes[0].legend(fontsize=7, ncol=2, title="mag_type")
    axes[0].tick_params(axis="x", rotation=30)

    zone_pivot = zone_ct.pivot(index="group_value", columns="mag_type", values="count").fillna(0)
    zone_pivot = zone_pivot.loc[[z for z in ZONE_ORDER if z in zone_pivot.index]]
    zone_pivot.plot(kind="bar", stacked=True, ax=axes[1], colormap="tab20")
    axes[1].set_title("Magnitude-type composition by zone")
    axes[1].set_xlabel("Zone")
    axes[1].set_ylabel("Number of events")
    axes[1].legend(fontsize=7, ncol=2, title="mag_type")

    fig.suptitle("Magnitude-type mixing across networks/zones\n"
                 "(affects comparability of b-values and mainshock thresholds across zones)")
    fig.tight_layout()
    fig.savefig(f"{FIG_DIR}/09_magtype_composition.png", dpi=150)
    plt.close(fig)


# --- (b) dedup offset diagnostics -------------------------------------------
def dedup_offset_diagnostics(time_window_s=DEDUP_TIME_WINDOW_S, dist_window_km=DEDUP_DIST_WINDOW_KM):
    tmd = _clean_mod.load_tmd()
    usgs = _clean_mod.load_usgs()
    _, match_df = _clean_mod.dedupe_with_diagnostics(tmd, usgs, time_window_s=time_window_s,
                                                       dist_window_km=dist_window_km)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].hist(match_df["time_offset_s"], bins=40, color="#1f77b4", alpha=0.75)
    axes[0].axvline(time_window_s, color="k", ls="--", label=f"cutoff = {time_window_s}s")
    axes[0].set_xlabel("Time offset (s)")
    axes[0].set_ylabel("Matched pairs")
    axes[0].set_title("TMD-USGS matched pair time offsets")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].hist(match_df["distance_km"], bins=40, color="#ff7f0e", alpha=0.75)
    axes[1].axvline(dist_window_km, color="k", ls="--", label=f"cutoff = {dist_window_km}km")
    axes[1].set_xlabel("Distance offset (km)")
    axes[1].set_ylabel("Matched pairs")
    axes[1].set_title("TMD-USGS matched pair distance offsets")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    fig.suptitle(f"Dedup offset diagnostics (n={len(match_df)} matched pairs) - "
                 f"checks whether the {time_window_s}s/{dist_window_km}km cutoffs are well chosen")
    fig.tight_layout()
    fig.savefig(f"{FIG_DIR}/10_dedup_offset_diagnostics.png", dpi=150)
    plt.close(fig)

    print(f"\nDedup diagnostics: {len(match_df)} matched pairs")
    print(match_df[["time_offset_s", "distance_km"]].describe().to_string())
    near_time = int((match_df["time_offset_s"] > time_window_s * 0.8).sum())
    near_dist = int((match_df["distance_km"] > dist_window_km * 0.8).sum())
    print(f"Pairs within 80-100% of the time cutoff ({time_window_s}s): {near_time}")
    print(f"Pairs within 80-100% of the distance cutoff ({dist_window_km}km): {near_dist}")
    return match_df


# --- (c) missing-data audit --------------------------------------------------
def load_tmd_with_audit():
    """Mirrors load_tmd()'s filtering but records rows dropped at each step
    separately (load_tmd() itself only reports the combined before/after)."""
    df = pd.read_csv(_clean_mod.RAW_TMD, encoding="cp874")
    df.columns = [c.strip() for c in df.columns]
    n_raw = len(df)

    out = pd.DataFrame()
    out["time_utc"] = pd.to_datetime(df["DATE-TIME UTC"], errors="coerce")
    out["lat"] = pd.to_numeric(df["LAT."], errors="coerce")
    out["lon"] = pd.to_numeric(df["LONG."], errors="coerce")
    out["mag"] = pd.to_numeric(df["MAG."], errors="coerce")

    steps = [("raw_rows", n_raw)]
    n = len(out)
    out = out.dropna(subset=["time_utc"]); steps.append(("dropped_unparseable_time", n - len(out))); n = len(out)
    out = out.dropna(subset=["lat", "lon"]); steps.append(("dropped_unparseable_latlon", n - len(out))); n = len(out)
    out = out[(out["lat"].between(-90, 90)) & (out["lon"].between(-180, 180))]
    steps.append(("dropped_out_of_range_latlon", n - len(out))); n = len(out)
    out = out.dropna(subset=["mag"]); steps.append(("dropped_unparseable_mag", n - len(out))); n = len(out)
    out = out[out["mag"] > 0]; steps.append(("dropped_mag_le_0", n - len(out))); n = len(out)
    steps.append(("kept", n))
    return out, steps


def load_usgs_with_audit():
    """Mirrors load_usgs()'s filtering but records rows dropped at each step."""
    df = pd.read_csv(_clean_mod.RAW_USGS)
    n_raw = len(df)
    steps = [("raw_rows", n_raw)]

    df = df[df["type"] == "earthquake"]
    steps.append(("dropped_non_earthquake_type", n_raw - len(df)))
    n = len(df)

    out = pd.DataFrame()
    out["time_utc"] = pd.to_datetime(df["time"], errors="coerce", utc=True).dt.tz_localize(None)
    out["lat"] = pd.to_numeric(df["latitude"], errors="coerce")
    out["lon"] = pd.to_numeric(df["longitude"], errors="coerce")
    out["mag"] = pd.to_numeric(df["mag"], errors="coerce")

    out = out.dropna(subset=["time_utc"]); steps.append(("dropped_unparseable_time", n - len(out))); n = len(out)
    out = out.dropna(subset=["lat", "lon"]); steps.append(("dropped_unparseable_latlon", n - len(out))); n = len(out)
    out = out[(out["lat"].between(-90, 90)) & (out["lon"].between(-180, 180))]
    steps.append(("dropped_out_of_range_latlon", n - len(out))); n = len(out)
    out = out.dropna(subset=["mag"]); steps.append(("dropped_unparseable_mag", n - len(out))); n = len(out)
    out = out[out["mag"] > 0]; steps.append(("dropped_mag_le_0", n - len(out))); n = len(out)
    steps.append(("kept", n))
    return out, steps


def missing_data_audit():
    _, tmd_steps = load_tmd_with_audit()
    _, usgs_steps = load_usgs_with_audit()
    rows = [{"source": "TMD", "step": s, "count": c} for s, c in tmd_steps]
    rows += [{"source": "USGS", "step": s, "count": c} for s, c in usgs_steps]
    audit = pd.DataFrame(rows)
    audit.to_csv(f"{REPORT_DIR}/missing_data_audit.csv", index=False)
    print("\nMissing-data audit (rows dropped per filter step):")
    print(audit.to_string(index=False))
    return audit


# --- (d) unclassified-event audit -------------------------------------------
def unclassified_audit(df):
    unclass = df[df["zone_code"] == "Unclassified"]
    frac = len(unclass) / len(df) * 100
    print(f"\nUnclassified events: {len(unclass)} / {len(df)} ({frac:.1f}%)")

    place_counts = unclass["place"].value_counts().head(30).reset_index()
    place_counts.columns = ["place", "count"]
    place_counts.to_csv(f"{REPORT_DIR}/unclassified_audit.csv", index=False)

    print("Top locations among unclassified events (see reports/unclassified_audit.csv for full list):")
    print(place_counts.head(15).to_string(index=False))
    print(f"\nUnclassified lat range: [{unclass['lat'].min():.2f}, {unclass['lat'].max():.2f}], "
          f"lon range: [{unclass['lon'].min():.2f}, {unclass['lon'].max():.2f}]")
    return unclass, place_counts


def main():
    os.makedirs(FIG_DIR, exist_ok=True)
    os.makedirs(REPORT_DIR, exist_ok=True)

    df = pd.read_csv(IN_PATH_ZONES, parse_dates=["time_utc"])

    print("Computing magnitude-type composition...")
    net_ct, zone_ct = magtype_composition(df)
    plot_magtype_composition(net_ct, zone_ct)

    print("\nComputing dedup offset diagnostics...")
    dedup_offset_diagnostics()

    print("\nRunning missing-data audit...")
    missing_data_audit()

    print("\nRunning unclassified-event audit...")
    unclassified_audit(df)

    print(f"\nDone. Figures: {FIG_DIR}/09_magtype_composition.png, {FIG_DIR}/10_dedup_offset_diagnostics.png")
    print(f"Tables: {REPORT_DIR}/magtype_composition.csv, {REPORT_DIR}/missing_data_audit.csv, "
          f"{REPORT_DIR}/unclassified_audit.csv")


if __name__ == "__main__":
    main()
