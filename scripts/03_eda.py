"""
Step 3 - Exploratory data analysis on the classified catalog.

Produces:
  figures/01_event_map.png           spatial distribution by source zone
  figures/02_magnitude_frequency.png Gutenberg-Richter plots per zone
  figures/03_temporal_distribution.png events per year by network
  figures/04_depth_histogram.png     depth distribution by zone
  reports/zone_summary.csv           per-zone summary statistics table
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

IN_PATH = "data/processed/catalog_with_zones.csv"
FIG_DIR = "figures"
REPORT_DIR = "reports"

ZONE_ORDER = ["Z1", "Z2", "Z3", "Z4", "Z5", "Z6", "Z7", "Unclassified"]
ZONE_COLOR = {
    "Z1": "#d62728", "Z2": "#ff7f0e", "Z3": "#2ca02c", "Z4": "#1f77b4",
    "Z5": "#9467bd", "Z6": "#8c564b", "Z7": "#17becf", "Unclassified": "#bbbbbb",
}


def max_curvature_mc(mags, dm=0.1):
    """Completeness magnitude via the maximum-curvature method: the
    magnitude bin with the highest event count (mode of the FMD)."""
    bins = np.arange(np.floor(mags.min() / dm) * dm, mags.max() + dm, dm)
    counts, edges = np.histogram(mags, bins=bins)
    if counts.sum() == 0:
        return np.nan
    return edges[np.argmax(counts)]


def aki_utsu_b_value(mags, mc, dm=0.1):
    """Maximum-likelihood b-value (Aki 1965 / Utsu 1965) for events >= mc."""
    m = mags[mags >= mc - dm / 2]
    n = len(m)
    if n < 30:
        return np.nan, np.nan, n
    mean_m = m.mean()
    b = np.log10(np.e) / (mean_m - (mc - dm / 2))
    b_std = b / np.sqrt(n)
    return b, b_std, n


def plot_event_map(df):
    fig, ax = plt.subplots(figsize=(11, 9))
    for zone in ZONE_ORDER:
        sub = df[df["zone_code"] == zone]
        if len(sub) == 0:
            continue
        ax.scatter(sub["lon"], sub["lat"], s=np.clip((sub["mag"] - 1) * 2, 1, None),
                   color=ZONE_COLOR[zone], alpha=0.35, linewidths=0, label=f"{zone} (n={len(sub)})")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title("Southeast Asia seismicity (2007-2026) by proposed source zone")
    ax.set_aspect("equal")
    ax.legend(markerscale=3, fontsize=8, loc="upper left", framealpha=0.9)
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(f"{FIG_DIR}/01_event_map.png", dpi=150)
    plt.close(fig)


def plot_magnitude_frequency(df, zone_summary_rows):
    zones = [z for z in ZONE_ORDER if z != "Unclassified"]
    fig, axes = plt.subplots(2, 4, figsize=(18, 8))
    axes = axes.flatten()
    dm = 0.1
    for i, zone in enumerate(zones):
        ax = axes[i]
        sub = df[df["zone_code"] == zone]
        mags = sub["mag"].dropna().values
        if len(mags) < 30:
            ax.set_title(f"{zone} (n={len(mags)}, too few events)")
            continue
        mc = max_curvature_mc(mags, dm)
        b, b_std, n_above = aki_utsu_b_value(mags, mc, dm)

        bins = np.arange(np.floor(mags.min()), np.ceil(mags.max()) + dm, dm)
        counts, edges = np.histogram(mags, bins=bins)
        centers = edges[:-1] + dm / 2
        cum = np.cumsum(counts[::-1])[::-1]  # N(>=M)

        ax.semilogy(centers, cum, "o", ms=3, color=ZONE_COLOR[zone])
        ax.axvline(mc, color="k", ls="--", lw=1, label=f"Mc={mc:.1f}")
        ax.set_title(f"{zone}: n={len(mags)}, b={b:.2f}±{b_std:.2f} (n≥Mc={n_above})")
        ax.set_xlabel("Magnitude")
        ax.set_ylabel("N (≥ M)")
        ax.legend(fontsize=7)
        ax.grid(alpha=0.3)

        zone_summary_rows.append({"zone_code": zone, "mc": mc, "b_value": b,
                                   "b_value_std": b_std, "n_events_ge_mc": n_above})
    for j in range(len(zones), len(axes)):
        axes[j].axis("off")
    fig.suptitle("Frequency-magnitude distributions by source zone (mixed magnitude types - see caveats)")
    fig.tight_layout()
    fig.savefig(f"{FIG_DIR}/02_magnitude_frequency.png", dpi=150)
    plt.close(fig)


def plot_temporal(df):
    df = df.copy()
    df["year"] = pd.to_datetime(df["time_utc"]).dt.year
    df["net_group"] = df["network"].apply(lambda x: "TMD" if x == "TMD" else "USGS")
    pivot = df.pivot_table(index="year", columns="net_group", values="event_id", aggfunc="count").fillna(0)
    fig, ax = plt.subplots(figsize=(11, 5))
    pivot.plot(kind="bar", stacked=True, ax=ax, color={"TMD": "#2ca02c", "USGS": "#1f77b4"})
    ax.set_ylabel("Number of events")
    ax.set_title("Events per year by network (catalog heterogeneity / completeness check)")
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(f"{FIG_DIR}/03_temporal_distribution.png", dpi=150)
    plt.close(fig)


def plot_depth(df):
    fig, ax = plt.subplots(figsize=(10, 5))
    zones = [z for z in ZONE_ORDER if z != "Unclassified"]
    data = [df.loc[df["zone_code"] == z, "depth_km"].dropna().clip(upper=300) for z in zones]
    bp = ax.boxplot(data, tick_labels=zones, showfliers=False, patch_artist=True)
    for patch, zone in zip(bp["boxes"], zones):
        patch.set_facecolor(ZONE_COLOR[zone])
        patch.set_alpha(0.6)
    ax.set_ylabel("Depth (km, clipped at 300)")
    ax.set_title("Hypocentral depth distribution by source zone")
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(f"{FIG_DIR}/04_depth_histogram.png", dpi=150)
    plt.close(fig)


def build_zone_summary(df, gr_rows):
    rows = []
    for zone in ZONE_ORDER:
        sub = df[df["zone_code"] == zone]
        if len(sub) == 0:
            continue
        rows.append({
            "zone_code": zone,
            "n_events": len(sub),
            "mag_min": sub["mag"].min(),
            "mag_max": sub["mag"].max(),
            "depth_median_km": sub["depth_km"].median(),
            "time_min": sub["time_utc"].min(),
            "time_max": sub["time_utc"].max(),
        })
    summary = pd.DataFrame(rows)
    gr = pd.DataFrame(gr_rows)
    if len(gr):
        summary = summary.merge(gr, on="zone_code", how="left")
    return summary


def main():
    os.makedirs(FIG_DIR, exist_ok=True)
    os.makedirs(REPORT_DIR, exist_ok=True)

    df = pd.read_csv(IN_PATH, parse_dates=["time_utc"])

    print("Plotting event map...")
    plot_event_map(df)

    print("Computing magnitude-frequency distributions and b-values...")
    gr_rows = []
    plot_magnitude_frequency(df, gr_rows)

    print("Plotting temporal distribution...")
    plot_temporal(df)

    print("Plotting depth distribution...")
    plot_depth(df)

    summary = build_zone_summary(df, gr_rows)
    summary.to_csv(f"{REPORT_DIR}/zone_summary.csv", index=False)
    print("\nZone summary:")
    print(summary.to_string(index=False))

    print(f"\nFigures saved to {FIG_DIR}/, summary table saved to {REPORT_DIR}/zone_summary.csv")


if __name__ == "__main__":
    main()


# ---------------------------------------------------------------------------
# Additive extension (does not change any function or the main() run above):
# a depth-colored map reconciled with THIS repo's classify() boundaries
# (scripts/02_source_zones.py), replacing an earlier ad hoc plot that used
# different, inconsistent zone names/boundaries. Instead of drawing the raw
# Z1-Z7 bounding boxes (several overlap - see classify()'s if/elif priority
# order), this shades the actual *resolved* classification on a fine lat/lon
# grid, so what's drawn always matches what classify() would assign.
# ---------------------------------------------------------------------------
import sys as _sys
import os as _os
import matplotlib.colors as mcolors

_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
from _modutil import load_script as _load_script

_zones_mod = _load_script("02_source_zones")
classify = _zones_mod.classify


def plot_depth_by_zone_viridis(df, grid_step=0.15):
    """Southeast Asia seismicity colored by depth (viridis), with a
    background shading of the resolved 7-zone classification (from
    classify() in scripts/02_source_zones.py) and Unclassified events marked
    distinctly. Saves figures/05_depth_by_zone.png."""
    lon_grid = np.arange(88, 143 + grid_step, grid_step)
    lat_grid = np.arange(-13, 30 + grid_step, grid_step)
    LON, LAT = np.meshgrid(lon_grid, lat_grid)

    zone_codes = ZONE_ORDER  # ["Z1", ..., "Z7", "Unclassified"]
    zone_to_int = {z: i for i, z in enumerate(zone_codes)}
    classify_vec = np.vectorize(lambda la, lo: zone_to_int[classify(la, lo)])
    ZGRID = classify_vec(LAT, LON)

    fig, ax = plt.subplots(figsize=(12, 10))

    bg_cmap = mcolors.ListedColormap([ZONE_COLOR[z] for z in zone_codes])
    bg_norm = mcolors.BoundaryNorm(np.arange(-0.5, len(zone_codes) + 0.5, 1), bg_cmap.N)
    ax.contourf(LON, LAT, ZGRID, levels=np.arange(-0.5, len(zone_codes) + 0.5, 1),
                cmap=bg_cmap, norm=bg_norm, alpha=0.15)

    for zone in zone_codes:
        if zone == "Unclassified":
            continue
        mask = ZGRID == zone_to_int[zone]
        if mask.any():
            ax.text(LON[mask].mean(), LAT[mask].mean(), zone, fontsize=13, fontweight="bold",
                    ha="center", va="center",
                    bbox=dict(boxstyle="round", fc="white", ec="black", alpha=0.85))

    classified = df[df["zone_code"] != "Unclassified"]
    unclassified = df[df["zone_code"] == "Unclassified"]

    sc = ax.scatter(classified["lon"], classified["lat"],
                     c=classified["depth_km"].clip(upper=300),
                     cmap="viridis", s=6, alpha=0.65, linewidths=0)
    ax.scatter(unclassified["lon"], unclassified["lat"], marker="x", color="red", s=14,
               alpha=0.6, linewidths=0.9, label=f"Unclassified (n={len(unclassified)})")

    cbar = fig.colorbar(sc, ax=ax, shrink=0.8)
    cbar.set_label("Depth (km, clipped at 300)")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title("SE Asia seismicity by depth, with resolved 7-zone classification\n"
                  "(zone boundaries per scripts/02_source_zones.py classify())")
    ax.set_aspect("equal")
    ax.legend(loc="lower left", fontsize=8, framealpha=0.9)
    ax.grid(alpha=0.15)
    fig.tight_layout()
    fig.savefig(f"{FIG_DIR}/05_depth_by_zone.png", dpi=150)
    plt.close(fig)
    print(f"Saved {FIG_DIR}/05_depth_by_zone.png")


if __name__ == "__main__":
    _df_full = pd.read_csv(IN_PATH, parse_dates=["time_utc"])
    plot_depth_by_zone_viridis(_df_full)
