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
