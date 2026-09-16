"""
Step 4 - Mainshock-aftershock clustering diagnostics.

The EDA in scripts/03_eda.py (magnitude-frequency, spatial/temporal/depth
distributions) describes the catalog overall, but never actually looks at
mainshock-aftershock structure. This script adds that evidence, ahead of
building a real declustering + spatial probability model:

  (a) Nearest-neighbor distance analysis (Baiesi & Gerstenberger, 2004;
      Zaliapin & Ben-Zion, 2013) - checks for a visually distinct clustered
      (mainshock-aftershock) population vs. background seismicity.
  (b) Inter-event time histograms vs. the Poisson (exponential) null
      hypothesis - checks for overdispersion consistent with clustering.
  (c) A candidate mainshock catalog (mag >= MAINSHOCK_MIN_MAG per zone).
  (d) Aftershock rate decay curves for the largest mainshocks, with an
      Omori-Utsu p=1 reference line, to visually confirm the decay pattern
      exists in this catalog.

Reuses (does not recompute) the per-zone b-values from reports/zone_summary.csv
(produced by scripts/03_eda.py) and haversine_km from scripts/01_clean_data.py.

Produces:
  figures/06_nn_distance_clustering.png
  figures/07_interevent_time_histograms.png
  figures/08_omori_decay_curves.png
  reports/candidate_mainshocks.csv
"""
import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _modutil import load_script

_clean_mod = load_script("01_clean_data")
haversine_km = _clean_mod.haversine_km

IN_PATH = "data/processed/catalog_with_zones.csv"
ZONE_SUMMARY_PATH = "reports/zone_summary.csv"
FIG_DIR = "figures"
REPORT_DIR = "reports"

ZONE_ORDER = ["Z1", "Z2", "Z3", "Z4", "Z5", "Z6", "Z7"]  # Unclassified excluded from clustering analysis
ZONE_COLOR = {
    "Z1": "#d62728", "Z2": "#ff7f0e", "Z3": "#2ca02c", "Z4": "#1f77b4",
    "Z5": "#9467bd", "Z6": "#8c564b", "Z7": "#17becf",
}

# --- tunable constants ------------------------------------------------------
MAINSHOCK_MIN_MAG = 5.5             # magnitude threshold for candidate mainshocks
AFTERSHOCK_RADIUS_KM = 100.0        # default fixed aftershock search radius
FRACTAL_DIMENSION = 1.6             # epicenter fractal dimension (Zaliapin & Ben-Zion, 2013 typical value)
NN_CANDIDATE_K = 30                 # nearest spatial neighbors considered per event (tractability approximation)
MIN_CANDIDATES_FOR_ZONE_MODEL = 5   # zones below this candidate count flagged as too sparse
TOP_N_OVERALL_MAINSHOCKS = 3
OMORI_MAX_DAYS = 365
OMORI_P_REFERENCE = 1.0


def nearest_neighbor_distances(df_zone, b_value, fractal_dim=FRACTAL_DIMENSION, k=NN_CANDIDATE_K):
    """
    Baiesi & Gerstenberger (2004) / Zaliapin & Ben-Zion (2013) nearest-
    neighbor distance: for each event i, find the preceding event j that
    minimizes eta_ij = T_ij * R_ij, where
        T_ij = t_ij * 10^(-b * M_j / 2)              (rescaled time, t_ij in days)
        R_ij = r_ij^fractal_dim * 10^(-b * M_j / 2)   (rescaled distance, r_ij in km)
    A clustered (aftershock-like) pair has small T and R; background pairs
    are spread out. Only the k nearest spatial neighbors of each event are
    evaluated as parent candidates (eta is dominated by r_ij, so this rarely
    misses the true minimum, and keeps this tractable for 10k+ event zones).

    Returns a DataFrame with columns: T, R, eta, parent_pos (position of the
    chosen parent in the time-sorted, reset-index version of df_zone).
    """
    d = df_zone.sort_values("time_utc").reset_index(drop=True)
    n = len(d)
    if n < 10:
        return pd.DataFrame(columns=["T", "R", "eta", "parent_pos"])

    lat = d["lat"].to_numpy()
    lon = d["lon"].to_numpy()
    mags = d["mag"].to_numpy()
    t_days = (d["time_utc"].to_numpy() - d["time_utc"].to_numpy()[0]) / np.timedelta64(1, "D")

    coords = np.column_stack([lat, lon])
    tree = cKDTree(coords)
    kk = min(k + 1, n)  # +1 since a point is its own nearest neighbor (distance 0)
    _, nbr_idx = tree.query(coords, k=kk)

    T_out = np.full(n, np.nan)
    R_out = np.full(n, np.nan)
    eta_out = np.full(n, np.nan)
    parent_out = np.full(n, -1, dtype=int)

    for i in range(1, n):
        cands = nbr_idx[i]
        cands = cands[(cands != i) & (t_days[cands] < t_days[i])]
        if len(cands) == 0:
            continue
        r_km = np.maximum(haversine_km(lat[i], lon[i], lat[cands], lon[cands]), 1e-3)
        t_ij = t_days[i] - t_days[cands]
        scale = 10 ** (-b_value * mags[cands] / 2)
        T = t_ij * scale
        R = (r_km ** fractal_dim) * scale
        eta = T * R
        best = np.argmin(eta)
        T_out[i], R_out[i], eta_out[i], parent_out[i] = T[best], R[best], eta[best], cands[best]

    mask = ~np.isnan(eta_out)
    return pd.DataFrame({"T": T_out[mask], "R": R_out[mask], "eta": eta_out[mask],
                          "parent_pos": parent_out[mask]})


def plot_nn_clustering(df, b_by_zone):
    zones = [z for z in ZONE_ORDER if not pd.isna(b_by_zone.get(z, np.nan))]
    fig, axes = plt.subplots(2, 4, figsize=(18, 8))
    axes = axes.flatten()
    for i, zone in enumerate(zones):
        ax = axes[i]
        sub = df[df["zone_code"] == zone]
        b = b_by_zone[zone]
        nn = nearest_neighbor_distances(sub, b)
        if len(nn) < 10:
            ax.set_title(f"{zone} (n={len(nn)}, too few pairs)")
            continue
        logT = np.log10(nn["T"].clip(lower=1e-8))
        logR = np.log10(nn["R"].clip(lower=1e-8))
        ax.scatter(logT, logR, s=3, alpha=0.25, color=ZONE_COLOR[zone], linewidths=0)
        ax.set_title(f"{zone} (b={b:.2f}, n pairs={len(nn)})")
        ax.set_xlabel("log10 T (rescaled time)")
        ax.set_ylabel("log10 R (rescaled distance)")
        ax.grid(alpha=0.3)
    for j in range(len(zones), len(axes)):
        axes[j].axis("off")
    fig.suptitle("Nearest-neighbor distance analysis (Baiesi & Gerstenberger 2004 / Zaliapin & Ben-Zion 2013)\n"
                 "a distinct low-T, low-R population indicates mainshock-aftershock clustering")
    fig.tight_layout()
    fig.savefig(f"{FIG_DIR}/06_nn_distance_clustering.png", dpi=150)
    plt.close(fig)


def plot_interevent_time_histograms(df):
    zones = [z for z in ZONE_ORDER if (df["zone_code"] == z).sum() >= 10]
    fig, axes = plt.subplots(2, 4, figsize=(18, 8))
    axes = axes.flatten()
    for i, zone in enumerate(zones):
        ax = axes[i]
        sub = df[df["zone_code"] == zone].sort_values("time_utc")
        t = sub["time_utc"].to_numpy()
        dt_days = np.diff(t) / np.timedelta64(1, "D")
        dt_days = dt_days[dt_days > 0]
        if len(dt_days) < 10:
            ax.set_title(f"{zone} (n={len(dt_days)}, too few)")
            continue
        rate = 1.0 / dt_days.mean()  # events/day, Poisson MLE
        bins = np.logspace(np.log10(max(dt_days.min(), 1e-4)), np.log10(dt_days.max()), 40)
        counts, edges = np.histogram(dt_days, bins=bins, density=True)
        positive = counts[counts > 0]
        ax.stairs(counts, edges, fill=True, alpha=0.6, color=ZONE_COLOR[zone], label="observed")

        # Exponential reference curve, computed in log-space and masked before
        # it underflows to ~0 (float underflow on a log-y axis otherwise
        # stretches the axis across hundreds of decades and hides the data).
        xs = np.logspace(np.log10(bins[0]), np.log10(bins[-1]), 200)
        log10_model = np.log10(rate) - rate * xs / np.log(10)
        floor = np.log10(positive.min()) - 2 if len(positive) else -10
        model_mask = log10_model > floor
        ax.plot(xs[model_mask], 10 ** log10_model[model_mask], "k--", lw=1.5,
                label=f"Poisson fit (rate={rate:.2f}/day)")

        ax.set_xscale("log")
        ax.set_yscale("log")
        if len(positive):
            ax.set_ylim(bottom=positive.min() * 0.5, top=max(positive.max(), 10 ** log10_model[model_mask].max() if model_mask.any() else positive.max()) * 2)
        ax.set_xlabel("Inter-event time (days)")
        ax.set_ylabel("Density")
        ax.set_title(zone)
        ax.legend(fontsize=7)
        ax.grid(alpha=0.3)
    for j in range(len(zones), len(axes)):
        axes[j].axis("off")
    fig.suptitle("Inter-event time distributions vs. Poisson (exponential) null hypothesis\n"
                 "excess of short inter-event times vs. the fit indicates clustering/overdispersion")
    fig.tight_layout()
    fig.savefig(f"{FIG_DIR}/07_interevent_time_histograms.png", dpi=150)
    plt.close(fig)


def build_candidate_mainshocks(df, min_mag=MAINSHOCK_MIN_MAG):
    cand = df[df["mag"] >= min_mag].copy()
    cand = cand.sort_values(["zone_code", "mag"], ascending=[True, False])
    cols = ["event_id", "zone_code", "zone_name", "time_utc", "lat", "lon",
            "depth_km", "mag", "mag_type", "network", "place"]
    cand = cand[cols]
    cand.to_csv(f"{REPORT_DIR}/candidate_mainshocks.csv", index=False)

    counts = cand["zone_code"].value_counts()
    print(f"\nCandidate mainshocks (mag >= {min_mag}): {len(cand)} total")
    print(counts.reindex(ZONE_ORDER, fill_value=0).to_string())
    for zone in ZONE_ORDER:
        n = int(counts.get(zone, 0))
        if n < MIN_CANDIDATES_FOR_ZONE_MODEL:
            print(f"  WARNING: {zone} has only {n} candidate mainshock(s) "
                  f"(< {MIN_CANDIDATES_FOR_ZONE_MODEL}) - too sparse to support a per-zone aftershock model.")
    return cand


def wells_coppersmith_rupture_length_km(mag):
    """Subsurface rupture length (RLD) vs. moment magnitude, all-fault-type
    regression from Wells & Coppersmith (1994), Table 2A:
        log10(RLD [km]) = -2.44 + 0.59 * M
    Used only as a rough distance-scaling heuristic for sizing the
    aftershock search radius of large mainshocks - not a precise rupture
    estimate (magnitude type and style-of-faulting mismatches are ignored)."""
    return 10 ** (-2.44 + 0.59 * mag)


def aftershock_search_radius_km(mag, fixed_radius=AFTERSHOCK_RADIUS_KM):
    """Larger of the fixed default radius and 5x the WC1994 rupture length,
    so the window scales up for large mainshocks instead of staying fixed
    at 100km regardless of magnitude."""
    return max(fixed_radius, 5.0 * wells_coppersmith_rupture_length_km(mag))


def compute_aftershock_rate(df, mainshock, radius_km, max_days=OMORI_MAX_DAYS):
    t0 = mainshock["time_utc"]
    dt_days_all = (df["time_utc"] - t0).dt.total_seconds() / 86400.0
    dist_km = haversine_km(mainshock["lat"], mainshock["lon"], df["lat"].to_numpy(), df["lon"].to_numpy())
    mask = ((dt_days_all > 0) & (dt_days_all <= max_days) & (dist_km <= radius_km) &
            (df["event_id"] != mainshock["event_id"]))
    dts = dt_days_all[mask].to_numpy()
    if len(dts) < 5:
        return None, None, len(dts)

    bins = np.logspace(np.log10(max(dts.min(), 1e-3)), np.log10(max_days), 25)
    counts, edges = np.histogram(dts, bins=bins)
    widths = np.diff(edges)
    centers = np.sqrt(edges[:-1] * edges[1:])  # geometric mean for log-spaced bins
    rate = counts / widths
    valid = counts > 0
    return centers[valid], rate[valid], len(dts)


def select_mainshocks_to_plot(candidates):
    """2-3 largest mainshocks overall, plus the largest in each zone that has
    enough candidates to support a per-zone model (skipping zones already
    covered by the overall top-N)."""
    selections = []
    overall_top = candidates.sort_values("mag", ascending=False).head(TOP_N_OVERALL_MAINSHOCKS)
    for _, row in overall_top.iterrows():
        selections.append((f"Overall #{len(selections) + 1}", row))

    already = {r["event_id"] for _, r in selections}
    counts = candidates["zone_code"].value_counts()
    for zone in ZONE_ORDER:
        if counts.get(zone, 0) < MIN_CANDIDATES_FOR_ZONE_MODEL:
            continue
        top_row = candidates[candidates["zone_code"] == zone].sort_values("mag", ascending=False).iloc[0]
        if top_row["event_id"] in already:
            continue
        selections.append((f"{zone} largest", top_row))
        already.add(top_row["event_id"])
    return selections


def plot_omori_curves(df, mainshocks):
    n = len(mainshocks)
    if n == 0:
        print("No mainshocks selected for Omori decay plots - skipping.")
        return
    ncols = min(4, n)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.5 * ncols, 4 * nrows), squeeze=False)
    axes = axes.flatten()
    for i, (label, row) in enumerate(mainshocks):
        ax = axes[i]
        radius = aftershock_search_radius_km(row["mag"])
        centers, rate, n_aft = compute_aftershock_rate(df, row, radius)
        if centers is None:
            ax.set_title(f"{label}: {row['event_id']}\n(n aftershocks={n_aft}, too few)")
            continue
        ax.loglog(centers, rate, "o", ms=4, color="#1f77b4", label="observed rate")
        c_offset = 0.1
        k_anchor = rate[0] * (centers[0] + c_offset)
        t_ref = np.logspace(np.log10(centers.min()), np.log10(centers.max()), 100)
        ax.loglog(t_ref, k_anchor / (t_ref + c_offset), "k--", lw=1.5,
                  label=f"Omori p={OMORI_P_REFERENCE:.0f} reference")
        ax.set_xlabel("Days since mainshock")
        ax.set_ylabel("Rate (events/day)")
        ax.set_title(f"{label}: {row['event_id']}\nM{row['mag']:.1f}, {row['zone_code']}, "
                     f"r={radius:.0f}km, n={n_aft}")
        ax.legend(fontsize=7)
        ax.grid(alpha=0.3, which="both")
    for j in range(len(mainshocks), len(axes)):
        axes[j].axis("off")
    fig.suptitle("Aftershock rate decay vs. days since mainshock (Omori-Utsu p=1 reference line)")
    fig.tight_layout()
    fig.savefig(f"{FIG_DIR}/08_omori_decay_curves.png", dpi=150)
    plt.close(fig)


def main():
    os.makedirs(FIG_DIR, exist_ok=True)
    os.makedirs(REPORT_DIR, exist_ok=True)

    df = pd.read_csv(IN_PATH, parse_dates=["time_utc"])
    df_classified = df[df["zone_code"] != "Unclassified"].copy()

    zone_summary = pd.read_csv(ZONE_SUMMARY_PATH)
    b_by_zone = zone_summary.set_index("zone_code")["b_value"].to_dict()
    print("Reusing per-zone b-values from reports/zone_summary.csv:")
    for z in ZONE_ORDER:
        print(f"  {z}: b={b_by_zone.get(z, float('nan')):.3f}")

    print("\nComputing nearest-neighbor distance clustering diagnostics...")
    plot_nn_clustering(df_classified, b_by_zone)

    print("Plotting inter-event time histograms...")
    plot_interevent_time_histograms(df_classified)

    print("\nBuilding candidate mainshock catalog...")
    candidates = build_candidate_mainshocks(df_classified)

    print("\nPlotting Omori decay curves for largest mainshocks...")
    mainshocks = select_mainshocks_to_plot(candidates)
    plot_omori_curves(df_classified, mainshocks)

    print(f"\nDone. Figures: {FIG_DIR}/06_nn_distance_clustering.png, "
          f"{FIG_DIR}/07_interevent_time_histograms.png, {FIG_DIR}/08_omori_decay_curves.png")
    print(f"Table: {REPORT_DIR}/candidate_mainshocks.csv")


if __name__ == "__main__":
    main()
