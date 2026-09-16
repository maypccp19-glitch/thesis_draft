"""
Step 1 - Data cleaning and standardization.

Combines two raw earthquake catalogs into a single, standardized catalog:
  - TMD local earthquake catalog (Thai Meteorological Department), 2007-2026
  - USGS regional catalog for Southeast Asia, 2012-2026

Output: data/processed/catalog_clean.csv
"""
import csv
import numpy as np
import pandas as pd

RAW_TMD = "data/raw/tmd_local_earthquake.csv"
RAW_USGS = "data/raw/usgs_combined_2012_2026.csv"
OUT_PATH = "data/processed/catalog_clean.csv"
LOG_PATH = "reports/01_cleaning_log.md"

log_lines = []


def log(msg):
    print(msg)
    log_lines.append(msg)


def load_tmd():
    # The TMD export is Thai-encoded (Windows-874 / TIS-620), not UTF-8.
    df = pd.read_csv(RAW_TMD, encoding="cp874")
    df.columns = [c.strip() for c in df.columns]
    n_raw = len(df)

    out = pd.DataFrame()
    out["event_id"] = "TMD_" + df["REF. ID"].astype(str).str.strip()
    out["time_utc"] = pd.to_datetime(df["DATE-TIME UTC"], errors="coerce")
    out["lat"] = pd.to_numeric(df["LAT."], errors="coerce")
    out["lon"] = pd.to_numeric(df["LONG."], errors="coerce")
    out["depth_km"] = pd.to_numeric(df["DEPTH."], errors="coerce")
    out["mag"] = pd.to_numeric(df["MAG."], errors="coerce")
    out["mag_type"] = "ML(TMD)"
    out["network"] = "TMD"
    # Source has two REGION columns (Thai description, sometimes English);
    # pandas renames the second to "REGION..1". Prefer Thai text, fall back to English.
    region_th = df["REGION."].astype(str).str.strip()
    region_en = df["REGION..1"].astype(str).str.strip() if "REGION..1" in df.columns else ""
    out["place"] = region_th.where(region_th.ne("") & region_th.ne("nan"), region_en)

    n_before = len(out)
    out = out.dropna(subset=["time_utc", "lat", "lon", "mag"])
    out = out[(out["lat"].between(-90, 90)) & (out["lon"].between(-180, 180))]
    out = out[out["mag"] > 0]
    n_after = len(out)
    log(f"TMD: loaded {n_raw} rows, {n_before} with parseable core fields, "
        f"{n_after} kept after validity filters ({n_before - n_after} dropped).")
    return out


def load_usgs():
    df = pd.read_csv(RAW_USGS)
    n_raw = len(df)
    df = df[df["type"] == "earthquake"]

    out = pd.DataFrame()
    out["event_id"] = "USGS_" + df["id"].astype(str)
    out["time_utc"] = pd.to_datetime(df["time"], errors="coerce", utc=True).dt.tz_localize(None)
    out["lat"] = pd.to_numeric(df["latitude"], errors="coerce")
    out["lon"] = pd.to_numeric(df["longitude"], errors="coerce")
    out["depth_km"] = pd.to_numeric(df["depth"], errors="coerce")
    out["mag"] = pd.to_numeric(df["mag"], errors="coerce")
    out["mag_type"] = df["magType"].astype(str)
    out["network"] = "USGS/" + df["net"].astype(str).str.upper()
    out["place"] = df["place"].astype(str)

    n_before = len(out)
    out = out.dropna(subset=["time_utc", "lat", "lon", "mag"])
    out = out[(out["lat"].between(-90, 90)) & (out["lon"].between(-180, 180))]
    out = out[out["mag"] > 0]
    n_after = len(out)
    log(f"USGS: loaded {n_raw} rows, {n_before} earthquake-type rows with parseable "
        f"core fields, {n_after} kept after validity filters ({n_before - n_after} dropped).")
    return out


def dedupe(tmd, usgs):
    """
    Flag TMD events that are very likely re-reports of a USGS event
    (both networks pick up larger regional shocks): match within 60s and ~50km.
    For matched pairs, keep the USGS record (standardized magType/QC fields)
    and drop the TMD duplicate; unmatched TMD events are the network's unique
    contribution (mostly small local events USGS does not detect).
    """
    tmd = tmd.sort_values("time_utc").reset_index(drop=True)
    usgs_sorted = usgs.sort_values("time_utc").reset_index(drop=True)
    u_times = usgs_sorted["time_utc"].values
    u_lat = usgs_sorted["lat"].values
    u_lon = usgs_sorted["lon"].values

    def haversine_km(lat1, lon1, lat2, lon2):
        r = 6371.0
        p1, p2 = np.radians(lat1), np.radians(lat2)
        dphi = np.radians(lat2 - lat1)
        dlmb = np.radians(lon2 - lon1)
        a = np.sin(dphi / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dlmb / 2) ** 2
        return 2 * r * np.arcsin(np.sqrt(a))

    window = np.timedelta64(60, "s")
    is_dup = np.zeros(len(tmd), dtype=bool)
    lo = 0
    for i, t in enumerate(tmd["time_utc"].values):
        while lo < len(u_times) and u_times[lo] < t - window:
            lo += 1
        j = lo
        while j < len(u_times) and u_times[j] <= t + window:
            d = haversine_km(tmd["lat"].iloc[i], tmd["lon"].iloc[i], u_lat[j], u_lon[j])
            if d <= 50:
                is_dup[i] = True
                break
            j += 1

    n_dup = int(is_dup.sum())
    log(f"Dedup: {n_dup} TMD events matched an USGS event within 60s / 50km and were "
        f"dropped in favor of the USGS record (kept as the network-of-record for shared events).")
    return tmd.loc[~is_dup].reset_index(drop=True)


def main():
    import os
    os.makedirs("data/processed", exist_ok=True)
    os.makedirs("reports", exist_ok=True)

    log("# Data cleaning log\n")
    tmd = load_tmd()
    usgs = load_usgs()

    tmd_unique = dedupe(tmd, usgs)

    catalog = pd.concat([tmd_unique, usgs], ignore_index=True)
    catalog = catalog.sort_values("time_utc").reset_index(drop=True)

    # Remove exact duplicate events (same network double-reporting itself)
    n_before = len(catalog)
    catalog = catalog.drop_duplicates(subset=["network", "time_utc", "lat", "lon", "mag"])
    n_after = len(catalog)
    if n_before != n_after:
        log(f"Removed {n_before - n_after} exact duplicate rows within the same network.")

    catalog.to_csv(OUT_PATH, index=False)
    log(f"\nFinal combined catalog: {len(catalog)} events -> {OUT_PATH}")
    log(f"Time range: {catalog['time_utc'].min()} to {catalog['time_utc'].max()}")
    log(f"Magnitude range: {catalog['mag'].min():.1f} to {catalog['mag'].max():.1f}")
    log(f"Spatial extent: lat [{catalog['lat'].min():.2f}, {catalog['lat'].max():.2f}], "
        f"lon [{catalog['lon'].min():.2f}, {catalog['lon'].max():.2f}]")
    log(f"\nEvents by network:\n{catalog['network'].value_counts().to_string()}")

    with open(LOG_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(log_lines) + "\n")


if __name__ == "__main__":
    main()


# ---------------------------------------------------------------------------
# Additive extension (does not change dedupe() or main() above): a version of
# the dedup step that also records the (time_offset, distance_offset) of every
# matched TMD-USGS pair, so the 60s / 50km cutoffs used by dedupe() can be
# sanity-checked instead of just asserted. See scripts/05_data_quality.py.
# ---------------------------------------------------------------------------
def haversine_km(lat1, lon1, lat2, lon2):
    r = 6371.0
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlmb = np.radians(lon2 - lon1)
    a = np.sin(dphi / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dlmb / 2) ** 2
    return 2 * r * np.arcsin(np.sqrt(a))


def dedupe_with_diagnostics(tmd, usgs, time_window_s=60, dist_window_km=50):
    """
    Same match criterion as dedupe() (a USGS event within `time_window_s`
    seconds and `dist_window_km` km), except it keeps the closest-distance
    match per TMD event (dedupe() takes the first one found) so the reported
    offsets reflect the best candidate pair. Returns the deduped TMD frame
    plus a DataFrame of every matched pair's actual time offset (seconds) and
    distance offset (km) - useful for checking whether the 60s/50km cutoffs
    used by dedupe() are well chosen, or too tight/loose, for this catalog.
    """
    tmd = tmd.sort_values("time_utc").reset_index(drop=True)
    usgs_sorted = usgs.sort_values("time_utc").reset_index(drop=True)
    u_times = usgs_sorted["time_utc"].values
    u_lat = usgs_sorted["lat"].values
    u_lon = usgs_sorted["lon"].values

    window = np.timedelta64(time_window_s, "s")
    is_dup = np.zeros(len(tmd), dtype=bool)
    matches = []  # (tmd_index, time_offset_s, distance_km)
    lo = 0
    for i, t in enumerate(tmd["time_utc"].values):
        while lo < len(u_times) and u_times[lo] < t - window:
            lo += 1
        best = None  # (distance_km, time_offset_s)
        jj = lo
        while jj < len(u_times) and u_times[jj] <= t + window:
            d = haversine_km(tmd["lat"].iloc[i], tmd["lon"].iloc[i], u_lat[jj], u_lon[jj])
            if d <= dist_window_km:
                dt_s = abs((u_times[jj] - t) / np.timedelta64(1, "s"))
                if best is None or d < best[0]:
                    best = (d, dt_s)
            jj += 1
        if best is not None:
            is_dup[i] = True
            matches.append((i, best[1], best[0]))

    match_df = pd.DataFrame(matches, columns=["tmd_index", "time_offset_s", "distance_km"])
    return tmd.loc[~is_dup].reset_index(drop=True), match_df
