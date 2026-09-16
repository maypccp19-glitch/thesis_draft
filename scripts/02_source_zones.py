"""
Step 2 - Define 7 tectonic source zones for Southeast Asia and classify
every event in the cleaned catalog into one of them.

These 7 zones are a first-order, literature-informed approximation of the
major tectonic elements that generate seismicity in the study region
(Hall, 2002; Simons et al., 2007; general SE Asia seismotectonic summaries).
They are deliberately coarse (simple linear boundaries / bounding boxes) and
are meant for exploratory data analysis only. Before any spatial-probability
modeling, the boundaries should be refined against an authoritative fault /
source database (e.g. the GEM Global Active Faults Database) or the
zonation from a specific PSHA study the thesis chooses to follow.

Zone list:
  Z1 Sunda Megathrust - Sumatra Segment        (offshore subduction interface)
  Z2 Great Sumatran Fault Zone                 (onshore dextral strike-slip)
  Z3 Sunda Megathrust - Java to Nusa Tenggara  (offshore subduction interface)
  Z4 Andaman-Nicobar-Myanmar Arc & Sagaing FZ  (Indo-Burma subduction + Sagaing)
  Z5 Shan-Thai / Indochina Intraplate Zone     (mainland SE Asia intraplate faults)
  Z6 Philippine Mobile Belt                    (Philippine Trench/Fault, Manila
                                                 Trench, Luzon-Taiwan collision)
  Z7 Sulawesi-Molucca Sea-Banda Arc            (eastern Indonesia collision zone)

Output: data/processed/catalog_with_zones.csv
"""
import numpy as np
import pandas as pd

IN_PATH = "data/processed/catalog_clean.csv"
OUT_PATH = "data/processed/catalog_with_zones.csv"
ZONE_DEF_PATH = "data/processed/source_zone_definitions.csv"

ZONES = [
    {"id": 1, "code": "Z1", "name": "Sunda Megathrust - Sumatra Segment"},
    {"id": 2, "code": "Z2", "name": "Great Sumatran Fault Zone"},
    {"id": 3, "code": "Z3", "name": "Sunda Megathrust - Java to Nusa Tenggara"},
    {"id": 4, "code": "Z4", "name": "Andaman-Nicobar-Myanmar Arc & Sagaing Fault System"},
    {"id": 5, "code": "Z5", "name": "Shan-Thai / Indochina Intraplate Zone"},
    {"id": 6, "code": "Z6", "name": "Philippine Mobile Belt"},
    {"id": 7, "code": "Z7", "name": "Sulawesi - Molucca Sea - Banda Arc"},
]
ZONE_NAME = {z["code"]: z["name"] for z in ZONES}


def sumatra_coast_lon(lat):
    """Approx. west-coast-of-Sumatra longitude as a function of latitude,
    anchored on Banda Aceh (95.3, 5.5) and Bengkulu (102.3, -3.8)."""
    return 99.44 - 0.7527 * lat


def classify(lat, lon):
    # Z4: Andaman Sea / Myanmar Indo-Burma arc / Sagaing fault
    if 92.0 <= lon <= 98.5 and 6.5 <= lat <= 28.5:
        return "Z4"

    # Z1 / Z2: Sumatra megathrust (offshore, incl. outer-rise/forearc
    # seismicity) vs Sumatran Fault (onshore), split by a linearized
    # paleo-coastline running NW-SE. Longitude is capped at 105.5 so the
    # Sunda Strait / Lampung tip does not bleed into western Java (Z3).
    if -7.0 <= lat <= 6.5 and lon <= 105.5:
        coast = sumatra_coast_lon(lat)
        if coast - 10.0 <= lon < coast - 0.3:
            return "Z1"
        if coast - 0.3 <= lon <= coast + 3.0:
            return "Z2"

    # Z5: mainland SE Asia intraplate (Thailand, Laos, Cambodia, Vietnam,
    # southern Yunnan, Malay Peninsula)
    if 97.0 <= lon <= 110.0 and 5.0 <= lat <= 23.5:
        return "Z5"

    # Z3: Java to Nusa Tenggara megathrust
    if 105.0 <= lon <= 126.0 and -11.5 <= lat <= -6.0:
        return "Z3"

    # Z6: Philippine Mobile Belt (incl. Manila Trench, Luzon-Taiwan collision)
    if 118.0 <= lon <= 127.5 and 4.0 <= lat <= 25.5:
        return "Z6"

    # Z7: Sulawesi - Molucca Sea - Banda Arc (eastern Indonesia)
    if 113.0 <= lon <= 135.0 and -11.0 <= lat <= 6.5:
        return "Z7"

    return "Unclassified"


def sanity_check():
    """Spot-check classification against well-known landmark coordinates."""
    checks = [
        ("Bangkok, Thailand", 13.75, 100.50, "Z5"),
        ("Chiang Mai, Thailand", 18.79, 98.98, "Z5"),
        ("Yangon, Myanmar", 16.80, 96.15, "Z4"),
        ("Banda Aceh, Indonesia (N. Sumatra)", 5.55, 95.32, "Z2"),
        ("Offshore W. Sumatra (2004 Sumatra-Andaman rupture area)", 3.30, 95.85, "Z1"),
        ("Jakarta, Indonesia (Java)", -6.21, 106.85, "Z3"),
        ("Manila, Philippines", 14.60, 120.98, "Z6"),
        ("Manado, Indonesia (N. Sulawesi)", 1.49, 124.85, "Z7"),
    ]
    print("\nSanity check against known landmarks:")
    ok = True
    for name, lat, lon, expected in checks:
        got = classify(lat, lon)
        status = "OK" if got == expected else "MISMATCH"
        if got != expected:
            ok = False
        print(f"  {status:8s} {name:55s} -> {got} (expected {expected})")
    return ok


def main():
    df = pd.read_csv(IN_PATH, parse_dates=["time_utc"])
    df["zone_code"] = [classify(la, lo) for la, lo in zip(df["lat"], df["lon"])]
    df["zone_name"] = df["zone_code"].map(ZONE_NAME).fillna("Outside defined source zones")

    ok = sanity_check()
    if not ok:
        print("\nWARNING: one or more landmark sanity checks failed - review zone boundaries.")

    print("\nEvent counts by zone:")
    print(df["zone_code"].value_counts().to_string())

    df.to_csv(OUT_PATH, index=False)
    pd.DataFrame(ZONES).to_csv(ZONE_DEF_PATH, index=False)
    print(f"\nSaved classified catalog -> {OUT_PATH}")
    print(f"Saved zone definitions -> {ZONE_DEF_PATH}")


if __name__ == "__main__":
    main()
