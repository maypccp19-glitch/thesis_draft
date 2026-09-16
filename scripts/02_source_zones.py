"""
Step 2 - Define 7 tectonic plates/microplates relevant to Southeast Asia and
classify every event in the cleaned catalog into one of them.

Replaces an earlier version of this script that zoned by fault segment
(Sumatra megathrust, Sumatran Fault, etc.) instead of by tectonic plate.

These 7 plates are the standard set used in regional geodynamics / GPS block
models of Southeast Asia (e.g. Bird, 2003 "An updated digital model of plate
boundaries"; Simons et al., 2007 "A decade of GPS in Southeast Asia"). As
before, the boundaries used here are a first-order, deliberately coarse
approximation (simple linear boundaries / bounding boxes) meant for
exploratory data analysis - not a substitute for an authoritative plate
boundary dataset (e.g. Bird's PB2002 model) before spatial-probability
modeling. Seismicity is generated at plate *boundaries*; the boxes below
approximate where each boundary/subduction zone sits, not the full interior
extent of each plate.

Zone list:
  Z1 Burma Plate               (Andaman Sea, Myanmar, Sagaing Fault, Indo-Burma Ranges)
  Z2 Indo-Australian Plate     (subducting plate: offshore Sumatra, Java-Bali trench)
  Z3 Sunda Plate                (overriding plate: Indochina, Malay Peninsula, Borneo,
                                 onshore Sumatra/Java-Nusa Tenggara)
  Z4 Philippine Sea Plate      (Philippine Mobile Belt, Manila Trench, Philippine
                                 Trench, Luzon-Taiwan collision)
  Z5 Molucca Sea Plate         (Sulawesi, Halmahera, Molucca Sea collision)
  Z6 Banda Sea / Timor Microplate  (eastern Indonesia collision zone)
  Z7 Pacific Plate             (western edge: Yap, Palau, Caroline Ridge)

Output: data/processed/catalog_with_zones.csv
"""
import numpy as np
import pandas as pd

IN_PATH = "data/processed/catalog_clean.csv"
OUT_PATH = "data/processed/catalog_with_zones.csv"
ZONE_DEF_PATH = "data/processed/source_zone_definitions.csv"

ZONES = [
    {"id": 1, "code": "Z1", "name": "Burma Plate"},
    {"id": 2, "code": "Z2", "name": "Indo-Australian Plate"},
    {"id": 3, "code": "Z3", "name": "Sunda Plate"},
    {"id": 4, "code": "Z4", "name": "Philippine Sea Plate"},
    {"id": 5, "code": "Z5", "name": "Molucca Sea Plate"},
    {"id": 6, "code": "Z6", "name": "Banda Sea / Timor Microplate"},
    {"id": 7, "code": "Z7", "name": "Pacific Plate"},
]
ZONE_NAME = {z["code"]: z["name"] for z in ZONES}


def sumatra_coast_lon(lat):
    """Approx. west-coast-of-Sumatra longitude as a function of latitude,
    anchored on Banda Aceh (95.3, 5.5) and Bengkulu (102.3, -3.8). Used to
    split the Sumatra segment into the subducting Indo-Australian Plate
    (offshore, west of the coast) and the overriding Sunda Plate (onshore)."""
    return 99.44 - 0.7527 * lat


def classify(lat, lon):
    # Z1: Burma Plate (Andaman Sea, Myanmar, Sagaing Fault, Indo-Burma Ranges)
    if 92.0 <= lon <= 98.5 and 6.5 <= lat <= 28.5:
        return "Z1"

    # Z2 / Z3: Sumatra split between the subducting Indo-Australian Plate
    # (offshore, incl. outer-rise/forearc seismicity) and the overriding
    # Sunda Plate (onshore), by a linearized paleo-coastline running NW-SE.
    # Longitude capped at 105.5 so the Sunda Strait / Lampung tip does not
    # bleed into western Java.
    if -7.0 <= lat <= 6.5 and lon <= 105.5:
        coast = sumatra_coast_lon(lat)
        if coast - 10.0 <= lon < coast - 0.3:
            return "Z2"  # Indo-Australian Plate (offshore)
        if coast - 0.3 <= lon <= coast + 3.0:
            return "Z3"  # Sunda Plate (onshore)

    # Z3: Sunda Plate - mainland Indochina & Malay Peninsula
    if 97.0 <= lon <= 110.0 and 5.0 <= lat <= 23.5:
        return "Z3"

    # Z3: Sunda Plate - Borneo
    if 108.0 <= lon <= 119.0 and -5.0 <= lat <= 7.5:
        return "Z3"

    # Z3: Sunda Plate - onshore Java to Nusa Tenggara (north of the coast)
    if 105.0 <= lon <= 118.0 and -8.5 <= lat <= -6.0:
        return "Z3"

    # Z2: Indo-Australian Plate - offshore Java-Bali trench (south of the coast)
    if 105.0 <= lon <= 118.0 and -11.5 <= lat <= -8.5:
        return "Z2"

    # Z4: Philippine Sea Plate (Philippine Mobile Belt, Manila Trench,
    # Philippine Trench, Luzon-Taiwan collision)
    if 118.0 <= lon <= 127.5 and 4.0 <= lat <= 25.5:
        return "Z4"

    # Z5: Molucca Sea Plate (Sulawesi, Halmahera, Molucca Sea collision)
    if 118.0 <= lon <= 129.0 and -3.5 <= lat <= 6.5:
        return "Z5"

    # Z6: Banda Sea / Timor Microplate (eastern Indonesia collision zone,
    # incl. Flores, Alor, Timor, Seram)
    if 118.0 <= lon <= 135.0 and -11.5 <= lat <= -3.5:
        return "Z6"

    # Z7: Pacific Plate (western edge - Yap, Palau, Caroline Ridge)
    if 127.0 <= lon <= 141.0 and -5.0 <= lat <= 15.0:
        return "Z7"

    return "Unclassified"


def sanity_check():
    """Spot-check classification against well-known landmark coordinates."""
    checks = [
        ("Bangkok, Thailand", 13.75, 100.50, "Z3"),
        ("Chiang Mai, Thailand", 18.79, 98.98, "Z3"),
        ("Yangon, Myanmar", 16.80, 96.15, "Z1"),
        ("Banda Aceh, Indonesia (N. Sumatra, onshore)", 5.55, 95.32, "Z3"),
        ("Offshore W. Sumatra (2004 Sumatra-Andaman rupture area)", 3.30, 95.85, "Z2"),
        ("Jakarta, Indonesia (Java, onshore)", -6.21, 106.85, "Z3"),
        ("Offshore south Java trench", -10.00, 110.00, "Z2"),
        ("Manila, Philippines", 14.60, 120.98, "Z4"),
        ("Manado, Indonesia (N. Sulawesi)", 1.49, 124.85, "Z5"),
        ("Dili, Timor-Leste", -8.55, 125.57, "Z6"),
        ("Koror, Palau", 7.34, 134.48, "Z7"),
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
