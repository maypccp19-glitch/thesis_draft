# Thesis: Spatial Probability of Aftershocks Given an Identified Mainshock (Southeast Asia, 7 Source Zones)

Data cleaning and exploratory analysis pipeline for a thesis predicting the spatial
probability of aftershocks following an identified mainshock in Southeast Asia,
referenced against 7 proposed tectonic source zones.

**Current status:** data cleaning + EDA complete. Aftershock/mainshock declustering
and the spatial probability model are not yet built — see `reports/02_eda_report.md`
section 6 for the planned next steps.

## Quick start

```bash
python3 -m venv venv && source venv/bin/activate
pip install pandas numpy matplotlib scipy

python3 scripts/01_clean_data.py     # -> data/processed/catalog_clean.csv
python3 scripts/02_source_zones.py   # -> data/processed/catalog_with_zones.csv
python3 scripts/03_eda.py            # -> figures/, reports/zone_summary.csv
```

## Data sources

- `data/raw/tmd_local_earthquake.csv` — Thai Meteorological Department local
  earthquake catalog, 2007-2026 (originally Windows-874/TIS-620 encoded).
- `data/raw/usgs_combined_2012_2026.csv` — USGS regional catalog for Southeast
  Asia, 2012-2026.

See `reports/02_eda_report.md` for full methodology, the 7 source zone
definitions, EDA findings, and known data-quality caveats (mixed magnitude
scales, time-varying completeness, coarse zone boundaries).
