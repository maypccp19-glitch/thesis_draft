# Data cleaning log

TMD: loaded 16656 rows, 16656 with parseable core fields, 16656 kept after validity filters (0 dropped).
USGS: loaded 47151 rows, 47151 earthquake-type rows with parseable core fields, 47151 kept after validity filters (0 dropped).
Dedup: 1270 TMD events matched an USGS event within 60s / 50km and were dropped in favor of the USGS record (kept as the network-of-record for shared events).
Removed 31 exact duplicate rows within the same network.

Final combined catalog: 62506 events -> data/processed/catalog_clean.csv
Time range: 2007-01-04 08:38:00 to 2026-09-11 00:53:48.954000
Magnitude range: 0.6 to 8.6
Spatial extent: lat [-11.01, 37.75], lon [90.10, 141.02]

Events by network:
network
USGS/US          47131
TMD              15374
USGS/OFFICIAL        1
