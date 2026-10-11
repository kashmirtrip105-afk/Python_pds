"""
config.py
---------
Central configuration for the Water Quality Monitoring Analytics project.
"""

import os

# ---------------------------------------------------------------------------
# Project paths
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_RAW_DIR = os.path.join(BASE_DIR, "data", "raw")
DATA_PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")
FIGURES_DIR = os.path.join(BASE_DIR, "reports", "figures")
LOGS_DIR = os.path.join(BASE_DIR, "logs")
DB_PATH = os.path.join(BASE_DIR, "data", "water_quality.db")

RAW_CSV_PATH = os.path.join(DATA_RAW_DIR, "water_quality_raw.csv")
PROCESSED_CSV_PATH = os.path.join(DATA_PROCESSED_DIR, "water_quality_processed.csv")
LOG_FILE_PATH = os.path.join(LOGS_DIR, "project.log")

# ---------------------------------------------------------------------------
# Monitoring station settings
# ---------------------------------------------------------------------------
STATION_NAME = "Tapi River, Surat, Gujarat"
STATION_ID = "TAPI_SURAT_001"

# # Indian monitoring configuration
INDIAN_RIVERS = {
    'Tapi':'Surat, Gujarat','Ganga':'Haridwar, Uttarakhand','Yamuna':'Prayagraj, Uttar Pradesh','Narmada':'Jabalpur, Madhya Pradesh','Sabarmati':'Ahmedabad, Gujarat','Godavari':'Nashik, Maharashtra','Krishna':'Vijayawada, Andhra Pradesh','Kaveri':'Mysuru, Karnataka','Brahmaputra':'Guwahati, Assam','Mahanadi':'Cuttack, Odisha'}
# Public reference page used for the web-scraping module (acceptable-range
# guideline values for drinking/surface water quality parameters)
WIKI_STANDARDS_URL = "https://en.wikipedia.org/wiki/Water_quality"

# Live "instantaneous values" water-monitoring API used by
# data_collector.fetch_live_readings(). No public, no-key Indian
# real-time river API exists, so this points at the real, working USGS
# Instantaneous Values service (same parameter-code scheme: 00010=temp,
# 00095=conductivity, 00300=dissolved oxygen, 00400=pH). Indian station
# IDs will simply return no data, which is caught by DataFetchError and
# the pipeline falls back to the synthetic generator below - this still
# demonstrates a genuine live HTTP call + graceful exception handling.
Indian_IV_URL = "https://waterservices.usgs.gov/nwis/iv/"
Indian_PARAM_CODES = "00010,00095,00300,00400"

# Number of synthetic real-time readings to generate for offline demo /
# historical analysis (one reading every 15 minutes by default)
SYNTHETIC_READINGS = 2000
READING_INTERVAL_MINUTES = 15

# ---------------------------------------------------------------------------
# Safe-range thresholds (WHO / EPA drinking-water guideline ballpark values,
# used as a hardcoded fallback if the live web-scrape of guideline values
# fails, and as the alerting thresholds for the real-time dashboard)
# ---------------------------------------------------------------------------
DEFAULT_THRESHOLDS = {
    "ph": (6.5, 8.5),                 # acceptable range
    "turbidity_ntu": (0, 5),          # max acceptable NTU
    "dissolved_oxygen_mgl": (5, 14),  # healthy range for aquatic life
    "conductivity_us_cm": (50, 1500), # typical freshwater range
    "temperature_c": (5, 30),         # typical surface water range
}

RANDOM_SEED = 42

# Real-time simulation settings (used by the CLI live-monitor mode and the
# Streamlit "Live Monitor" page)
LIVE_POLL_SECONDS = 3       # how often a new simulated reading arrives
LIVE_DEFAULT_ITERATIONS = 20  # how many readings the CLI demo streams
