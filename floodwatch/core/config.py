"""Shared settings for every stage. Change things here, not inside the stage scripts."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
DATA_RAW.mkdir(parents=True, exist_ok=True)
DATA_PROCESSED.mkdir(parents=True, exist_ok=True)

TIMEZONE = "Asia/Kathmandu"
START_DATE = "2001-01-01"
END_DATE = "2025-12-31"
MONSOON_MONTHS = [6, 7, 8, 9, 10]

# Basin outlet: DHM station 550.05, Bagmati River at Khokana
KHOKANA = {"name": "Bagmati at Khokana", "station_index": "550.05",
           "lat": 27.632222, "lon": 85.293333}

# 3x3 grid over the Kathmandu Valley (Bagmati upstream of Khokana).
# Weather is averaged across these points to get one basin-wide value per day.
BASIN_LATS = [27.62, 27.70, 27.78]
BASIN_LONS = [85.28, 85.36, 85.44]
BASIN_POINTS = [(lat, lon) for lat in BASIN_LATS for lon in BASIN_LONS]

# GloFAS is on a ~5 km grid and may snap to the wrong river near Khokana,
# so we fetch the station point plus 4 nearby offsets. GLOFAS_CHOSEN fixes the selected point.
GLOFAS_POINTS = {
    "q_station": (KHOKANA["lat"], KHOKANA["lon"]),
    "q_north": (KHOKANA["lat"] + 0.05, KHOKANA["lon"]),
    "q_south": (KHOKANA["lat"] - 0.05, KHOKANA["lon"]),
    "q_east": (KHOKANA["lat"], KHOKANA["lon"] + 0.05),
    "q_west": (KHOKANA["lat"], KHOKANA["lon"] - 0.05),
}

# Known event used as a sanity check at every stage
CASE_STUDY_START = "2024-09-26"
CASE_STUDY_END = "2024-10-02"
GLOFAS_CHOSEN = "q_north"  

# Stage 1b sources and files. GFS calibration is deferred; forecast archive is skipped.
STAGE1_FILE = DATA_PROCESSED / "stage1_daily.csv"
STAGE1B_FILE = DATA_PROCESSED / "stage1b_daily.csv"
GFS_TCWV_FILE = DATA_PROCESSED / "gfs_tcwv_daily.csv"
STAGE1B_CACHE = DATA_RAW / "stage1b"
STAGE1B_REPORT = DATA_PROCESSED / "stage1b_checks.csv"
ERA5_URL = "https://archive-api.open-meteo.com/v1/archive"
GFS_HISTORY_URL = "https://historical-forecast-api.open-meteo.com/v1/forecast"
ONI_URL = "https://www.cpc.ncep.noaa.gov/data/indices/oni.ascii.txt"
TCWV_VARIABLE = "total_column_integrated_water_vapour"
ERA5_MODEL = "era5"
GFS_MODEL = "gfs_seamless"
GFS_START = "2021-03-23"
TRAIN_START = "2001-01-01"
TRAIN_END = "2019-12-31"
PERCENTILE_WINDOW_DAYS = 15
PERCENTILE_MIN_SAMPLES = 30
TCWV_CORRELATION_WARN = 0.7
TCWV_EVENT_START = "2024-09-26"
TCWV_EVENT_END = "2024-09-28"
TCWV_EVENT_PERCENTILE = 75.0
ONI_LAG_MONTHS = 2
ONI_SEASONS = ("DJF", "JFM", "FMA", "MAM", "AMJ", "MJJ", "JJA", "JAS", "ASO", "SON", "OND", "NDJ")
API_TIMEOUT = 120
API_PAUSE_SECONDS = 2
HOURS_PER_DAY = 24
CALENDAR_REFERENCE_YEAR = 2000
CALENDAR_CYCLE_DAYS = 366
ONI_RAW_FILE = STAGE1B_CACHE / 'oni.ascii.txt'

ONI_DAILY_FILE = DATA_PROCESSED / "oni_daily.csv"
GFS_CALIBRATION_REPORT = DATA_PROCESSED / "gfs_calibration_checks.csv"
ERA5_REPAIR_YEAR = 2024
ERA5_REPAIR_CACHE = STAGE1B_CACHE / 'era5_tcwv_2024_preserved_valid_hours.json'

# Accepted Open-Meteo ERA5 source gap in API local hourly timestamps.
TCWV_GAP_START = "2024-01-01 05:00:00"
TCWV_GAP_END = "2024-07-01 04:00:00"

# Stage 2: descriptive event analysis, not model features.
STAGE2_DIR = ROOT / 'outputs' / 'stage2'
STAGE2_DAILY = DATA_PROCESSED / 'stage2_daily.csv'
STAGE2_EVENTS = STAGE2_DIR / 'events.csv'
STAGE2_ANNUAL = STAGE2_DIR / 'events_per_year.csv'
STAGE2_CLIMATOLOGY = STAGE2_DIR / 'calendar_climatology.csv'
STAGE2_WINDOWS = STAGE2_DIR / 'event_windows.csv'
STAGE2_COMPOSITES = STAGE2_DIR / 'composites.csv'
STAGE2_PRECURSORS = STAGE2_DIR / 'precursor_timing.csv'
STAGE2_METADATA = STAGE2_DIR / 'metadata.json'
STAGE2_REPORT = STAGE2_DIR / 'checks.csv'
DHM_FILE = DATA_RAW / 'dhm_khokana.csv'
DHM_DATE_COLUMN = 'date'
DHM_LEVEL_COLUMN = 'water_level'
DHM_DANGER_COLUMN = 'danger_level'
FLOOD_QUANTILE = 0.99
EVENT_DRY_DAYS = 7
EVENT_MIN_COUNT_WARN = 15
COMPOSITE_LAG_START = -14
COMPOSITE_LAG_END = 3
NORMAL_QUANTILE = 0.75
PRECURSOR_MIN_EVENTS = 3
CASE_PEAK_DATE = '2024-09-28'
CASE_PEAK_TOLERANCE_DAYS = 1
STAGE2_PLOT_DPI = 140
STAGE2_PLOT_SIZE = (9, 6)
STAGE2_UNITS = {
    'precip_mm': 'mm/day', 'precip_max_point_mm': 'mm/day',
    'sm_0_7': 'm³/m³', 'sm_7_28': 'm³/m³', 'sm_28_100': 'm³/m³',
    'tcwv_mm': 'mm', 'oni': '°C',
    **{key: 'm³/s' for key in GLOFAS_POINTS},
}
MATPLOTLIB_CACHE = STAGE2_DIR / '.matplotlib'

# Stage 2 conditional probabilities and Stage 3 onset prediction.
STAGE2_CONDITIONAL = STAGE2_DIR / 'conditional_probabilities.csv'
STAGE3_DIR = ROOT / 'outputs' / 'stage3'
STAGE3_FEATURES = DATA_PROCESSED / 'stage3_features.csv'
STAGE3_REFERENCE = STAGE3_DIR / 'feature_reference.csv'
STAGE3_REPORT = STAGE3_DIR / 'checks.csv'
STAGE3_METRICS = STAGE3_DIR / 'metrics.csv'
STAGE3_CASE = STAGE3_DIR / 'case_study_predictions.csv'
STAGE3_ONI_ABLATION = STAGE3_DIR / 'oni_ablation.csv'
STAGE3_READY = STAGE3_DIR / 'primary_ready.json'
TARGET_HORIZONS = (1, 3)
TEST_START = '2020-01-01'
TEST_END = END_DATE
CASE_REPORT_START = '2024-09-01'
CASE_REPORT_END = '2024-09-30'
CASE_PLOT_START = '2024-09-20'
CASE_PLOT_END = CASE_STUDY_END
RAINFALL_WINDOWS = (1, 3, 7, 14)
SOIL_COLUMNS = ('sm_0_7', 'sm_7_28', 'sm_28_100')
SOIL_CHANGE_DAYS = 7
DISCHARGE_RISE_WINDOWS = (1, 3)
MONSOON_START_MONTH = 6
MONSOON_START_DAY = 1
CONDITIONAL_TCWV_LEVELS = (90, 95)
CONDITIONAL_RAIN_PERCENTILE = 95
CONDITIONAL_SOIL_PERCENTILE = 75
VALIDATION_FOLDS = (
    ('2009-12-31', '2010-01-01', '2012-12-31'),
    ('2012-12-31', '2013-01-01', '2015-12-31'),
    ('2015-12-31', '2016-01-01', '2019-12-31'),
)
BASELINE_PERCENTILE_THRESHOLDS = (50, 60, 70, 75, 80, 85, 90, 92, 95, 97, 98, 99, 99.5, 100)
XGB_SCORE_THRESHOLDS = (0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)
XGB_PARAMS = dict(n_estimators=150, max_depth=2, learning_rate=0.04,
                  min_child_weight=1, reg_lambda=5, subsample=1,
                  colsample_bytree=1, tree_method='hist', n_jobs=2,
                  random_state=42, objective='binary:logistic', eval_metric='logloss')
MODEL_VARIANTS = ('xgb_core', 'xgb_tcwv', 'xgb_oni_ablation')
PRIMARY_MODEL_VARIANT = 'xgb_tcwv'
STAGE3_PLOT_DPI = 140
STAGE3_PLOT_SIZE = (10, 7)
# Ignore sub-nanounit float round-trip noise when identifying percentile ties.
PERCENTILE_ROUND_DECIMALS = 8

# Read-only Stage 3 diagnostics; saved weights and thresholds are immutable.
STAGE3_DIAGNOSTIC_DIR = STAGE3_DIR / 'diagnostics'
DIAGNOSTIC_START = '2024-09-20'
DIAGNOSTIC_END = '2024-09-27'
DIAGNOSTIC_CONTRIBUTION_DATES = ('2024-09-25', '2024-09-26')
DIAGNOSTIC_TOP_FEATURES = 5

# Stage 3 calendar-prior ablation; the existing accepted artifacts remain intact.
STAGE3_CALENDAR_DIR = STAGE3_DIR / 'calendar_ablation'
STAGE3_CALENDAR_SELECTED = STAGE3_CALENDAR_DIR / 'selected_models.json'
STAGE3_CALENDAR_FEATURES = ('tcwv_pct', 'rain_1d_pct', 'rain_3d_pct', 'rain_7d_pct',
                            'sm_0_7', 'discharge_rise_1d')

# Stage 4: Euclidean analogue windows and pre-onset alignments, no DTW.
STAGE4_DIR = ROOT / 'outputs' / 'stage4'
STAGE4_MATCHES = STAGE4_DIR / 'daily_analogues.csv'
STAGE4_TEMPLATES = STAGE4_DIR / 'templates.csv'
STAGE4_SCALERS = STAGE4_DIR / 'scaling_by_date.csv'
STAGE4_METADATA = STAGE4_DIR / 'metadata.json'
STAGE4_REPORT = STAGE4_DIR / 'checks.csv'
STAGE4_TEMPLATE_LEADS = tuple(range(1, 15))
STAGE4_CASE_HISTORY_END = '2023-12-31'
STAGE4_EXCLUDE_CASE_TEMPLATES = True
STAGE4_WINDOW_CANDIDATES = (14, 7)
STAGE4_MATCH_FEATURES = ('tcwv_pct', 'rain_1d_pct', 'rain_3d_pct', 'rain_7d_pct',
                         'sm_0_7', 'discharge_rise_1d')

# Stage 5 historical replay only.
STAGE5_DIR = ROOT / 'outputs' / 'stage5'
STAGE5_REPLAY_FILE = STAGE5_DIR / 'september_2024_replay.csv'
STAGE5_CASE_TABLE = STAGE5_DIR / 'september_20_28_diagnostics.csv'
STAGE5_REPORT = STAGE5_DIR / 'checks.csv'
