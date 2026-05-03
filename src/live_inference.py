"""
F1 Live Inference Module
DS591 - Big Data Workloads, Boston University
----------------------------------------------
Called by live_casting.py Thread 3 every INFERENCE_INTERVAL seconds (once per lap).

Pipeline per inference cycle:
  1. Load all 20 drivers' telemetry from session.car_data (last 30 seconds)
  2. Resample each driver to exactly 10 Hz → 300 timesteps
  3. For each target driver, find driver_ahead and driver_behind via relative_distance
  4. Join lap-level features (compound, tyre_life, lap_number) from session.laps
  5. Join weather features from session.weather_data
  6. OHE encode compound, gear, team, track_status to match training feature_cols exactly
  7. Build (300, 129) tensor per driver → run through channel_{DRIVER}.pt
  8. Stack all embeddings → run through random_forest.joblib
  9. Return {driver_code: {predicted_position, prediction_confidence}}

Thread safety:
  _latest_predictions is protected by _lock.
  Thread 2 (poll, 5s) reads via get_predictions().
  Thread 3 (inference, 90s) writes via set_predictions().

All torch/joblib imports are deferred inside functions — this file is safe
to import even if torch is not installed (INFERENCE_AVAILABLE will be False
in live_casting.py and inference simply won't run).
"""

import logging
import os
import threading
from datetime import datetime

import numpy as np
import pandas as pd

log = logging.getLogger("F1Pipeline")

# ── Model paths ───────────────────────────────────────────────────────────────
MODEL_DIR = os.path.join(os.path.dirname(__file__), "model_weights")

# ── ADLS paths (radio_live.parquet written by radio_data.run_radio_live) ──────
SILVER_CONTAINER  = os.environ.get("SILVER_CONTAINER", "silver")
STORAGE_ACCT_NAME = os.environ.get("STORAGE_ACCOUNT_NAME", "")
STORAGE_ACCT_KEY  = os.environ.get("STORAGE_ACCOUNT_KEY", "")
RADIO_LIVE_PATH   = f"abfs://{SILVER_CONTAINER}@{STORAGE_ACCT_NAME}.dfs.core.windows.net/radio_live.parquet"

# ── Social media JSON — loaded once at module import ─────────────────────────
# Miami GP is May 2026. Most recent data is April 2026 ("4").
# Structure: {year: {month: {driver_code: score}}}
_SOCIAL_JSON_PATH = os.path.join(
    os.path.dirname(__file__), "..", "silver", "social_media_silver.json"
)
_SOCIAL_YEAR  = "2026"
_SOCIAL_MONTH = "4"   # April — most recent before Miami (May)

def _load_social_scores() -> dict[str, float | None]:
    """
    Load driver social life scores from silver/social_media_silver.json.
    Returns {driver_code: float} for all drivers present in April 2026.
    Returns empty dict (all drivers will get None → model fills 5.0) on error.

    This is called once at module load and cached in _SOCIAL_SCORES.
    The file is static — it never changes during a race session.
    """
    try:
        with open(_SOCIAL_JSON_PATH, "r") as f:
            data = json.load(f)
        scores = data.get(_SOCIAL_YEAR, {}).get(_SOCIAL_MONTH, {})
        log.info(f"Loaded social scores for {len(scores)} drivers "
                 f"({_SOCIAL_YEAR}-{_SOCIAL_MONTH}): {scores}")
        return scores
    except FileNotFoundError:
        log.warning(f"social_media_silver.json not found at {_SOCIAL_JSON_PATH} — "
                    f"all social scores will be None (model default 5.0).")
        return {}
    except Exception as e:
        log.error(f"Error loading social scores: {e}", exc_info=True)
        return {}

# Module-level cache — loaded once, never reloaded (file is static)
import json as _json_module   # alias to avoid shadowing
import json
_SOCIAL_SCORES: dict[str, float | None] = _load_social_scores()

# ── Radio feature columns — exact mapping from radio_data.classify_radio
#    output schema to the 13 radio_cols the model was trained on ───────────────
#
# radio_data.classify_radio() returns nested dicts. After pd.json_normalize
# in run_radio_silver/run_radio_live, column names become dot-separated:
#   confidence                              → radio_confidence
#   car_issue_signal.has_issue              → radio_car_issue_signal.has_issue
#   strategy_signal.pit_related             → radio_strategy_signal.pit_related
#   strategy_signal.tire_related            → radio_strategy_signal.tire_related
#   strategy_signal.fuel_saving             → radio_strategy_signal.fuel_saving
#   strategy_signal.pace_change             → radio_strategy_signal.pace_change
#   strategy_signal.weather_related         → radio_strategy_signal.weather_related
#   strategy_signal.safety_related          → radio_strategy_signal.safety_related
#   racecraft_signal.traffic_mentioned      → radio_racecraft_signal.traffic_mentioned
#   racecraft_signal.overtake_mentioned     → radio_racecraft_signal.overtake_mentioned
#   racecraft_signal.defend_mentioned       → radio_racecraft_signal.defend_mentioned
#   racecraft_signal.drs_mentioned          → radio_racecraft_signal.drs_mentioned
#   racecraft_signal.gap_management_mentioned → radio_racecraft_signal.gap_management_mentioned
#
# RADIO_COL_MAP: {parquet_column → model_radio_col}
RADIO_COL_MAP = {
    "confidence":                                  "radio_confidence",
    "car_issue_signal.has_issue":                  "radio_car_issue_signal.has_issue",
    "strategy_signal.pit_related":                 "radio_strategy_signal.pit_related",
    "strategy_signal.tire_related":                "radio_strategy_signal.tire_related",
    "strategy_signal.fuel_saving":                 "radio_strategy_signal.fuel_saving",
    "strategy_signal.pace_change":                 "radio_strategy_signal.pace_change",
    "strategy_signal.weather_related":             "radio_strategy_signal.weather_related",
    "strategy_signal.safety_related":              "radio_strategy_signal.safety_related",
    "racecraft_signal.traffic_mentioned":          "radio_racecraft_signal.traffic_mentioned",
    "racecraft_signal.overtake_mentioned":         "radio_racecraft_signal.overtake_mentioned",
    "racecraft_signal.defend_mentioned":           "radio_racecraft_signal.defend_mentioned",
    "racecraft_signal.drs_mentioned":              "radio_racecraft_signal.drs_mentioned",
    "racecraft_signal.gap_management_mentioned":   "radio_racecraft_signal.gap_management_mentioned",
}

# Ordered list of the 13 radio_cols exactly as saved in the checkpoint
RADIO_COLS = [
    "radio_confidence",
    "radio_car_issue_signal.has_issue",
    "radio_strategy_signal.pit_related",
    "radio_strategy_signal.tire_related",
    "radio_strategy_signal.fuel_saving",
    "radio_strategy_signal.pace_change",
    "radio_strategy_signal.weather_related",
    "radio_strategy_signal.safety_related",
    "radio_racecraft_signal.traffic_mentioned",
    "radio_racecraft_signal.overtake_mentioned",
    "radio_racecraft_signal.defend_mentioned",
    "radio_racecraft_signal.drs_mentioned",
    "radio_racecraft_signal.gap_management_mentioned",
]

# ── Inference constants ───────────────────────────────────────────────────────
SEQUENCE_LENGTH = 300       # timesteps the model expects
WINDOW_SECONDS  = 30        # seconds of history to use
TARGET_HZ       = 10        # resample target: 10 Hz → 300 rows over 30s

# ── Exact feature_cols the model was trained on (from checkpoint inspection) ──
FEATURE_COLS = [
    'air_temp', 'distance_ahead_target', 'distance_target_behind',
    'driver_ahead_brake', 'driver_ahead_compound_HARD',
    'driver_ahead_compound_INTERMEDIATE', 'driver_ahead_compound_MEDIUM',
    'driver_ahead_compound_SOFT', 'driver_ahead_compound_Unknown',
    'driver_ahead_compound_WET', 'driver_ahead_compound_nan',
    'driver_ahead_gear_0.0', 'driver_ahead_gear_1.0', 'driver_ahead_gear_2.0',
    'driver_ahead_gear_3.0', 'driver_ahead_gear_4.0', 'driver_ahead_gear_5.0',
    'driver_ahead_gear_6.0', 'driver_ahead_gear_7.0', 'driver_ahead_gear_8.0',
    'driver_ahead_lap_number', 'driver_ahead_number', 'driver_ahead_pos',
    'driver_ahead_speed', 'driver_ahead_team_Alpine',
    'driver_ahead_team_Aston Martin', 'driver_ahead_team_Ferrari',
    'driver_ahead_team_Haas F1 Team', 'driver_ahead_team_Kick Sauber',
    'driver_ahead_team_McLaren', 'driver_ahead_team_Mercedes',
    'driver_ahead_team_RB', 'driver_ahead_team_Racing Bulls',
    'driver_ahead_team_Red Bull Racing', 'driver_ahead_team_Unknown',
    'driver_ahead_team_Williams', 'driver_ahead_tyre_life',
    'driver_behind_brake', 'driver_behind_compound_HARD',
    'driver_behind_compound_INTERMEDIATE', 'driver_behind_compound_MEDIUM',
    'driver_behind_compound_SOFT', 'driver_behind_compound_Unknown',
    'driver_behind_compound_WET', 'driver_behind_compound_nan',
    'driver_behind_gear_0.0', 'driver_behind_gear_1.0',
    'driver_behind_gear_2.0', 'driver_behind_gear_3.0',
    'driver_behind_gear_4.0', 'driver_behind_gear_5.0',
    'driver_behind_gear_6.0', 'driver_behind_gear_7.0',
    'driver_behind_gear_8.0', 'driver_behind_lap_number',
    'driver_behind_number', 'driver_behind_pos', 'driver_behind_speed',
    'driver_behind_team_Alpine', 'driver_behind_team_Aston Martin',
    'driver_behind_team_Ferrari', 'driver_behind_team_Haas F1 Team',
    'driver_behind_team_Kick Sauber', 'driver_behind_team_McLaren',
    'driver_behind_team_Mercedes', 'driver_behind_team_RB',
    'driver_behind_team_Racing Bulls', 'driver_behind_team_Red Bull Racing',
    'driver_behind_team_Unknown', 'driver_behind_team_Williams',
    'driver_behind_tyre_life', 'has_driver_ahead', 'has_driver_behind',
    'humidity', 'pressure', 'radio_event_count', 'rainfall',
    'target_brake', 'target_compound_HARD', 'target_compound_INTERMEDIATE',
    'target_compound_MEDIUM', 'target_compound_SOFT', 'target_compound_nan',
    'target_driver_number', 'target_gear_0.0', 'target_gear_1.0',
    'target_gear_2.0', 'target_gear_3.0', 'target_gear_4.0',
    'target_gear_5.0', 'target_gear_6.0', 'target_gear_7.0',
    'target_gear_8.0', 'target_lap_number', 'target_relative_distance',
    'target_speed', 'target_team_Red Bull Racing', 'target_tyre_life',
    'target_x', 'target_y', 'target_z', 'track_status_1', 'track_status_12',
    'track_status_124', 'track_status_125', 'track_status_1254',
    'track_status_126', 'track_status_1264', 'track_status_14',
    'track_status_16', 'track_status_167', 'track_status_2',
    'track_status_21', 'track_status_216', 'track_status_24',
    'track_status_26', 'track_status_4', 'track_status_41',
    'track_status_412', 'track_status_6', 'track_status_64',
    'track_status_67', 'track_status_671', 'track_status_6712',
    'track_status_71', 'track_temp', 'wind_direction_cos',
    'wind_direction_sin', 'wind_speed',
]

# OHE categories — must match training exactly
COMPOUND_CATS = ['HARD', 'INTERMEDIATE', 'MEDIUM', 'SOFT', 'Unknown', 'WET', 'nan']
GEAR_CATS     = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
TEAM_CATS     = [
    'Alpine', 'Aston Martin', 'Ferrari', 'Haas F1 Team', 'Kick Sauber',
    'McLaren', 'Mercedes', 'RB', 'Racing Bulls', 'Red Bull Racing',
    'Unknown', 'Williams',
]
# All track_status values seen across training data
TRACK_STATUS_CATS = [
    '1', '12', '124', '125', '1254', '126', '1264', '14', '16', '167',
    '2', '21', '216', '24', '26', '4', '41', '412', '6', '64', '67',
    '671', '6712', '71',
]

# ── Shared predictions store (Thread 2 reads, Thread 3 writes) ────────────────
_lock               = threading.Lock()
_latest_predictions: dict = {}   # {driver_code: {predicted_position, prediction_confidence}}


def get_predictions() -> dict:
    with _lock:
        return dict(_latest_predictions)


def set_predictions(preds: dict):
    with _lock:
        _latest_predictions.clear()
        _latest_predictions.update(preds)


# ── Model cache (loaded once at first inference call) ─────────────────────────
_channel_cache: dict  = {}   # driver_code → DriverChannel model
_rf_cache             = None
_device               = None


def _get_device():
    global _device
    if _device is None:
        import torch
        _device = torch.device(
            "mps"  if torch.backends.mps.is_available()  else
            "cuda" if torch.cuda.is_available()           else
            "cpu"
        )
        log.info(f"Inference device: {_device}")
    return _device


def _load_channel(driver_code: str):
    """Load and cache DriverChannel for one driver. Returns model or None."""
    if driver_code in _channel_cache:
        return _channel_cache[driver_code]

    import torch
    from src.model import DriverChannel, EMBEDDING_DIM

    path = os.path.join(MODEL_DIR, f"channel_{driver_code}.pt")
    if not os.path.exists(path):
        log.warning(f"No checkpoint for {driver_code} at {path} — skipping.")
        _channel_cache[driver_code] = None
        return None

    ckpt   = torch.load(path, map_location=_get_device(), weights_only=False)
    model  = DriverChannel(
        input_size=ckpt["input_size"],
        embedding_dim=EMBEDDING_DIM,
        radio_dim=ckpt["radio_dim"],
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(_get_device())
    model.eval()
    _channel_cache[driver_code] = model
    log.info(f"Loaded channel_{driver_code}.pt")
    return model


def _load_rf():
    """Load and cache the RandomForest. Returns model or None."""
    global _rf_cache
    if _rf_cache is not None:
        return _rf_cache

    import joblib
    path = os.path.join(MODEL_DIR, "random_forest.joblib")
    if not os.path.exists(path):
        log.warning(f"random_forest.joblib not found at {path}")
        return None

    _rf_cache = joblib.load(path)
    log.info("Loaded random_forest.joblib")
    return _rf_cache


# ═════════════════════════════════════════════════════════════════════════════
# STEP 1 — Build all-driver telemetry snapshot from session.car_data
# ═════════════════════════════════════════════════════════════════════════════
def _build_all_drivers_telemetry(session) -> dict[str, pd.DataFrame]:
    """
    For every driver, extract the last WINDOW_SECONDS of raw telemetry
    from session.car_data, resample to TARGET_HZ, and return as a dict:
        {driver_number_str: DataFrame with columns matching bronze telemetry}

    Resampling strategy:
        - Create a regular time index at 0.1s intervals covering last 30s
        - Merge-asof (forward-fill) irregular raw samples onto that index
        - This gives exactly SEQUENCE_LENGTH rows per driver

    Returns empty dict if car_data is unavailable.
    """
    all_tel = {}
    try:
        car_data = session.car_data   # {driver_number_str: DataFrame}
    except Exception as e:
        log.error(f"session.car_data unavailable: {e}")
        return {}

    for drv_num, df in car_data.items():
        try:
            if df is None or df.empty:
                continue

            df = df.copy()

            # session.car_data uses 'SessionTime' (Timedelta) column
            if 'SessionTime' in df.columns:
                df['session_time_s'] = df['SessionTime'].dt.total_seconds()
            elif 'Time' in df.columns:
                df['session_time_s'] = df['Time'].dt.total_seconds()
            else:
                log.warning(f"Driver {drv_num}: no SessionTime/Time column — skipping")
                continue

            df = df.sort_values('session_time_s')
            t_max = df['session_time_s'].max()
            t_min = t_max - WINDOW_SECONDS

            window = df[df['session_time_s'] >= t_min].copy()
            if len(window) < 5:
                log.debug(f"Driver {drv_num}: only {len(window)} rows in last {WINDOW_SECONDS}s — skipping")
                continue

            # Build regular 10Hz time grid: [t_min, t_min+0.1, ..., t_max]
            regular_times = np.arange(t_min, t_max + 0.001, 1.0 / TARGET_HZ)
            # Trim to exactly SEQUENCE_LENGTH
            regular_times = regular_times[-SEQUENCE_LENGTH:]
            if len(regular_times) < SEQUENCE_LENGTH:
                log.debug(f"Driver {drv_num}: not enough history for {SEQUENCE_LENGTH} steps — skipping")
                continue

            # Merge-asof: for each regular timestamp, take the last observed value
            grid = pd.DataFrame({'session_time_s': regular_times})
            merged = pd.merge_asof(
                grid,
                window.sort_values('session_time_s'),
                on='session_time_s',
                direction='backward',
            )

            # Normalise column names to match bronze telemetry
            rename = {
                'Speed':            'speed',
                'nGear':            'gear',
                'Brake':            'brake',
                'X':                'x',
                'Y':                'y',
                'Z':                'z',
                'Distance':         'relative_distance',
                'RelativeDistance': 'relative_distance',
                'RPM':              'rpm',
                'DRS':              'drs',
                'Status':           'track_status',
            }
            merged = merged.rename(columns={k: v for k, v in rename.items() if k in merged.columns})

            # Ensure numeric
            for col in ['speed', 'gear', 'brake', 'x', 'y', 'z', 'relative_distance']:
                if col in merged.columns:
                    merged[col] = pd.to_numeric(merged[col], errors='coerce').fillna(0.0)

            all_tel[str(drv_num)] = merged

        except Exception as e:
            log.error(f"Error processing telemetry for driver {drv_num}: {e}", exc_info=True)
            continue

    log.info(f"Built telemetry for {len(all_tel)} drivers.")
    return all_tel


# ═════════════════════════════════════════════════════════════════════════════
# STEP 2 — Build lap-level lookup (compound, tyre_life, lap_number, team)
# ═════════════════════════════════════════════════════════════════════════════
def _build_lap_lookup(session) -> dict[str, dict]:
    """
    Returns {driver_number_str: {compound, tyre_life, lap_number, team, track_status}}
    using the latest completed lap for each driver from session.laps.
    """
    lookup = {}
    try:
        laps = session.laps
        if laps is None or laps.empty:
            return {}

        latest = laps.groupby("DriverNumber").tail(1)
        for _, row in latest.iterrows():
            drv_num = str(row.get("DriverNumber", ""))
            lookup[drv_num] = {
                "compound":     str(row.get("Compound", "Unknown") or "Unknown"),
                "tyre_life":    float(row.get("TyreLife", 0) or 0),
                "lap_number":   float(row.get("LapNumber", 0) or 0),
                "team":         str(row.get("Team", "Unknown") or "Unknown"),
                "track_status": str(row.get("TrackStatus", "1") or "1"),
                "driver_code":  str(row.get("Driver", "")),
            }
    except Exception as e:
        log.error(f"Error building lap lookup: {e}", exc_info=True)
    return lookup


# ═════════════════════════════════════════════════════════════════════════════
# STEP 3 — Build weather lookup (scalar values for the current moment)
# ═════════════════════════════════════════════════════════════════════════════
def _build_weather(session) -> dict:
    """
    Returns the most recent weather row as a plain dict.
    Falls back to safe defaults if weather_data is unavailable.
    """
    defaults = {
        "air_temp": 25.0, "humidity": 50.0, "pressure": 1013.0,
        "rainfall": 0.0,  "track_temp": 35.0, "wind_speed": 2.0,
        "wind_direction_sin": 0.0, "wind_direction_cos": 1.0,
    }
    try:
        wd = session.weather_data
        if wd is None or wd.empty:
            return defaults
        row = wd.iloc[-1]
        wd_deg = float(row.get("WindDirection", 0) or 0)
        wd_rad = np.deg2rad(wd_deg)
        return {
            "air_temp":          float(row.get("AirTemp",   defaults["air_temp"])),
            "humidity":          float(row.get("Humidity",  defaults["humidity"])),
            "pressure":          float(row.get("Pressure",  defaults["pressure"])),
            "rainfall":          float(bool(row.get("Rainfall", False))),
            "track_temp":        float(row.get("TrackTemp", defaults["track_temp"])),
            "wind_speed":        float(row.get("WindSpeed", defaults["wind_speed"])),
            "wind_direction_sin": float(np.sin(wd_rad)),
            "wind_direction_cos": float(np.cos(wd_rad)),
        }
    except Exception as e:
        log.warning(f"Weather unavailable ({e}) — using defaults.")
        return defaults


# ═════════════════════════════════════════════════════════════════════════════
# STEP 3b — Load live radio features from silver/radio_live.parquet
# ═════════════════════════════════════════════════════════════════════════════
def _load_radio_features(current_session_time_s: float) -> dict[str, dict]:
    """
    Read silver/radio_live.parquet from ADLS (written by radio_data.run_radio_live
    every 15 seconds) and build the 13 radio feature values for each driver.

    Strategy:
        - Filter to radio events within the last WINDOW_SECONDS of the current
          session time (same 30s window as the telemetry).
        - For each driver, aggregate signals across all events in that window:
            - radio_confidence        → mean confidence of events
            - radio_*_signal.*        → max of each boolean signal (1 if any event fired it)
            - radio_event_count       → count of events (fed into FEATURE_COLS, not RADIO_COLS)
        - radio_gate = 1.0 if any events found, 0.0 if none (controls attention gate).

    Returns:
        {driver_code: {"features": np.array (13,), "gate": float, "event_count": int}}

    Falls back to zero-filled features + gate=0.0 if parquet unavailable.
    """
    zero_result = {
        "features":    np.zeros(len(RADIO_COLS), dtype=np.float32),
        "gate":        0.0,
        "event_count": 0,
    }

    try:
        import adlfs
        fs = adlfs.AzureBlobFileSystem(
            account_name=STORAGE_ACCT_NAME,
            account_key=STORAGE_ACCT_KEY,
        )
        import pyarrow.parquet as pq
        with fs.open(RADIO_LIVE_PATH, "rb") as fh:
            radio_df = pq.read_table(fh).to_pandas()

    except FileNotFoundError:
        log.debug("radio_live.parquet not found yet — no radio events this cycle.")
        return {}
    except Exception as e:
        log.warning(f"Could not read radio_live.parquet: {e}")
        return {}

    if radio_df.empty:
        return {}

    # Filter to events within the last WINDOW_SECONDS
    t_min = current_session_time_s - WINDOW_SECONDS
    t_max = current_session_time_s

    if "radio_session_time" not in radio_df.columns:
        log.warning("radio_live.parquet missing radio_session_time column — skipping radio.")
        return {}

    window_df = radio_df[
        (radio_df["radio_session_time"] >= t_min) &
        (radio_df["radio_session_time"] <= t_max)
    ].copy()

    if window_df.empty:
        log.debug(f"No radio events in window [{t_min:.0f}s, {t_max:.0f}s].")
        return {}

    # driver_abb is the identifier in radio_live.parquet
    results = {}
    for driver_code, grp in window_df.groupby("driver_abb"):
        event_count = len(grp)
        features    = np.zeros(len(RADIO_COLS), dtype=np.float32)

        for parquet_col, model_col in RADIO_COL_MAP.items():
            if parquet_col not in grp.columns:
                continue
            idx = RADIO_COLS.index(model_col)
            vals = pd.to_numeric(grp[parquet_col], errors="coerce").fillna(0.0)

            if model_col == "radio_confidence":
                # Mean confidence across events in window
                features[idx] = float(vals.mean())
            else:
                # Boolean signals: 1.0 if any event in window fired this signal
                features[idx] = float(vals.max())

        results[str(driver_code)] = {
            "features":    features,
            "gate":        1.0,   # radio data is present for this driver
            "event_count": event_count,
        }

    log.info(f"Radio features built for {len(results)} drivers "
             f"in window [{t_min:.0f}s–{t_max:.0f}s].")
    return results


# ═════════════════════════════════════════════════════════════════════════════
# STEP 4 — OHE helpers
# ═════════════════════════════════════════════════════════════════════════════
def _ohe_compound(value: str, prefix: str, df: pd.DataFrame):
    """Add one-hot compound columns to df (in-place)."""
    val = str(value) if value and str(value) != 'nan' else 'nan'
    for cat in COMPOUND_CATS:
        df[f"{prefix}_compound_{cat}"] = float(val == cat)


def _ohe_gear(series: pd.Series, prefix: str, df: pd.DataFrame):
    """Add one-hot gear columns to df (in-place) for each row."""
    for cat in GEAR_CATS:
        df[f"{prefix}_gear_{cat}"] = (series == cat).astype(float)


def _ohe_team(value: str, prefix: str, df: pd.DataFrame):
    """Add one-hot team columns to df (in-place)."""
    val = str(value) if value else 'Unknown'
    for cat in TEAM_CATS:
        df[f"{prefix}_team_{cat}"] = float(val == cat)


def _ohe_track_status(value: str, df: pd.DataFrame):
    """Add one-hot track_status columns to df (in-place)."""
    val = str(value) if value else '1'
    for cat in TRACK_STATUS_CATS:
        df[f"track_status_{cat}"] = float(val == cat)


# ═════════════════════════════════════════════════════════════════════════════
# STEP 5 — Build (SEQUENCE_LENGTH, 129) feature matrix for one target driver
# ═════════════════════════════════════════════════════════════════════════════
def _build_feature_matrix(
    drv_num:         str,
    all_tel:         dict[str, pd.DataFrame],
    lap_lookup:      dict[str, dict],
    weather:         dict,
    social_score:    float,
    radio_event_count: int,
) -> np.ndarray | None:
    """
    Build the (300, 129) float32 feature matrix for one target driver,
    replicating exactly what the gold pipeline produced during training.

    Returns None if this driver's telemetry window is unavailable.
    """
    target_tel = all_tel.get(drv_num)
    if target_tel is None or len(target_tel) < SEQUENCE_LENGTH:
        return None

    target_tel = target_tel.tail(SEQUENCE_LENGTH).copy().reset_index(drop=True)
    lap_info   = lap_lookup.get(drv_num, {})

    n = SEQUENCE_LENGTH

    # ── Base feature DataFrame (one row per timestep) ────────────────────────
    feat = pd.DataFrame(index=range(n))

    # Target driver — raw telemetry
    feat['target_x']                 = target_tel.get('x', pd.Series(np.zeros(n))).values
    feat['target_y']                 = target_tel.get('y', pd.Series(np.zeros(n))).values
    feat['target_z']                 = target_tel.get('z', pd.Series(np.zeros(n))).values
    feat['target_speed']             = target_tel.get('speed', pd.Series(np.zeros(n))).values
    feat['target_brake']             = target_tel.get('brake', pd.Series(np.zeros(n))).astype(float).values
    feat['target_relative_distance'] = target_tel.get('relative_distance', pd.Series(np.zeros(n))).values
    feat['target_driver_number']     = float(drv_num) if drv_num.isdigit() else 0.0
    feat['target_lap_number']        = lap_info.get('lap_number', 0.0)
    feat['target_tyre_life']         = lap_info.get('tyre_life', 0.0)

    # Target gear OHE
    gear_series = pd.to_numeric(target_tel.get('gear', pd.Series(np.zeros(n))), errors='coerce').fillna(0.0)
    _ohe_gear(gear_series, 'target', feat)

    # Target compound OHE (same value for all 300 rows — lap-level)
    _ohe_compound(lap_info.get('compound', 'Unknown'), 'target', feat)

    # Target team OHE
    _ohe_team(lap_info.get('team', 'Unknown'), 'target', feat)

    # Track status OHE
    _ohe_track_status(lap_info.get('track_status', '1'), feat)

    # Weather (same for all 300 rows — session-level)
    feat['air_temp']           = weather['air_temp']
    feat['humidity']           = weather['humidity']
    feat['pressure']           = weather['pressure']
    feat['rainfall']           = weather['rainfall']
    feat['track_temp']         = weather['track_temp']
    feat['wind_speed']         = weather['wind_speed']
    feat['wind_direction_sin'] = weather['wind_direction_sin']
    feat['wind_direction_cos'] = weather['wind_direction_cos']

    # Radio event count — from live radio pipeline (how many events in last 30s)
    feat['radio_event_count'] = float(radio_event_count)

    # ── Driver ahead / behind — computed per timestep ────────────────────────
    # Strategy: for each of the 300 timesteps, look at what other drivers'
    # telemetry says about relative_distance at that same timestep index.
    # Since all drivers are resampled to the same 10Hz grid covering the same
    # last-30s window, timestep index i is the same real time for all drivers.

    # Build position array: {drv_num_str: relative_distance at each timestep}
    rd_matrix = {}   # drv_num_str → np.array (n,)
    pos_matrix = {}  # drv_num_str → np.array (n,) of position (int)
    for other_num, other_tel in all_tel.items():
        if len(other_tel) < SEQUENCE_LENGTH:
            continue
        other_tail = other_tel.tail(SEQUENCE_LENGTH).reset_index(drop=True)
        rd_matrix[other_num]  = other_tail.get('relative_distance', pd.Series(np.zeros(n))).values.astype(float)
        pos_matrix[other_num] = other_tail.get('position', pd.Series(np.zeros(n))).values.astype(float)

    target_rd = feat['target_relative_distance'].values  # (n,)

    # Pre-allocate ahead/behind arrays
    ahead_speed    = np.zeros(n); ahead_brake   = np.zeros(n)
    ahead_gear     = np.zeros(n); ahead_pos     = np.zeros(n)
    ahead_rd       = np.zeros(n); ahead_num     = np.zeros(n)
    ahead_lap      = np.zeros(n); ahead_tyre    = np.zeros(n)
    ahead_compound = ['Unknown'] * n; ahead_team = ['Unknown'] * n
    has_ahead      = np.zeros(n)

    behind_speed    = np.zeros(n); behind_brake  = np.zeros(n)
    behind_gear     = np.zeros(n); behind_pos    = np.zeros(n)
    behind_rd       = np.zeros(n); behind_num    = np.zeros(n)
    behind_lap      = np.zeros(n); behind_tyre   = np.zeros(n)
    behind_compound = ['Unknown'] * n; behind_team = ['Unknown'] * n
    has_behind      = np.zeros(n)

    for i in range(n):
        t_rd = target_rd[i]
        best_ahead_gap  = np.inf
        best_behind_gap = np.inf
        best_ahead_num  = None
        best_behind_num = None

        for other_num, other_rd_arr in rd_matrix.items():
            if other_num == drv_num:
                continue
            o_rd = other_rd_arr[i]
            gap  = o_rd - t_rd   # positive → other is ahead

            if 0 < gap < best_ahead_gap:
                best_ahead_gap = gap
                best_ahead_num = other_num

            if gap < 0 and abs(gap) < best_behind_gap:
                best_behind_gap = abs(gap)
                best_behind_num = other_num

        # Fill ahead
        if best_ahead_num is not None:
            has_ahead[i]      = 1.0
            o_tel = all_tel[best_ahead_num].tail(SEQUENCE_LENGTH).reset_index(drop=True)
            o_lap = lap_lookup.get(best_ahead_num, {})
            ahead_speed[i]    = float(o_tel.get('speed',  pd.Series([0.0])).iloc[i] if i < len(o_tel) else 0.0)
            ahead_brake[i]    = float(o_tel.get('brake',  pd.Series([0.0])).iloc[i] if i < len(o_tel) else 0.0)
            ahead_gear[i]     = float(o_tel.get('gear',   pd.Series([0.0])).iloc[i] if i < len(o_tel) else 0.0)
            ahead_pos[i]      = float(pos_matrix[best_ahead_num][i])
            ahead_rd[i]       = float(rd_matrix[best_ahead_num][i])
            ahead_num[i]      = float(best_ahead_num) if best_ahead_num.isdigit() else 0.0
            ahead_lap[i]      = o_lap.get('lap_number', 0.0)
            ahead_tyre[i]     = o_lap.get('tyre_life',  0.0)
            ahead_compound[i] = o_lap.get('compound', 'Unknown')
            ahead_team[i]     = o_lap.get('team', 'Unknown')

        # Fill behind
        if best_behind_num is not None:
            has_behind[i]      = 1.0
            o_tel = all_tel[best_behind_num].tail(SEQUENCE_LENGTH).reset_index(drop=True)
            o_lap = lap_lookup.get(best_behind_num, {})
            behind_speed[i]    = float(o_tel.get('speed',  pd.Series([0.0])).iloc[i] if i < len(o_tel) else 0.0)
            behind_brake[i]    = float(o_tel.get('brake',  pd.Series([0.0])).iloc[i] if i < len(o_tel) else 0.0)
            behind_gear[i]     = float(o_tel.get('gear',   pd.Series([0.0])).iloc[i] if i < len(o_tel) else 0.0)
            behind_pos[i]      = float(pos_matrix[best_behind_num][i])
            behind_rd[i]       = float(rd_matrix[best_behind_num][i])
            behind_num[i]      = float(best_behind_num) if best_behind_num.isdigit() else 0.0
            behind_lap[i]      = o_lap.get('lap_number', 0.0)
            behind_tyre[i]     = o_lap.get('tyre_life',  0.0)
            behind_compound[i] = o_lap.get('compound', 'Unknown')
            behind_team[i]     = o_lap.get('team', 'Unknown')

    # Assign ahead columns
    feat['has_driver_ahead']      = has_ahead
    feat['has_driver_behind']     = has_behind
    feat['distance_ahead_target'] = np.where(has_ahead, [rd_matrix.get(best_ahead_num, np.zeros(n))[i] - target_rd[i] if has_ahead[i] else 0.0 for i in range(n)], 0.0)
    feat['distance_target_behind']= np.where(has_behind, [target_rd[i] - rd_matrix.get(best_behind_num, np.zeros(n))[i] if has_behind[i] else 0.0 for i in range(n)], 0.0)

    feat['driver_ahead_speed']      = ahead_speed
    feat['driver_ahead_brake']      = ahead_brake
    feat['driver_ahead_pos']        = ahead_pos
    feat['driver_ahead_number']     = ahead_num
    feat['driver_ahead_lap_number'] = ahead_lap
    feat['driver_ahead_tyre_life']  = ahead_tyre
    feat['driver_behind_speed']     = behind_speed
    feat['driver_behind_brake']     = behind_brake
    feat['driver_behind_pos']       = behind_pos
    feat['driver_behind_number']    = behind_num
    feat['driver_behind_lap_number']= behind_lap
    feat['driver_behind_tyre_life'] = behind_tyre

    # Ahead gear OHE
    _ohe_gear(pd.Series(ahead_gear), 'driver_ahead', feat)
    # Behind gear OHE
    _ohe_gear(pd.Series(behind_gear), 'driver_behind', feat)

    # Ahead compound OHE (per-timestep — can change if driver pits mid-window)
    for cat in COMPOUND_CATS:
        feat[f'driver_ahead_compound_{cat}']  = [float(v == cat) for v in ahead_compound]
        feat[f'driver_behind_compound_{cat}'] = [float(v == cat) for v in behind_compound]

    # Ahead/behind team OHE (lap-level — won't change mid-window)
    for cat in TEAM_CATS:
        feat[f'driver_ahead_team_{cat}']  = [float(v == cat) for v in ahead_team]
        feat[f'driver_behind_team_{cat}'] = [float(v == cat) for v in behind_team]

    # ── Final assembly — select FEATURE_COLS in exact order ─────────────────
    # Any feature_col not built above defaults to 0.0
    for col in FEATURE_COLS:
        if col not in feat.columns:
            feat[col] = 0.0

    matrix = feat[FEATURE_COLS].astype(np.float32).fillna(0.0).values
    assert matrix.shape == (SEQUENCE_LENGTH, len(FEATURE_COLS)), \
        f"Shape mismatch: {matrix.shape} vs expected ({SEQUENCE_LENGTH}, {len(FEATURE_COLS)})"

    return matrix


# ═════════════════════════════════════════════════════════════════════════════
# STEP 6 — Run channels + RF, return predictions
# ═════════════════════════════════════════════════════════════════════════════
def run_inference_cycle(session) -> dict[str, dict]:
    """
    Full inference cycle. Called by Thread 3 in live_casting.py.

    1. Build all-driver telemetry snapshot
    2. Build lap lookup, weather, social scores, radio features
    3. For each driver: build (300, 129) feature matrix → DriverChannel → embedding
       Also build (300, 13) radio tensor + gate + social score for channel input
    4. Stack all embeddings → RandomForest → predicted positions

    Returns:
        {driver_code: {"predicted_position": int, "prediction_confidence": float}}
    """
    import torch

    t_start = datetime.utcnow()
    log.info("Inference cycle starting...")

    try:
        rf = _load_rf()
        if rf is None:
            log.warning("RF model unavailable — skipping inference.")
            return {}

        # ── Build shared data once per cycle ──────────────────────────────
        all_tel    = _build_all_drivers_telemetry(session)
        lap_lookup = _build_lap_lookup(session)
        weather    = _build_weather(session)

        if not all_tel:
            log.warning("No telemetry data available — skipping inference.")
            return {}

        # Current session time (seconds) — used to filter radio window
        # Take the max session_time across any driver's telemetry
        current_session_time_s = max(
            df['session_time_s'].max()
            for df in all_tel.values()
            if 'session_time_s' in df.columns
        )

        # Radio features: {driver_code: {features (13,), gate, event_count}}
        radio_by_driver = _load_radio_features(current_session_time_s)

        # Social scores: use module-level cache (loaded once at startup)
        # _SOCIAL_SCORES = {driver_code: float} for April 2026
        social_scores = _SOCIAL_SCORES   # static — no reload needed

        device = _get_device()

        # ── Per-driver: feature matrix + radio tensor → channel embedding ──
        embeddings = {}   # driver_code → np.array (EMBEDDING_DIM,)

        for drv_num, lap_info in lap_lookup.items():
            driver_code = lap_info.get('driver_code')
            if not driver_code:
                continue

            channel = _load_channel(driver_code)
            if channel is None:
                continue

            # ── Social score ───────────────────────────────────────────────
            # Use April 2026 score if available, else None → model default 5.0
            raw_social = social_scores.get(driver_code)
            social_val = float(raw_social) if raw_social is not None else 5.0

            # ── Radio features for this driver ─────────────────────────────
            radio_info    = radio_by_driver.get(driver_code, {})
            radio_feats   = radio_info.get("features", np.zeros(len(RADIO_COLS), dtype=np.float32))
            radio_gate    = radio_info.get("gate", 0.0)
            radio_ev_cnt  = radio_info.get("event_count", 0)

            # ── Feature matrix (300, 129) ──────────────────────────────────
            matrix = _build_feature_matrix(
                drv_num, all_tel, lap_lookup, weather,
                social_score=social_val,
                radio_event_count=radio_ev_cnt,
            )
            if matrix is None:
                log.debug(f"{driver_code}: feature matrix unavailable — skipping.")
                continue

            # ── Tensors ────────────────────────────────────────────────────
            # Main features: (1, 300, 129)
            feat_tensor = torch.tensor(
                matrix, dtype=torch.float32
            ).unsqueeze(0).to(device)

            # Get radio_dim from checkpoint (cached after first load)
            ckpt_path = os.path.join(MODEL_DIR, f"channel_{driver_code}.pt")
            ckpt      = torch.load(ckpt_path, map_location=device, weights_only=False)
            r_dim     = ckpt.get("radio_dim", 13)

            # Radio features: (1, 300, 13)
            # Broadcast the 13-dim radio vector across all 300 timesteps.
            # The RadioAttention gate controls whether it's used per-timestep.
            radio_matrix = np.tile(radio_feats, (SEQUENCE_LENGTH, 1))  # (300, 13)
            radio_tensor = torch.tensor(
                radio_matrix, dtype=torch.float32
            ).unsqueeze(0).to(device)   # (1, 300, 13)

            # Gate: (1, 300) — 1.0 if radio data present, 0.0 otherwise
            gate_array  = np.full(SEQUENCE_LENGTH, radio_gate, dtype=np.float32)
            gate_tensor = torch.tensor(
                gate_array, dtype=torch.float32
            ).unsqueeze(0).to(device)   # (1, 300)

            # Social: (1, 1)
            social_tensor = torch.tensor(
                [[social_val]], dtype=torch.float32
            ).to(device)

            with torch.no_grad():
                embedding = channel(
                    feat_tensor, radio_tensor, gate_tensor, social_tensor
                )   # (1, EMBEDDING_DIM)

            embeddings[driver_code] = embedding.squeeze(0).cpu().numpy()
            log.debug(
                f"{driver_code}: embedding done | "
                f"social={social_val:.2f} | "
                f"radio_gate={radio_gate} | "
                f"radio_events={radio_ev_cnt}"
            )

        if not embeddings:
            log.warning("No embeddings produced — inference aborted.")
            return {}

        # ── Stack all embeddings → RF ──────────────────────────────────────
        from src.model import DRIVER_LIST, EMBEDDING_DIM

        rf_input_parts = []
        for drv in DRIVER_LIST:
            if drv in embeddings:
                rf_input_parts.append(embeddings[drv])
            else:
                # Zero-pad retired/missing drivers
                rf_input_parts.append(np.zeros(EMBEDDING_DIM, dtype=np.float32))

        rf_input  = np.concatenate(rf_input_parts).reshape(1, -1)
        proba     = rf.predict_proba(rf_input)[0]   # (n_classes,)
        predicted = rf.predict(rf_input)[0]

        results = {}
        for driver_code in embeddings:
            results[driver_code] = {
                "predicted_position":    int(predicted),
                "prediction_confidence": float(proba.max()),
            }

        elapsed = (datetime.utcnow() - t_start).total_seconds()
        log.info(
            f"Inference complete: {len(embeddings)} drivers in {elapsed:.1f}s | "
            f"P1 confidence: {proba.max():.2%} | "
            f"Radio: {len(radio_by_driver)} drivers with live data | "
            f"Social: {len([v for v in social_scores.values() if v is not None])} drivers scored"
        )

        set_predictions(results)
        return results

    except Exception as e:
        log.error(f"Inference cycle failed: {e}", exc_info=True)
        return {}
