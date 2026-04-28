import os
import logging
import gc
import json
import fsspec
from pathlib import Path
from typing import List, Optional
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import numpy as np

# --------------------------- Configuration ------------------------------
DATA_ROOT = Path(os.getenv("DATA_ROOT", "data"))

SILVER_PATH = DATA_ROOT / "silver"
GOLD_PATH   = DATA_ROOT / "gold"

YEARS_ENV = os.getenv("YEARS", "2024,2025")
YEARS = [int(y.strip()) for y in YEARS_ENV.split(",") if y.strip()]

TARGET_DRIVER = os.getenv("TARGET_DRIVER", "LEC")
SESSION_TYPE  = os.getenv("SESSION_TYPE", "R")

race_locations = [
    "Italian Grand Prix", "Azerbaijan Grand Prix", "Singapore Grand Prix", "Mexico City Grand Prix",
    "Brazilian Grand Prix", "Las Vegas Grand Prix", "Qatar Grand Prix", "Abu Dhabi Grand Prix",
    "United States Grand Prix", "Australian Grand Prix", "Austrian Grand Prix", "Bahrain Grand Prix",
    "Belgian Grand Prix", "British Grand Prix", "Canadian Grand Prix", "Chinese Grand Prix",
    "Dutch Grand Prix", "Eifel Grand Prix", "Emilia Romagna Grand Prix", "French Grand Prix",
    "German Grand Prix", "Hungarian Grand Prix", "Japanese Grand Prix", "Miami Grand Prix",
    "Monaco Grand Prix", "Portuguese Grand Prix", "Russian Grand Prix", "Sakhir Grand Prix",
    "Saudi Arabian Grand Prix", "Spanish Grand Prix", "Styrian Grand Prix", "São Paulo Grand Prix",
    "Turkish Grand Prix", "Tuscan Grand Prix"
]


# ── Azure helpers ──────────────────────────────────────────────────────────────

def _get_storage_options(storage_account: str, storage_key: str) -> dict:
    return {"account_name": storage_account, "account_key": storage_key}


def _get_fs(storage_options: dict):
    return fsspec.filesystem("abfs", **storage_options)


def _abfs(container: str, path: str) -> str:
    return f"abfs://{container}/{path}"


# ── File path helpers ──────────────────────────────────────────────────────────

def _get_silver_file(
    race_year: int,
    race_location: str,
    storage_options: dict = None,
    silver_container: str = None
) -> str:
    safe_location = race_location.replace(" ", "_")
    if storage_options and silver_container:
        return _abfs(silver_container, f"{race_year}/{SESSION_TYPE}/{safe_location}.parquet")
    return str(SILVER_PATH / str(race_year) / SESSION_TYPE / f"{safe_location}.parquet")


def _get_social_file(
    storage_options: dict = None,
    silver_container: str = None
):
    """Returns (path, storage_options, is_azure)"""
    if storage_options and silver_container:
        return _abfs(silver_container, "social_media_silver.json"), storage_options, True
    return SILVER_PATH / "social_media_silver.json", {}, False


def _get_radio_file(
    storage_options: dict = None,
    silver_container: str = None
):
    """Returns (path, storage_options, is_azure)"""
    if storage_options and silver_container:
        return _abfs(silver_container, "radio.parquet"), storage_options, True
    return SILVER_PATH / "radio.parquet", {}, False


def _silver_exists(
    silver_file: str,
    use_azure: bool,
    storage_options: dict
) -> bool:
    if use_azure:
        try:
            fs = _get_fs(storage_options)
            bare = silver_file.replace("abfs://", "", 1)
            return fs.exists(bare)
        except Exception as e:
            logging.warning(f"Could not check existence of {silver_file}: {e}")
            return False
    return Path(silver_file).exists()


# ── Preprocessing ──────────────────────────────────────────────────────────────

def preprocess_race_df(df: pd.DataFrame, target_driver: str) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()

    required_cols = ["session_time", "position", "driver_code"]
    if not all(c in df.columns for c in required_cols):
        logging.warning("Missing required columns in silver")
        return pd.DataFrame()

    target_df = df[df["driver_code"] == target_driver].copy()
    if target_df.empty:
        return pd.DataFrame()

    target_df["target_pos_lookup"] = pd.to_numeric(target_df["position"], errors="coerce")

    target_df = target_df.rename(columns={
        "x":                "target_x",
        "y":                "target_y",
        "z":                "target_z",
        "speed":            "target_speed",
        "gear":             "target_gear",
        "brake":            "target_brake",
        "position":         "target_pos",
        "relative_distance":"target_relative_distance",
        "Driver":           "target_driver",
        "DriverNumber":     "target_driver_number",
        "LapNumber":        "target_lap_number",
        "TyreLife":         "target_tyre_life",
        "Compound":         "target_compound",
        "Team":             "target_team",
        "TrackStatus":      "track_status",
        "AirTemp":          "air_temp",
        "Humidity":         "humidity",
        "Pressure":         "pressure",
        "Rainfall":         "rainfall",
        "TrackTemp":        "track_temp",
        "WindDirection":    "wind_direction",
        "WindSpeed":        "wind_speed",
    })

    target_cols = [
        "target_x", "target_y", "target_z", "target_speed", "target_gear",
        "target_brake", "target_pos", "target_relative_distance", "target_driver",
        "target_driver_number", "target_lap_number", "target_tyre_life",
        "target_compound", "target_team", "track_status", "air_temp", "humidity",
        "pressure", "rainfall", "track_temp", "wind_direction", "wind_speed"
    ]
    for col in target_cols:
        if col not in target_df.columns:
            target_df[col] = pd.NA

    target_df["ahead_pos_lookup"]  = target_df["target_pos_lookup"] - 1
    target_df["behind_pos_lookup"] = target_df["target_pos_lookup"] + 1

    base_cols = [
        "session_time", "race_id", "race_date", "race_year", "race_location",
        "target_x", "target_y", "target_z",
        "target_speed", "target_gear", "target_brake",
        "target_pos", "target_relative_distance", "target_driver", "target_driver_number",
        "target_lap_number", "target_tyre_life", "target_compound", "target_team",
        "track_status",
        "air_temp", "humidity", "pressure", "rainfall", "track_temp",
        "wind_direction", "wind_speed",
        "target_pos_lookup", "ahead_pos_lookup", "behind_pos_lookup"
    ]
    base_cols  = [c for c in base_cols if c in target_df.columns]
    target_df  = target_df[base_cols].copy()

    others = df.copy()
    others["position_lookup"] = pd.to_numeric(others["position"], errors="coerce")

    join_cols = ["race_id", "race_year", "race_location", "session_time"]

    # ---------- AHEAD ----------
    ahead_source_cols = [
        "race_id", "race_year", "race_location", "session_time", "position_lookup",
        "speed", "gear", "brake", "position", "relative_distance",
        "Driver", "DriverNumber", "LapNumber", "TyreLife", "Compound", "Team"
    ]
    ahead_source_cols = [c for c in ahead_source_cols if c in others.columns]

    ahead_df = others[ahead_source_cols].rename(columns={
        "position_lookup":  "ahead_pos_lookup",
        "speed":            "driver_ahead_speed",
        "gear":             "driver_ahead_gear",
        "brake":            "driver_ahead_brake",
        "position":         "driver_ahead_pos",
        "relative_distance":"driver_ahead_relative_distance",
        "Driver":           "driver_ahead",
        "DriverNumber":     "driver_ahead_number",
        "LapNumber":        "driver_ahead_lap_number",
        "TyreLife":         "driver_ahead_tyre_life",
        "Compound":         "driver_ahead_compound",
        "Team":             "driver_ahead_team",
    })

    ahead_keep = [
        "race_id", "race_year", "race_location", "session_time", "ahead_pos_lookup",
        "driver_ahead_speed", "driver_ahead_gear",
        "driver_ahead_brake", "driver_ahead_pos", "driver_ahead_relative_distance",
        "driver_ahead", "driver_ahead_number", "driver_ahead_lap_number",
        "driver_ahead_tyre_life", "driver_ahead_compound", "driver_ahead_team"
    ]
    ahead_keep = [c for c in ahead_keep if c in ahead_df.columns]
    ahead_df   = ahead_df[ahead_keep]

    gold_df = target_df.merge(ahead_df, on=join_cols + ["ahead_pos_lookup"], how="left")

    # ---------- BEHIND ----------
    behind_source_cols = [
        "race_id", "race_year", "race_location", "session_time", "position_lookup",
        "speed", "gear", "brake", "position", "relative_distance",
        "Driver", "DriverNumber", "LapNumber", "TyreLife", "Compound", "Team"
    ]
    behind_source_cols = [c for c in behind_source_cols if c in others.columns]

    behind_df = others[behind_source_cols].rename(columns={
        "position_lookup":  "behind_pos_lookup",
        "speed":            "driver_behind_speed",
        "gear":             "driver_behind_gear",
        "brake":            "driver_behind_brake",
        "position":         "driver_behind_pos",
        "relative_distance":"driver_behind_relative_distance",
        "Driver":           "driver_behind",
        "DriverNumber":     "driver_behind_number",
        "LapNumber":        "driver_behind_lap_number",
        "TyreLife":         "driver_behind_tyre_life",
        "Compound":         "driver_behind_compound",
        "Team":             "driver_behind_team",
    })

    behind_keep = [
        "race_id", "race_year", "race_location", "session_time", "behind_pos_lookup",
        "driver_behind_speed", "driver_behind_gear",
        "driver_behind_brake", "driver_behind_pos", "driver_behind_relative_distance",
        "driver_behind", "driver_behind_number", "driver_behind_lap_number",
        "driver_behind_tyre_life", "driver_behind_compound", "driver_behind_team"
    ]
    behind_keep = [c for c in behind_keep if c in behind_df.columns]
    behind_df   = behind_df[behind_keep]

    gold_df = gold_df.merge(behind_df, on=join_cols + ["behind_pos_lookup"], how="left")

    logging.info(f"[{target_driver}] After merge shape: {gold_df.shape}")

    # ---------- FALLBACK LOGIC ----------
    others_min_cols = [
        "race_id", "race_year", "race_location", "session_time", "position_lookup",
        "speed", "gear", "brake", "position", "relative_distance",
        "Driver", "DriverNumber", "LapNumber", "TyreLife", "Compound", "Team"
    ]
    others_min_cols = [c for c in others_min_cols if c in others.columns]
    others_min = others[others_min_cols].dropna(subset=["position_lookup"]).copy()

    group_keys = ["race_id", "race_year", "race_location", "session_time"]
    same_time_lookup = {
        key: group.sort_values("position_lookup")
        for key, group in others_min.groupby(group_keys, sort=False)
    }

    result_rows = []
    for idx, row in enumerate(gold_df.itertuples(index=False)):
        row_dict = row._asdict()
        try:
            target_pos_num = int(row_dict.get("target_pos"))
        except Exception:
            result_rows.append(row_dict)
            continue

        key = (
            row_dict.get("race_id"),
            row_dict.get("race_year"),
            row_dict.get("race_location"),
            row_dict.get("session_time")
        )
        same_time = same_time_lookup.get(key)

        if same_time is None or same_time.empty:
            result_rows.append(row_dict)
            continue

        if pd.isna(row_dict.get("driver_ahead_pos")):
            ahead_candidates = same_time[same_time["position_lookup"] < target_pos_num]
            if not ahead_candidates.empty:
                best = ahead_candidates.iloc[-1]
                row_dict["driver_ahead_speed"]       = best.get("speed")
                row_dict["driver_ahead_gear"]        = best.get("gear")
                row_dict["driver_ahead_brake"]       = best.get("brake")
                row_dict["driver_ahead_pos"]         = best.get("position")
                row_dict["driver_ahead_relative_distance"] = best.get("relative_distance")
                row_dict["driver_ahead"]             = best.get("Driver")
                row_dict["driver_ahead_number"]      = best.get("DriverNumber")
                row_dict["driver_ahead_lap_number"]  = best.get("LapNumber")
                row_dict["driver_ahead_tyre_life"]   = best.get("TyreLife")
                row_dict["driver_ahead_compound"]    = best.get("Compound")
                row_dict["driver_ahead_team"]        = best.get("Team")

        if pd.isna(row_dict.get("driver_behind_pos")):
            behind_candidates = same_time[same_time["position_lookup"] > target_pos_num]
            if not behind_candidates.empty:
                best = behind_candidates.iloc[0]
                row_dict["driver_behind_speed"]       = best.get("speed")
                row_dict["driver_behind_gear"]        = best.get("gear")
                row_dict["driver_behind_brake"]       = best.get("brake")
                row_dict["driver_behind_pos"]         = best.get("position")
                row_dict["driver_behind_relative_distance"] = best.get("relative_distance")
                row_dict["driver_behind"]             = best.get("Driver")
                row_dict["driver_behind_number"]      = best.get("DriverNumber")
                row_dict["driver_behind_lap_number"]  = best.get("LapNumber")
                row_dict["driver_behind_tyre_life"]   = best.get("TyreLife")
                row_dict["driver_behind_compound"]    = best.get("Compound")
                row_dict["driver_behind_team"]        = best.get("Team")

        result_rows.append(row_dict)

    result_df = pd.DataFrame(result_rows)

    result_df["distance_ahead_target"] = (
        pd.to_numeric(result_df["driver_ahead_relative_distance"],  errors="coerce")
        - pd.to_numeric(result_df["target_relative_distance"], errors="coerce")
    )
    result_df["distance_target_behind"] = (
        pd.to_numeric(result_df["target_relative_distance"], errors="coerce")
        - pd.to_numeric(result_df["driver_behind_relative_distance"], errors="coerce")
    )
    result_df = result_df.drop(
        columns=[
            "target_pos_lookup", "ahead_pos_lookup", "behind_pos_lookup",
            "driver_behind_relative_distance", "driver_ahead_relative_distance"
        ],
        errors="ignore"
    )

    logging.info(f"[{target_driver}] Target rows after preprocess: {len(target_df)}")
    return result_df


def feature_engineering(gold_df: pd.DataFrame) -> pd.DataFrame:
    df = gold_df.copy()

    bool_cols = ["target_brake", "rainfall", "driver_ahead_brake", "driver_behind_brake"]
    for col in bool_cols:
        if col in df.columns:
            df[col] = df[col].fillna(False).astype(int)

    numeric_cols = [
        "session_time",
        "target_x", "target_y", "target_z",
        "target_speed", "target_gear", "target_pos",
        "target_lap_number", "target_tyre_life",
        "air_temp", "humidity", "pressure", "track_temp",
        "wind_direction", "wind_speed",
        "driver_ahead_speed", "driver_ahead_gear",
        "driver_ahead_pos", "driver_ahead_lap_number", "driver_ahead_tyre_life",
        "driver_behind_speed", "driver_behind_gear",
        "driver_behind_pos", "driver_behind_lap_number", "driver_behind_tyre_life",
        "distance_ahead_target", "distance_target_behind"
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    if "distance_ahead_target" in df.columns:
        df["has_driver_ahead"] = df["distance_ahead_target"].notna().astype(int)
    if "distance_target_behind" in df.columns:
        df["has_driver_behind"] = df["distance_target_behind"].notna().astype(int)

    categorical_cols = [
        "target_compound", "target_team", "track_status",
        "driver_ahead_compound", "driver_ahead_team",
        "driver_behind_compound", "driver_behind_team"
    ]
    categorical_cols = [c for c in categorical_cols if c in df.columns]
    for col in categorical_cols:
        df[col] = df[col].fillna("Unknown").astype(str)
    df = pd.get_dummies(df, columns=categorical_cols, dummy_na=False)

    radians = np.deg2rad(pd.to_numeric(df["wind_direction"], errors="coerce"))
    df["wind_direction_sin"] = np.sin(radians)
    df["wind_direction_cos"] = np.cos(radians)
    df = df.drop(columns=["wind_direction"], errors="ignore")

    df = pd.get_dummies(
        df,
        columns=["target_gear", "driver_ahead_gear", "driver_behind_gear"],
        dummy_na=False
    )

    logging.info(f"After feature engineering shape: {df.shape}")
    return df


def _add_social_info(
    gold_df: pd.DataFrame,
    storage_options: dict = None,
    silver_container: str = None
) -> pd.DataFrame:
    social_path, so, is_azure = _get_social_file(storage_options, silver_container)

    social_data = None

    if is_azure:
        try:
            with fsspec.open(str(social_path), "r", **so) as f:
                social_data = json.load(f)
            logging.info("Loaded social_media_silver.json from Azure")
        except Exception as e:
            logging.warning(f"Could not read social file from Azure: {e}. Skipping social join.")
            gold_df["social_life_score"] = float("nan")
            return gold_df
    else:
        if not Path(social_path).exists():
            logging.warning(f"Social media silver file not found at {social_path}. Skipping.")
            gold_df["social_life_score"] = float("nan")
            return gold_df
        with open(social_path, "r") as f:
            social_data = json.load(f)

    social_lookup: dict = {}
    for year_key, months in social_data.items():
        if year_key == "Status":
            continue
        if not isinstance(months, dict):
            continue
        for month_key, drivers in months.items():
            if not isinstance(drivers, dict):
                continue
            for driver_abb, score in drivers.items():
                social_lookup[(str(year_key), str(month_key), driver_abb)] = score

    if gold_df.empty:
        gold_df["social_life_score"] = float("nan")
        return gold_df

    if "race_date" in gold_df.columns:
        race_dates   = pd.to_datetime(gold_df["race_date"], errors="coerce")
        year_series  = race_dates.dt.year.fillna(
            gold_df.get("race_year", pd.Series(dtype=float))
        ).astype("Int64").astype(str)
        month_series = race_dates.dt.month.astype("Int64").astype(str)
    else:
        year_series  = gold_df["race_year"].astype(str)
        month_series = pd.Series([""] * len(gold_df), index=gold_df.index)

    driver_series = gold_df["target_driver"].astype(str).str.upper()

    gold_df["social_life_score"] = [
        social_lookup.get((y, m, d), float("nan"))
        for y, m, d in zip(year_series, month_series, driver_series)
    ]

    matched = gold_df["social_life_score"].notna().sum()
    logging.info(
        f"Social media join complete: {matched}/{len(gold_df)} rows matched "
        f"({matched/len(gold_df)*100:.1f}%)"
    )
    return gold_df


def _add_radio_info(
    gold_df: pd.DataFrame,
    storage_options: dict = None,
    silver_container: str = None
) -> pd.DataFrame:
    import numpy as np

    radio_path, so, is_azure = _get_radio_file(storage_options, silver_container)

    if not is_azure and not Path(radio_path).exists():
        logging.warning(f"radio.parquet not found at {radio_path}. Skipping radio join.")
        gold_df["radio_data_available"] = 0
        return gold_df

    try:
        radio_df = pd.read_parquet(
            str(radio_path),
            storage_options=so if is_azure else None
        )
        logging.info("Loaded radio.parquet successfully")
    except Exception as e:
        logging.warning(f"Could not read radio.parquet: {e}. Skipping radio join.")
        gold_df["radio_data_available"] = 0
        return gold_df

    if radio_df.empty:
        logging.warning("radio.parquet is empty. Skipping radio join.")
        gold_df["radio_data_available"] = 0
        return gold_df

    drop_cols = [
        "session_key", "meeting_key", "driver_number",
        "recording_time", "recording_url", "transcript_cleaned",
    ]
    radio_df = radio_df.drop(columns=[c for c in drop_cols if c in radio_df.columns])

    quality_map = {"low": 0, "medium": 1, "high": 2}
    if "transcript_quality" in radio_df.columns:
        radio_df["transcript_quality"] = radio_df["transcript_quality"].map(quality_map)

    ohe_cols = [c for c in [
        "primary_event_type",
        "action_type",
        "car_issue_signal.issue_type",
        "car_issue_signal.severity",
    ] if c in radio_df.columns]
    radio_df = pd.get_dummies(radio_df, columns=ohe_cols, prefix=ohe_cols, dtype=float)

    if "secondary_event_types" in radio_df.columns:
        radio_df = radio_df.reset_index(drop=True)
        radio_df["_row_idx"] = radio_df.index

        sec_exploded = (
            radio_df[["_row_idx", "secondary_event_types"]]
            .explode("secondary_event_types")
        )
        sec_exploded["secondary_event_types"] = (
            sec_exploded["secondary_event_types"].fillna("none")
        )
        sec_dummies = pd.get_dummies(
            sec_exploded["secondary_event_types"],
            prefix="secondary_event_type",
            dtype=float,
        )
        sec_dummies = (
            pd.concat([sec_exploded[["_row_idx"]], sec_dummies], axis=1)
            .groupby("_row_idx", as_index=True)
            .sum()
            .clip(upper=1)
        )
        none_col = "secondary_event_type_none"
        if none_col in sec_dummies.columns:
            sec_dummies = sec_dummies.drop(columns=[none_col])

        radio_df = (
            radio_df.drop(columns=["secondary_event_types"])
            .join(sec_dummies, on="_row_idx")
            .drop(columns=["_row_idx"])
        )

    radio_df = radio_df.rename(columns={"radio_session_time": "session_time"})

    join_keys = {"year", "driver_abb", "grand_prix_name", "session_time"}
    rename_map = {c: f"radio_{c}" for c in radio_df.columns if c not in join_keys}
    radio_df = radio_df.rename(columns=rename_map)
    radio_feature_cols = [c for c in radio_df.columns if c not in join_keys]

    radio_df["session_time"] = pd.to_numeric(radio_df["session_time"], errors="coerce")
    gold_df["session_time"]  = pd.to_numeric(gold_df["session_time"],  errors="coerce")

    for col in radio_feature_cols:
        gold_df[col] = float("nan")
    gold_df["radio_data_available"] = 0

    gold_df["race_year"] = gold_df["race_year"].astype(str)
    total_matched = 0

    for (driver, year, location), gold_group in gold_df.groupby(
        ["target_driver", "race_year", "race_location"], sort=False
    ):
        radio_group = radio_df[
            (radio_df["driver_abb"] == driver)
            & (radio_df["year"].astype(str) == str(year))
            & (radio_df["grand_prix_name"] == location)
        ].copy()

        if radio_group.empty:
            continue

        gold_sorted  = gold_group.sort_values("session_time")
        radio_sorted = (
            radio_group[["session_time"] + radio_feature_cols]
            .sort_values("session_time")
            .copy()
        )

        merged = pd.merge_asof(
            gold_sorted.reset_index(),
            radio_sorted,
            on="session_time",
            direction="nearest",
            suffixes=("", "_radio_tmp"),
        )
        original_idx  = merged["index"].values
        gold_times    = merged["session_time"].values
        radio_times   = radio_sorted["session_time"].values

        diffs = np.abs(gold_times[:, None] - radio_times[None, :])
        nearest_gold_pos = diffs.argmin(axis=0)

        for radio_col in radio_feature_cols:
            if radio_col in merged.columns:
                gold_df.loc[
                    original_idx[nearest_gold_pos], radio_col
                ] = merged[radio_col].iloc[nearest_gold_pos].values

        gold_df.loc[original_idx[nearest_gold_pos], "radio_data_available"] = 1
        total_matched += len(nearest_gold_pos)

    logging.info(
        f"Radio join complete: {total_matched} session_time rows matched | "
        f"{len(radio_feature_cols)} radio feature columns added."
    )
    return gold_df


def write_gold_output(
    gold_df: pd.DataFrame,
    target_driver: str,
    storage_options: dict = None,
    gold_container: str = None,
) -> str:
    if storage_options and gold_container:
        out_file = _abfs(gold_container, f"{SESSION_TYPE}/{target_driver}_gold.parquet")
        gold_df.to_parquet(out_file, index=False, storage_options=storage_options)
    else:
        out_dir = GOLD_PATH / SESSION_TYPE
        out_dir.mkdir(parents=True, exist_ok=True)
        out_file = str(out_dir / f"{target_driver}_gold.parquet")
        gold_df.to_parquet(out_file, index=False)

    logging.info(f"Saved gold to: {out_file}")
    logging.info(f"Final gold shape: {gold_df.shape}")
    return out_file


def run_gold_pipeline(
    target_driver: str = TARGET_DRIVER,
    storage_account: str = None,
    storage_key: str = None,
    silver_container: str = None,
    gold_container: str = None,
) -> List[str]:

    if storage_account and storage_key:
        storage_options  = _get_storage_options(storage_account, storage_key)
        _silver_container = silver_container or "silver"
        _gold_container   = gold_container   or "gold"
        use_azure         = True
    else:
        storage_options   = {}
        _silver_container = None
        _gold_container   = None
        use_azure         = False

    gold_frames = []

    for year in YEARS:
        for race_location in race_locations:
            silver_file = _get_silver_file(
                year, race_location,
                storage_options if use_azure else None,
                _silver_container
            )

            if not _silver_exists(silver_file, use_azure, storage_options):
                logging.info(f"Skipping {race_location} - {year}: silver file not found")
                continue

            try:
                df = pd.read_parquet(
                    silver_file,
                    storage_options=storage_options if use_azure else None
                )
                logging.info(f"Loaded silver file for {race_location} - {year}: shape={df.shape}")

                gold_df = preprocess_race_df(df, target_driver)
                gold_df = feature_engineering(gold_df)

                if gold_df.empty:
                    logging.info(f"No gold data for {target_driver} in {race_location} - {year}")
                    continue

                gold_frames.append(gold_df)

                del df, gold_df
                gc.collect()

            except Exception as e:
                logging.exception(f"Error processing gold for {race_location} - {year}: {e}")
                gc.collect()
                continue

    if not gold_frames:
        logging.warning(f"No gold files created for target_driver={target_driver}")
        return []

    final_df = pd.concat(gold_frames, ignore_index=True, sort=False)

    final_df = _add_social_info(
        final_df,
        storage_options if use_azure else None,
        _silver_container
    )
    final_df = _add_radio_info(
        final_df,
        storage_options if use_azure else None,
        _silver_container
    )

    out_file = write_gold_output(
        final_df,
        target_driver=target_driver,
        storage_options=storage_options if use_azure else None,
        gold_container=_gold_container,
    )

    return [out_file]


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
    run_gold_pipeline()
