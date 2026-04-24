import os
import logging
import gc
from pathlib import Path
from typing import List
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import pandas as pd
import numpy as np

# --------------------------- Configuration ------------------------------
DATA_ROOT = Path(os.getenv("DATA_ROOT", "data"))

SILVER_PATH = DATA_ROOT / "silver"
GOLD_PATH = DATA_ROOT / "gold"

YEARS_ENV = os.getenv("YEARS", "2024,2025")
YEARS = [int(y.strip()) for y in YEARS_ENV.split(",") if y.strip()]

TARGET_DRIVER = os.getenv("TARGET_DRIVER", "LEC")
SESSION_TYPE = os.getenv("SESSION_TYPE", "R")

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


def _get_silver_file(race_year: int, race_location: str) -> Path:
    safe_location = race_location.replace(" ", "_")
    print(SILVER_PATH / str(race_year) / SESSION_TYPE / f"{safe_location}.parquet")
    return SILVER_PATH / str(race_year) / SESSION_TYPE / f"{safe_location}.parquet"

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

    # ---------------- TARGET ----------------
    target_df["target_pos_lookup"] = pd.to_numeric(target_df["position"], errors="coerce")

    target_df = target_df.rename(columns={
        "x": "target_x",
        "y": "target_y",
        "z": "target_z",
        "speed": "target_speed",
        "gear": "target_gear",
        "brake": "target_brake",
        "position": "target_pos",
        "relative_distance": "target_relative_distance",
        "Driver": "target_driver",
        "DriverNumber": "target_driver_number",
        "LapNumber": "target_lap_number",
        "TyreLife": "target_tyre_life",
        "Compound": "target_compound",
        "Team": "target_team",
        "TrackStatus": "track_status",
        "AirTemp": "air_temp",
        "Humidity": "humidity",
        "Pressure": "pressure",
        "Rainfall": "rainfall",
        "TrackTemp": "track_temp",
        "WindDirection": "wind_direction",
        "WindSpeed": "wind_speed",
    })

    target_cols = [
        "target_x", "target_y", "target_z", "target_speed", "target_gear", 
        "target_brake", "target_pos", "target_relative_distance", "target_driver", "target_driver_number",
        "target_lap_number", "target_tyre_life", "target_compound", "target_team",
        "track_status", "air_temp", "humidity", "pressure", "rainfall",
        "track_temp", "wind_direction", "wind_speed"
    ]
    for col in target_cols:
        if col not in target_df.columns:
            target_df[col] = pd.NA

    target_df["ahead_pos_lookup"] = target_df["target_pos_lookup"] - 1
    target_df["behind_pos_lookup"] = target_df["target_pos_lookup"] + 1

    base_cols = [
        "session_time", "race_id", "race_date", "race_year", "race_location",
        "target_x", "target_y", "target_z",
        "target_speed", "target_gear", "target_brake",
        "target_pos", "target_relative_distance", "target_driver", "target_driver_number",
        "target_lap_number", "target_tyre_life", "target_compound", "target_team",
        "track_status",
        "air_temp", "humidity", "pressure", "rainfall", "track_temp", "wind_direction", "wind_speed",
        "target_pos_lookup", "ahead_pos_lookup", "behind_pos_lookup"
    ]
    base_cols = [c for c in base_cols if c in target_df.columns]
    target_df = target_df[base_cols].copy()

    # ---------------- OTHERS ----------------
    others = df.copy()
    others["position_lookup"] = pd.to_numeric(others["position"], errors="coerce")

    join_cols = ["race_id", "race_year", "race_location", "session_time"]

    # ---------------- AHEAD ----------------
    ahead_source_cols = [
        "race_id", "race_year", "race_location", "session_time", "position_lookup",
        "speed", "gear", "brake", "position", "relative_distance",
        "Driver", "DriverNumber", "LapNumber", "TyreLife", "Compound", "Team"
    ]
    ahead_source_cols = [c for c in ahead_source_cols if c in others.columns]

    ahead_df = others[ahead_source_cols].rename(columns={
        "position_lookup": "ahead_pos_lookup",
        "speed": "driver_ahead_speed",
        "gear": "driver_ahead_gear",
        "brake": "driver_ahead_brake",
        "position": "driver_ahead_pos",
        "relative_distance": "driver_ahead_relative_distance",
        "Driver": "driver_ahead",
        "DriverNumber": "driver_ahead_number",
        "LapNumber": "driver_ahead_lap_number",
        "TyreLife": "driver_ahead_tyre_life",
        "Compound": "driver_ahead_compound",
        "Team": "driver_ahead_team",
    })

    ahead_keep = [
        "race_id", "race_year", "race_location", "session_time", "ahead_pos_lookup",
        "driver_ahead_speed", "driver_ahead_gear", 
        "driver_ahead_brake", "driver_ahead_pos", "driver_ahead_relative_distance",
        "driver_ahead", "driver_ahead_number", "driver_ahead_lap_number",
        "driver_ahead_tyre_life", "driver_ahead_compound", "driver_ahead_team"
    ]
    ahead_keep = [c for c in ahead_keep if c in ahead_df.columns]
    ahead_df = ahead_df[ahead_keep]

    gold_df = target_df.merge(
        ahead_df,
        on=join_cols + ["ahead_pos_lookup"],
        how="left"
    )

    # ---------------- BEHIND ----------------
    behind_source_cols = [
        "race_id", "race_year", "race_location", "session_time", "position_lookup",
        "speed", "gear", "brake", "position", "relative_distance",
        "Driver", "DriverNumber", "LapNumber", "TyreLife", "Compound", "Team"
    ]
    behind_source_cols = [c for c in behind_source_cols if c in others.columns]

    behind_df = others[behind_source_cols].rename(columns={
        "position_lookup": "behind_pos_lookup",
        "speed": "driver_behind_speed",
        "gear": "driver_behind_gear",
        "brake": "driver_behind_brake",
        "position": "driver_behind_pos",
        "relative_distance": "driver_behind_relative_distance",
        "Driver": "driver_behind",
        "DriverNumber": "driver_behind_number",
        "LapNumber": "driver_behind_lap_number",
        "TyreLife": "driver_behind_tyre_life",
        "Compound": "driver_behind_compound",
        "Team": "driver_behind_team",
    })

    behind_keep = [
        "race_id", "race_year", "race_location", "session_time", "behind_pos_lookup",
        "driver_behind_speed", "driver_behind_gear", 
        "driver_behind_brake", "driver_behind_pos", "driver_behind_relative_distance",
        "driver_behind", "driver_behind_number", "driver_behind_lap_number",
        "driver_behind_tyre_life", "driver_behind_compound", "driver_behind_team"
    ]
    behind_keep = [c for c in behind_keep if c in behind_df.columns]
    behind_df = behind_df[behind_keep]

    gold_df = gold_df.merge(
        behind_df,
        on=join_cols + ["behind_pos_lookup"],
        how="left"
    )

    logging.info(f"[{target_driver}] After merge shape: {gold_df.shape}")

    # ---------------- FALLBACK LOGIC ----------------
    others_min_cols = [
        "race_id", "race_year", "race_location", "session_time", "position_lookup",
        "speed", "gear", "brake", "position", "relative_distance",
        "Driver", "DriverNumber", "LapNumber", "TyreLife", "Compound", "Team"
    ]
    others_min_cols = [c for c in others_min_cols if c in others.columns]

    others_min = (
        others[others_min_cols]
        .dropna(subset=["position_lookup"])
        .copy()
    )

    # Build lookup once: each timestamp maps to sorted drivers at that moment
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

        # Ahead fallback
        if pd.isna(row_dict.get("driver_ahead_pos")):
            ahead_candidates = same_time[same_time["position_lookup"] < target_pos_num]

            if not ahead_candidates.empty:
                best = ahead_candidates.iloc[-1]  # closest position ahead

                row_dict["driver_ahead_speed"] = best.get("speed")
                row_dict["driver_ahead_gear"] = best.get("gear")
                row_dict["driver_ahead_brake"] = best.get("brake")
                row_dict["driver_ahead_pos"] = best.get("position")
                row_dict["driver_ahead_relative_distance"] = best.get("relative_distance")
                row_dict["driver_ahead"] = best.get("Driver")
                row_dict["driver_ahead_number"] = best.get("DriverNumber")
                row_dict["driver_ahead_lap_number"] = best.get("LapNumber")
                row_dict["driver_ahead_tyre_life"] = best.get("TyreLife")
                row_dict["driver_ahead_compound"] = best.get("Compound")
                row_dict["driver_ahead_team"] = best.get("Team")

        # Behind fallback
        if pd.isna(row_dict.get("driver_behind_pos")):
            behind_candidates = same_time[same_time["position_lookup"] > target_pos_num]

            if not behind_candidates.empty:
                best = behind_candidates.iloc[0]  # closest position behind

                row_dict["driver_behind_speed"] = best.get("speed")
                row_dict["driver_behind_gear"] = best.get("gear")
                row_dict["driver_behind_brake"] = best.get("brake")
                row_dict["driver_behind_pos"] = best.get("position")
                row_dict["driver_behind_relative_distance"] = best.get("relative_distance")
                row_dict["driver_behind"] = best.get("Driver")
                row_dict["driver_behind_number"] = best.get("DriverNumber")
                row_dict["driver_behind_lap_number"] = best.get("LapNumber")
                row_dict["driver_behind_tyre_life"] = best.get("TyreLife")
                row_dict["driver_behind_compound"] = best.get("Compound")
                row_dict["driver_behind_team"] = best.get("Team")

        result_rows.append(row_dict)

    result_df = pd.DataFrame(result_rows)

    result_df["distance_ahead_target"] = (
        pd.to_numeric(result_df["driver_ahead_relative_distance"], errors="coerce")
        - pd.to_numeric(result_df["target_relative_distance"], errors="coerce")
    )

    result_df["distance_target_behind"] = (
        pd.to_numeric(result_df["target_relative_distance"], errors="coerce")
        - pd.to_numeric(result_df["driver_behind_relative_distance"], errors="coerce")
    )

    result_df = result_df.drop(
        columns=["target_pos_lookup", "ahead_pos_lookup", "behind_pos_lookup", "driver_behind_relative_distance", "driver_ahead_relative_distance"],
        errors="ignore"
    )

    logging.info(f"[{target_driver}] Target rows after preprocess: {len(target_df)}")

    return result_df


def write_gold_output(gold_df: pd.DataFrame, race_year: int, race_location: str, target_driver: str) -> Path:
    out_dir = GOLD_PATH / str(race_year) / SESSION_TYPE
    out_dir.mkdir(parents=True, exist_ok=True)

    safe_location = race_location.replace(" ", "_")
    out_file = out_dir / f"{target_driver}_{safe_location}_gold.parquet"
    gold_df.to_parquet(out_file, index=False)

    logging.info(f"Saved gold data to: {out_file}")
    logging.info(f"Final gold shape: {gold_df.shape}")
    return out_file


def feature_engineering(gold_df: pd.DataFrame) -> pd.DataFrame:
    df = gold_df.copy()

    # 1. Convert boolean columns to integer
    bool_cols = [
        "target_brake",
        "rainfall",
        "driver_ahead_brake",
        "driver_behind_brake"
    ]

    for col in bool_cols:
        if col in df.columns:
            df[col] = df[col].fillna(False).astype(int)

    # 2. Convert numeric columns
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
        "distance_ahead_target",
        "distance_target_behind"
    ]

    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # 3. Driver ahead/behind null indicators for model training
    if "distance_ahead_target" in df.columns:
        df["has_driver_ahead"] = df["distance_ahead_target"].notna().astype(int)

    if "distance_target_behind" in df.columns:
        df["has_driver_behind"] = df["distance_target_behind"].notna().astype(int)


    # # 4. Fill missing ahead/behind values (-1 means no valid adjacent driver available)
    # relational_cols = [
    #     "driver_ahead_speed", "driver_ahead_gear", 
    #     "driver_ahead_pos", "driver_ahead_lap_number", "driver_ahead_tyre_life",
    #     "distance_ahead_target",

    #     "driver_behind_speed", "driver_behind_gear", 
    #     "driver_behind_pos", "driver_behind_lap_number", "driver_behind_tyre_life",
    #     "distance_target_behind"
    # ]

    # for col in relational_cols:
    #     if col in df.columns:
    #         df[col] = df[col].fillna('NA')

    # 5. One-hot encoding categorical columns
    categorical_cols = [
        "target_compound",
        "target_team",
        "track_status",
        "driver_ahead_compound",
        "driver_ahead_team",
        "driver_behind_compound",
        "driver_behind_team"
    ]

    categorical_cols = [c for c in categorical_cols if c in df.columns]

    for col in categorical_cols:
        df[col] = df[col].fillna("Unknown").astype(str)

    df = pd.get_dummies(df, columns=categorical_cols, dummy_na=False)

    # Convert wind direction degrees to sin/cos before scaling
    radians = np.deg2rad(pd.to_numeric(df["wind_direction"], errors="coerce"))
    df["wind_direction_sin"] = np.sin(radians)
    df["wind_direction_cos"] = np.cos(radians)
    df = df.drop(columns=["wind_direction"], errors="ignore")

    # One-hot encode gear, as it is discrete 
    df = pd.get_dummies(df, columns=["target_gear", "driver_ahead_gear", "driver_behind_gear"], dummy_na=False)

    logging.info(f"After feature engineering shape: {gold_df.shape}")

    return df


def run_gold_pipeline(target_driver: str = TARGET_DRIVER) -> List[str]:
    gold_frames = []

    for year in YEARS:
        for race_location in race_locations:
            silver_file = _get_silver_file(year, race_location)

            if not silver_file.exists():
                logging.info(f"Skipping {race_location} - {year}: silver file not found")
                continue

            try:
                df = pd.read_parquet(silver_file)
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

    final_gold_df = pd.concat(gold_frames, ignore_index=True, sort=False)

    out_dir = GOLD_PATH / SESSION_TYPE
    out_dir.mkdir(parents=True, exist_ok=True)

    out_file = out_dir / f"{target_driver}_gold.parquet"
    final_gold_df.to_parquet(out_file, index=False)

    logging.info(f"Saved final gold file: {out_file}")
    logging.info(f"Final combined gold shape: {final_gold_df.shape}")

    return [str(out_file)]

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
    run_gold_pipeline()