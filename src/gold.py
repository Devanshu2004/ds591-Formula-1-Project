import os
import logging
from typing import Optional
import pandas as pd


TARGET_DRIVER = os.getenv("TARGET_DRIVER", "LEC")

driver_abb = {
    "ALB": ["Alexander Albon", "Alex Albon"],
    "ALO": ["Fernando Alonso"],
    "ANT": ["Kimi Antonelli"],
    "BEA": ["Oliver Bearman"],
    "BOR": ["Gabriel Bortoleto"],
    "BOT": ["Valtteri Bottas"],
    "COL": ["Franco Colapinto"],
    "DEV": ["Nyck De Vries"],
    "DOO": ["Jack Doohan"],
    "ERI": ["Marcus Ericsson"],
    "FIT": ["Pietro Fittipaldi"],
    "GAS": ["Pierre Gasly"],
    "GIO": ["Antonio Giovinazzi"],
    "GRO": ["Romain Grosjean"],
    "HAD": ["Isack Hadjar"],
    "HAM": ["Lewis Hamilton"],
    "HAR": ["Brendon Hartley"],
    "HUL": ["Nico Hülkenberg"],
    "KUB": ["Robert Kubica"],
    "KVY": ["Daniil Kvyat"],
    "LAT": ["Nicholas Latifi"],
    "LAW": ["Liam Lawson"],
    "LEC": ["Charles Leclerc"],
    "MAG": ["Kevin Magnussen"],
    "MSC": ["Mick Schumacher"],
    "MAZ": ["Nikita Mazepin"],
    "NOR": ["Lando Norris"],
    "OCO": ["Esteban Ocon"],
    "PER": ["Sergio Perez"],
    "PIA": ["Oscar Piastri"],
    "RAI": ["Kimi Räikkönen"],
    "RIC": ["Daniel Ricciardo"],
    "RUS": ["George Russell"],
    "SAI": ["Carlos Sainz", "Carlos Sainz Jr."],
    "SAR": ["Logan Sargeant"],
    "SIR": ["Sergey Sirotkin"],
    "STR": ["Lance Stroll"],
    "TSU": ["Yuki Tsunoda"],
    "VAN": ["Stoffel Vandoorne"],
    "VER": ["Max Verstappen"],
    "VET": ["Sebastian Vettel"],
    "ZHO": ["Zhou Guanyu"]
}


def get_storage_options(storage_account: str, storage_key: str) -> dict:
    return {
        "account_name": storage_account,
        "account_key": storage_key
    }


def _abfs_path(container: str, relative_path: str) -> str:
    return f"abfs://{container}/{relative_path}"


def _safe_get(row, attr_name, default=None):
    return getattr(row, attr_name, default)


def preprocess_df(df: pd.DataFrame, target_driver: str) -> pd.DataFrame:
    var_attributes = [
        "x", "y", "z", "speed", "gear", "rpm", "drs", "brake", "position", "Driver", "DriverNumber",
        "LapNumber", "TyreLife", "Compound", "Team"
    ]
    const_attributes = [
        "race_id", "race_year", "race_location",
        "AirTemp", "Humidity", "Pressure", "Rainfall", "TrackTemp",
        "WindDirection", "WindSpeed"
    ]

    prefix = "Target"
    all_row_data = []

    for row in df.itertuples(index=False):
        track_append_data = {}
        target_driver_attrs = {}

        for value in var_attributes:
            target_driver_attrs[value] = f"{prefix}_{target_driver}_{value}"

        target_driver_attrs["TrackStatus"] = f"{prefix}_{target_driver}_TrackStatus"

        for value in const_attributes:
            target_driver_attrs[value] = value

        track_append_data["session_time"] = _safe_get(row, "session_time")

        track_append_data["target_x"] = _safe_get(row, target_driver_attrs["x"])
        track_append_data["target_y"] = _safe_get(row, target_driver_attrs["y"])
        track_append_data["target_z"] = _safe_get(row, target_driver_attrs["z"])
        track_append_data["target_speed"] = _safe_get(row, target_driver_attrs["speed"])
        track_append_data["target_gear"] = _safe_get(row, target_driver_attrs["gear"])
        track_append_data["target_rpm"] = _safe_get(row, target_driver_attrs["rpm"])
        track_append_data["target_drs"] = _safe_get(row, target_driver_attrs["drs"])
        track_append_data["target_brake"] = _safe_get(row, target_driver_attrs["brake"])
        track_append_data["target_pos"] = _safe_get(row, target_driver_attrs["position"])
        track_append_data["target_driver"] = _safe_get(row, target_driver_attrs["Driver"])
        track_append_data["target_driver_number"] = _safe_get(row, target_driver_attrs["DriverNumber"])
        track_append_data["target_lap_number"] = _safe_get(row, target_driver_attrs["LapNumber"])
        track_append_data["target_tyre_life"] = _safe_get(row, target_driver_attrs["TyreLife"])
        track_append_data["target_compound"] = _safe_get(row, target_driver_attrs["Compound"])
        track_append_data["target_team"] = _safe_get(row, target_driver_attrs["Team"])
        track_append_data["track_status"] = _safe_get(row, target_driver_attrs["TrackStatus"])

        track_append_data["race_id"] = _safe_get(row, target_driver_attrs["race_id"])
        track_append_data["race_year"] = _safe_get(row, target_driver_attrs["race_year"])
        track_append_data["race_location"] = _safe_get(row, target_driver_attrs["race_location"])
        track_append_data["air_temp"] = _safe_get(row, target_driver_attrs["AirTemp"])
        track_append_data["humidity"] = _safe_get(row, target_driver_attrs["Humidity"])
        track_append_data["pressure"] = _safe_get(row, target_driver_attrs["Pressure"])
        track_append_data["rainfall"] = _safe_get(row, target_driver_attrs["Rainfall"])
        track_append_data["track_temp"] = _safe_get(row, target_driver_attrs["TrackTemp"])
        track_append_data["wind_direction"] = _safe_get(row, target_driver_attrs["WindDirection"])
        track_append_data["wind_speed"] = _safe_get(row, target_driver_attrs["WindSpeed"])

        target_pos = _safe_get(row, target_driver_attrs["position"])

        try:
            target_pos = int(target_pos)
        except Exception:
            all_row_data.append(track_append_data)
            continue

        driver_ahead = target_pos - 1
        driver_behind = target_pos + 1

        found_driver_ahead = False
        found_driver_behind = False

        for driver in driver_abb.keys():
            try:
                selected_driver_pos = _safe_get(row, f"{driver}_position")
                if selected_driver_pos is None:
                    continue

                selected_driver_pos = int(selected_driver_pos)

                if selected_driver_pos == driver_ahead:
                    track_append_data["driver_ahead_x"] = _safe_get(row, f"{driver}_x")
                    track_append_data["driver_ahead_y"] = _safe_get(row, f"{driver}_y")
                    track_append_data["driver_ahead_z"] = _safe_get(row, f"{driver}_z")
                    track_append_data["driver_ahead_speed"] = _safe_get(row, f"{driver}_speed")
                    track_append_data["driver_ahead_gear"] = _safe_get(row, f"{driver}_gear")
                    track_append_data["driver_ahead_rpm"] = _safe_get(row, f"{driver}_rpm")
                    track_append_data["driver_ahead_drs"] = _safe_get(row, f"{driver}_drs")
                    track_append_data["driver_ahead_brake"] = _safe_get(row, f"{driver}_brake")
                    track_append_data["driver_ahead_pos"] = _safe_get(row, f"{driver}_position")
                    track_append_data["driver_ahead"] = _safe_get(row, f"{driver}_Driver")
                    track_append_data["driver_ahead_number"] = _safe_get(row, f"{driver}_DriverNumber")
                    track_append_data["driver_ahead_lap_number"] = _safe_get(row, f"{driver}_LapNumber")
                    track_append_data["driver_ahead_tyre_life"] = _safe_get(row, f"{driver}_TyreLife")
                    track_append_data["driver_ahead_compound"] = _safe_get(row, f"{driver}_Compound")
                    track_append_data["driver_ahead_team"] = _safe_get(row, f"{driver}_Team")
                    found_driver_ahead = True

                if selected_driver_pos == driver_behind:
                    track_append_data["driver_behind_x"] = _safe_get(row, f"{driver}_x")
                    track_append_data["driver_behind_y"] = _safe_get(row, f"{driver}_y")
                    track_append_data["driver_behind_z"] = _safe_get(row, f"{driver}_z")
                    track_append_data["driver_behind_speed"] = _safe_get(row, f"{driver}_speed")
                    track_append_data["driver_behind_gear"] = _safe_get(row, f"{driver}_gear")
                    track_append_data["driver_behind_rpm"] = _safe_get(row, f"{driver}_rpm")
                    track_append_data["driver_behind_drs"] = _safe_get(row, f"{driver}_drs")
                    track_append_data["driver_behind_brake"] = _safe_get(row, f"{driver}_brake")
                    track_append_data["driver_behind_pos"] = _safe_get(row, f"{driver}_position")
                    track_append_data["driver_behind"] = _safe_get(row, f"{driver}_Driver")
                    track_append_data["driver_behind_number"] = _safe_get(row, f"{driver}_DriverNumber")
                    track_append_data["driver_behind_lap_number"] = _safe_get(row, f"{driver}_LapNumber")
                    track_append_data["driver_behind_tyre_life"] = _safe_get(row, f"{driver}_TyreLife")
                    track_append_data["driver_behind_compound"] = _safe_get(row, f"{driver}_Compound")
                    track_append_data["driver_behind_team"] = _safe_get(row, f"{driver}_Team")
                    found_driver_behind = True

            except Exception:
                continue

        try:
            closest_driver_ahead = None
            closest_driver_behind = None
            best_ahead_dist = 999
            best_behind_dist = 999

            if not found_driver_ahead:
                for driver in driver_abb.keys():
                    try:
                        selected_driver_pos = _safe_get(row, f"{driver}_position")
                        if selected_driver_pos is None:
                            continue
                        selected_driver_pos = int(selected_driver_pos)

                        rank_dist = target_pos - selected_driver_pos
                        if 0 < rank_dist < best_ahead_dist:
                            best_ahead_dist = rank_dist
                            closest_driver_ahead = driver
                    except Exception:
                        continue

                if closest_driver_ahead:
                    d = closest_driver_ahead
                    track_append_data["driver_ahead_x"] = _safe_get(row, f"{d}_x")
                    track_append_data["driver_ahead_y"] = _safe_get(row, f"{d}_y")
                    track_append_data["driver_ahead_z"] = _safe_get(row, f"{d}_z")
                    track_append_data["driver_ahead_speed"] = _safe_get(row, f"{d}_speed")
                    track_append_data["driver_ahead_gear"] = _safe_get(row, f"{d}_gear")
                    track_append_data["driver_ahead_rpm"] = _safe_get(row, f"{d}_rpm")
                    track_append_data["driver_ahead_drs"] = _safe_get(row, f"{d}_drs")
                    track_append_data["driver_ahead_brake"] = _safe_get(row, f"{d}_brake")
                    track_append_data["driver_ahead_pos"] = _safe_get(row, f"{d}_position")
                    track_append_data["driver_ahead"] = _safe_get(row, f"{d}_Driver")
                    track_append_data["driver_ahead_number"] = _safe_get(row, f"{d}_DriverNumber")
                    track_append_data["driver_ahead_lap_number"] = _safe_get(row, f"{d}_LapNumber")
                    track_append_data["driver_ahead_tyre_life"] = _safe_get(row, f"{d}_TyreLife")
                    track_append_data["driver_ahead_compound"] = _safe_get(row, f"{d}_Compound")
                    track_append_data["driver_ahead_team"] = _safe_get(row, f"{d}_Team")

            if not found_driver_behind:
                for driver in driver_abb.keys():
                    try:
                        selected_driver_pos = _safe_get(row, f"{driver}_position")
                        if selected_driver_pos is None:
                            continue
                        selected_driver_pos = int(selected_driver_pos)

                        rank_dist = selected_driver_pos - target_pos
                        if 0 < rank_dist < best_behind_dist:
                            best_behind_dist = rank_dist
                            closest_driver_behind = driver
                    except Exception:
                        continue

                if closest_driver_behind:
                    d = closest_driver_behind
                    track_append_data["driver_behind_x"] = _safe_get(row, f"{d}_x")
                    track_append_data["driver_behind_y"] = _safe_get(row, f"{d}_y")
                    track_append_data["driver_behind_z"] = _safe_get(row, f"{d}_z")
                    track_append_data["driver_behind_speed"] = _safe_get(row, f"{d}_speed")
                    track_append_data["driver_behind_gear"] = _safe_get(row, f"{d}_gear")
                    track_append_data["driver_behind_rpm"] = _safe_get(row, f"{d}_rpm")
                    track_append_data["driver_behind_drs"] = _safe_get(row, f"{d}_drs")
                    track_append_data["driver_behind_brake"] = _safe_get(row, f"{d}_brake")
                    track_append_data["driver_behind_pos"] = _safe_get(row, f"{d}_position")
                    track_append_data["driver_behind"] = _safe_get(row, f"{d}_Driver")
                    track_append_data["driver_behind_number"] = _safe_get(row, f"{d}_DriverNumber")
                    track_append_data["driver_behind_lap_number"] = _safe_get(row, f"{d}_LapNumber")
                    track_append_data["driver_behind_tyre_life"] = _safe_get(row, f"{d}_TyreLife")
                    track_append_data["driver_behind_compound"] = _safe_get(row, f"{d}_Compound")
                    track_append_data["driver_behind_team"] = _safe_get(row, f"{d}_Team")

        except Exception:
            all_row_data.append(track_append_data)
            continue

        all_row_data.append(track_append_data)

    return pd.DataFrame(all_row_data)


def run_gold_pipeline(
    target_driver: str = TARGET_DRIVER,
    storage_account: Optional[str] = None,
    storage_key: Optional[str] = None,
    silver_container: Optional[str] = None,
    gold_container: Optional[str] = None
) -> str:
    storage_account = storage_account or os.getenv("STORAGE_ACCOUNT_NAME")
    storage_key = storage_key or os.getenv("STORAGE_ACCOUNT_KEY")
    silver_container = silver_container or os.getenv("SILVER_CONTAINER", "silver")
    gold_container = gold_container or os.getenv("GOLD_CONTAINER", "gold")

    if not storage_account:
        raise ValueError("Missing STORAGE_ACCOUNT_NAME")
    if not storage_key:
        raise ValueError("Missing STORAGE_ACCOUNT_KEY")
    if not silver_container:
        raise ValueError("Missing SILVER_CONTAINER")
    if not gold_container:
        raise ValueError("Missing GOLD_CONTAINER")

    storage_options = get_storage_options(storage_account, storage_key)

    silver_file = _abfs_path(silver_container, f"{target_driver}_silver.parquet")
    gold_file = _abfs_path(gold_container, f"{target_driver}_gold.parquet")

    df = pd.read_parquet(silver_file, storage_options=storage_options)
    gold_df = preprocess_df(df, target_driver)

    if gold_df.empty:
        raise ValueError(f"Gold dataframe is empty for target_driver={target_driver}")

    gold_df.to_parquet(gold_file, index=False, storage_options=storage_options)

    logging.info(f"Saved gold data to: {gold_file}")
    logging.info(f"Final gold shape: {gold_df.shape}")

    return gold_file


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run_gold_pipeline()