import os
import logging
from typing import Optional, Tuple
import pandas as pd
import fsspec


# --------------------------- Configuration ------------------------------
# YEARS = [2024, 2025]
YEARS=[2024]
SESSION_TYPE = os.getenv("SESSION_TYPE", "R")
TARGET_DRIVER = os.getenv("TARGET_DRIVER", "LEC")

race_locations = [
    "Italian Grand Prix" 
    # , "Azerbaijan Grand Prix", "Singapore Grand Prix", "Mexico City Grand Prix",
    # "Brazilian Grand Prix", "Las Vegas Grand Prix", "Qatar Grand Prix", "Abu Dhabi Grand Prix",
    # "United States Grand Prix", "Australian Grand Prix", "Austrian Grand Prix", "Bahrain Grand Prix",
    # "Belgian Grand Prix", "British Grand Prix", "Canadian Grand Prix", "Chinese Grand Prix",
    # "Dutch Grand Prix", "Eifel Grand Prix", "Emilia Romagna Grand Prix", "French Grand Prix",
    # "German Grand Prix", "Hungarian Grand Prix", "Japanese Grand Prix", "Miami Grand Prix",
    # "Monaco Grand Prix", "Portuguese Grand Prix", "Russian Grand Prix", "Sakhir Grand Prix",
    # "Saudi Arabian Grand Prix", "Spanish Grand Prix", "Styrian Grand Prix", "São Paulo Grand Prix",
    # "Turkish Grand Prix", "Tuscan Grand Prix"
]

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


def get_filesystem(storage_account: str, storage_key: str):
    return fsspec.filesystem(
        "abfs",
        account_name=storage_account,
        account_key=storage_key
    )


def _abfs_path(container: str, relative_path: str) -> str:
    return f"abfs://{container}/{relative_path}"


def _to_seconds(series: pd.Series) -> pd.Series:
    td = pd.to_timedelta(series, errors="coerce")
    return td.dt.total_seconds()


def _get_bronze_paths(
    bronze_container: str,
    race_year: int,
    race_location: str,
    session_type: str,
    selected_driver: str
) -> Tuple[str, str, str]:
    base = f"{race_year}/{race_location}/{session_type}"

    telemetry_file = _abfs_path(
        bronze_container,
        f"{base}/{selected_driver}/{selected_driver}_telemetry.parquet"
    )
    overall_file = _abfs_path(
        bronze_container,
        f"{base}/{selected_driver}/{selected_driver}_laps.parquet"
    )
    weather_file = _abfs_path(
        bronze_container,
        f"{base}/weather.parquet"
    )

    return telemetry_file, overall_file, weather_file


def _exists(fs, abfs_path: str) -> bool:
    # fsspec abfs fs.exists expects container/path rather than abfs://...
    normalized = abfs_path.replace("abfs://", "", 1)
    return fs.exists(normalized)


def load_data_files(
    target_driver: str,
    storage_account: str,
    storage_key: str,
    bronze_container: str
) -> pd.DataFrame:
    race_id = 0
    prepared_data = pd.DataFrame()

    storage_options = get_storage_options(storage_account, storage_key)
    fs = get_filesystem(storage_account, storage_key)

    for race_year in YEARS:
        for race_location in race_locations:
            logging.info("=" * 60)
            logging.info(f"Processing {race_location} - {race_year}")
            logging.info("=" * 60)

            driver_data_dict = {}
            target_driver_times: Optional[pd.DataFrame] = None
            weather_data_dict: Optional[pd.DataFrame] = None

            for selected_driver in driver_abb.keys():
                try:
                    telemetry_file, overall_file, weather_file = _get_bronze_paths(
                        bronze_container=bronze_container,
                        race_year=race_year,
                        race_location=race_location,
                        session_type=SESSION_TYPE,
                        selected_driver=selected_driver
                    )

                    if not _exists(fs, telemetry_file) or not _exists(fs, overall_file):
                        logging.info(f"Skipping {selected_driver} - files not found")
                        continue

                    logging.info(f"Processing {selected_driver}...")

                    sector_data = pd.read_parquet(overall_file, storage_options=storage_options)
                    tel_data = pd.read_parquet(telemetry_file, storage_options=storage_options)

                    if weather_data_dict is None and _exists(fs, weather_file):
                        weather_data = pd.read_parquet(weather_file, storage_options=storage_options)
                        if "Time" in weather_data.columns:
                            weather_data["Time"] = _to_seconds(weather_data["Time"])
                            weather_data = weather_data.dropna(subset=["Time"])
                            weather_data_dict = weather_data

                    if "LapStartTime" not in sector_data.columns or "session_time" not in tel_data.columns:
                        logging.warning(f"Missing required time columns for {selected_driver}, skipping")
                        continue

                    sector_data["LapStartTime"] = _to_seconds(sector_data["LapStartTime"])
                    tel_data["session_time"] = _to_seconds(tel_data["session_time"])

                    tel_keep = [
                        "session_time", "x", "y", "z", "speed", "gear", "rpm", "drs", "brake", "position"
                    ]
                    sector_keep = [
                        "LapStartTime", "Driver", "DriverNumber", "LapNumber", "PitInTime", "PitOutTime",
                        "TyreLife", "Compound", "Team", "TrackStatus"
                    ]

                    tel_data = tel_data[[c for c in tel_keep if c in tel_data.columns]]
                    sector_data = sector_data[[c for c in sector_keep if c in sector_data.columns]]

                    tel_data = tel_data.dropna(subset=["session_time"])
                    sector_data = sector_data.dropna(subset=["LapStartTime"])

                    if tel_data.empty or sector_data.empty:
                        logging.info(f"Skipping {selected_driver} - empty telemetry or laps data")
                        continue

                    driver_race_details = pd.merge_asof(
                        left=tel_data.sort_values("session_time"),
                        right=sector_data.sort_values("LapStartTime"),
                        left_on="session_time",
                        right_on="LapStartTime"
                    )

                    if "LapStartTime" in driver_race_details.columns:
                        driver_race_details = driver_race_details.drop(columns=["LapStartTime"])

                    if selected_driver == target_driver:
                        target_driver_times = driver_race_details[["session_time"]].copy().reset_index(drop=True)

                    prefix = f"Target_{target_driver}_" if selected_driver == target_driver else f"{selected_driver}_"
                    driver_race_details = driver_race_details.add_prefix(prefix)

                    if selected_driver == target_driver:
                        driver_race_details["race_id"] = race_id
                        driver_race_details["race_year"] = race_year
                        driver_race_details["race_location"] = race_location

                    driver_race_details = driver_race_details.reset_index(drop=True)
                    driver_data_dict[selected_driver] = driver_race_details

                    logging.info(f"{selected_driver} data processed")

                except Exception as e:
                    logging.exception(f"Error processing {selected_driver}: {str(e)}")
                    continue

            if not driver_data_dict:
                logging.info(f"No data collected for {race_location} - {race_year}, skipping...")
                continue

            if target_driver not in driver_data_dict or target_driver_times is None:
                logging.info(f"Target driver {target_driver} not found for {race_location} - {race_year}, skipping...")
                continue

            if weather_data_dict is None or weather_data_dict.empty:
                logging.info(f"No weather data for {race_location} - {race_year}, skipping...")
                continue

            logging.info(f"Merging data for {len(driver_data_dict)} drivers...")

            race_data = target_driver_times.copy()

            for driver, driver_df in driver_data_dict.items():
                if driver == target_driver:
                    prefix = f"Target_{target_driver}_"
                    target_cols = [col for col in driver_df.columns if col != f"{prefix}session_time"]

                    race_data = pd.merge(
                        race_data,
                        driver_df[target_cols],
                        left_index=True,
                        right_index=True,
                        how="left"
                    )

                    race_data = race_data.sort_values("session_time")
                    weather_data_sorted = weather_data_dict.sort_values("Time")

                    race_data = pd.merge_asof(
                        race_data,
                        weather_data_sorted,
                        left_on="session_time",
                        right_on="Time",
                        direction="nearest"
                    )

                    if "Time" in race_data.columns:
                        race_data = race_data.drop(columns=["Time"])

                else:
                    session_col = f"{driver}_session_time"
                    if session_col not in driver_df.columns:
                        continue

                    other_cols = [col for col in driver_df.columns if col != session_col]
                    temp_df = driver_df[[session_col] + other_cols].copy().sort_values(session_col)

                    race_data = pd.merge_asof(
                        race_data.sort_values("session_time"),
                        temp_df,
                        left_on="session_time",
                        right_on=session_col,
                        direction="nearest"
                    )

                    if session_col in race_data.columns:
                        race_data = race_data.drop(columns=[session_col])

            logging.info(f"Race data merged: {race_data.shape[0]} rows, {race_data.shape[1]} columns")

            prepared_data = pd.concat([prepared_data, race_data], axis=0, ignore_index=True, sort=False)
            logging.info(f"Database updated: Total rows = {prepared_data.shape[0]}")

            race_id += 1

    return prepared_data


def run_silver_pipeline(
    target_driver: str = TARGET_DRIVER,
    storage_account: Optional[str] = None,
    storage_key: Optional[str] = None,
    bronze_container: Optional[str] = None,
    silver_container: Optional[str] = None
) -> str:
    storage_account = storage_account or os.getenv("STORAGE_ACCOUNT_NAME")
    storage_key = storage_key or os.getenv("STORAGE_ACCOUNT_KEY")
    bronze_container = bronze_container or os.getenv("BRONZE_CONTAINER", "bronze")
    silver_container = silver_container or os.getenv("SILVER_CONTAINER", "silver")

    if not storage_account:
        raise ValueError("Missing STORAGE_ACCOUNT_NAME")
    if not storage_key:
        raise ValueError("Missing STORAGE_ACCOUNT_KEY")
    if not bronze_container:
        raise ValueError("Missing BRONZE_CONTAINER")
    if not silver_container:
        raise ValueError("Missing SILVER_CONTAINER")

    data = load_data_files(
        target_driver=target_driver,
        storage_account=storage_account,
        storage_key=storage_key,
        bronze_container=bronze_container
    )

    if data.empty:
        raise ValueError(f"No silver data prepared for target_driver={target_driver}")

    storage_options = get_storage_options(storage_account, storage_key)
    output_file = _abfs_path(silver_container, f"{target_driver}_silver.parquet")

    data.to_parquet(output_file, index=False, storage_options=storage_options)

    logging.info("=" * 60)
    logging.info(f"FINAL DATASET: {data.shape[0]} rows × {data.shape[1]} columns")
    logging.info("=" * 60)
    logging.info(f"Saved silver data to: {output_file}")

    return output_file


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run_silver_pipeline()