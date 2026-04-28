import os
import logging
import gc
import re
from typing import List, Optional, Tuple
import pandas as pd
import fsspec


# --------------------------- Configuration ------------------------------
YEARS_ENV = os.getenv("YEARS", "2024,2025")
YEARS = [int(y.strip()) for y in YEARS_ENV.split(",") if y.strip()]

SESSION_TYPE = "R"

RACE_LOCATIONS = [
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

RACE_CALENDAR = {
    2024: {
        "Bahrain Grand Prix": "2024-03-02",
        "Saudi Arabian Grand Prix": "2024-03-09",
        "Australian Grand Prix": "2024-03-24",
        "Japanese Grand Prix": "2024-04-07",
        "Chinese Grand Prix": "2024-04-21",
        "Miami Grand Prix": "2024-05-05",
        "Emilia Romagna Grand Prix": "2024-05-19",
        "Monaco Grand Prix": "2024-05-26",
        "Canadian Grand Prix": "2024-06-09",
        "Spanish Grand Prix": "2024-06-23",
        "Austrian Grand Prix": "2024-06-30",
        "British Grand Prix": "2024-07-07",
        "Hungarian Grand Prix": "2024-07-21",
        "Belgian Grand Prix": "2024-07-28",
        "Dutch Grand Prix": "2024-08-25",
        "Italian Grand Prix": "2024-09-01",
        "Azerbaijan Grand Prix": "2024-09-15",
        "Singapore Grand Prix": "2024-09-22",
        "United States Grand Prix": "2024-10-20",
        "Mexico City Grand Prix": "2024-10-27",
        "São Paulo Grand Prix": "2024-11-03",
        "Las Vegas Grand Prix": "2024-11-23",
        "Qatar Grand Prix": "2024-12-01",
        "Abu Dhabi Grand Prix": "2024-12-08",
    },
    2025: {
        "Australian Grand Prix": "2025-03-16",
        "Chinese Grand Prix": "2025-03-23",
        "Japanese Grand Prix": "2025-04-06",
        "Bahrain Grand Prix": "2025-04-13",
        "Saudi Arabian Grand Prix": "2025-04-20",
        "Miami Grand Prix": "2025-05-04",
        "Emilia Romagna Grand Prix": "2025-05-18",
        "Monaco Grand Prix": "2025-05-25",
        "Spanish Grand Prix": "2025-06-01",
        "Canadian Grand Prix": "2025-06-15",
        "Austrian Grand Prix": "2025-06-29",
        "British Grand Prix": "2025-07-06",
        "Belgian Grand Prix": "2025-07-27",
        "Hungarian Grand Prix": "2025-08-03",
        "Dutch Grand Prix": "2025-08-31",
        "Italian Grand Prix": "2025-09-07",
        "Azerbaijan Grand Prix": "2025-09-21",
        "Singapore Grand Prix": "2025-10-05",
        "United States Grand Prix": "2025-10-19",
        "Mexico City Grand Prix": "2025-10-26",
        "São Paulo Grand Prix": "2025-11-09",
        "Las Vegas Grand Prix": "2025-11-22",
        "Qatar Grand Prix": "2025-11-30",
        "Abu Dhabi Grand Prix": "2025-12-07",
    },
    2026: {
        "Australian Grand Prix": "2026-03-08",
        "Chinese Grand Prix": "2026-03-15",
        "Japanese Grand Prix": "2026-03-29",
        "Miami Grand Prix": "2026-05-03",
        "Canadian Grand Prix": "2026-05-24",
        "Monaco Grand Prix": "2026-06-07",
        "Spanish Grand Prix": "2026-06-14",
        "Austrian Grand Prix": "2026-06-28",
        "British Grand Prix": "2026-07-05",
        "Belgian Grand Prix": "2026-07-19",
        "Hungarian Grand Prix": "2026-07-26",
        "Dutch Grand Prix": "2026-08-23",
        "Italian Grand Prix": "2026-09-06",
        "Azerbaijan Grand Prix": "2026-09-27",
        "Singapore Grand Prix": "2026-10-11",
        "United States Grand Prix": "2026-10-25",
        "Mexico City Grand Prix": "2026-11-01",
        "São Paulo Grand Prix": "2026-11-08",
        "Las Vegas Grand Prix": "2026-11-21",
        "Qatar Grand Prix": "2026-11-29",
        "Abu Dhabi Grand Prix": "2026-12-06",
        "Barcelona-Catalunya Grand Prix": "2026-06-14",
        "Spain Grand Prix": "2026-09-13",
        "Bahrain Grand Prix": "2026-04-12",
        "Saudi Arabian Grand Prix": "2026-04-19",
    }
}

FLAT_RACE_CALENDAR = {
    (year, loc): date
    for year, races in RACE_CALENDAR.items()
    for loc, date in races.items()
}

TELEMETRY = [
    "session_time", "x", "y", "z", "speed", "gear", "rpm", "drs", "brake", "position", "relative_distance"
]

LAP = [
    "LapStartTime", "Driver", "DriverNumber", "LapNumber", "TyreLife", "Compound", "Team", "TrackStatus"
]

WEATHER = [
    "Time", "AirTemp", "Humidity", "Pressure", "Rainfall",
    "TrackTemp", "WindDirection", "WindSpeed"
]


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


def _exists(fs, abfs_path: str) -> bool:
    normalized = abfs_path.replace("abfs://", "", 1)
    return fs.exists(normalized)


def _to_seconds(series: pd.Series) -> pd.Series:
    td = pd.to_timedelta(series, errors="coerce")
    return td.dt.total_seconds()


def _get_bronze_paths(
    bronze_container: str,
    race_year: int,
    race_location: str,
    session_type: str,
    driver: str
) -> Tuple[str, str, str]:
    base = f"{race_year}/{race_location}/{session_type}"
    return (
        _abfs_path(bronze_container, f"{base}/{driver}/{driver}_telemetry.parquet"),
        _abfs_path(bronze_container, f"{base}/{driver}/{driver}_laps.parquet"),
        _abfs_path(bronze_container, f"{base}/weather.parquet")
    )


def add_race_date(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["race_date"] = list(zip(df["race_year"], df["race_location"]))
    df["race_date"] = df["race_date"].map(FLAT_RACE_CALENDAR)
    df["race_date"] = pd.to_datetime(df["race_date"], errors="coerce")
    return df


def clean_gear_column(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["gear"] = pd.to_numeric(df["gear"], errors="coerce")
    df.loc[~df["gear"].between(0, 8), "gear"] = pd.NA
    df["gear"] = df["gear"].ffill().bfill()
    return df


def _prepare_weather(weather_file: str, storage_options: dict) -> pd.DataFrame:
    try:
        weather_data = pd.read_parquet(
            weather_file,
            columns=[c for c in WEATHER],
            storage_options=storage_options
        )
    except Exception:
        logging.info(f"Weather file not found or unreadable: {weather_file}")
        return pd.DataFrame()

    if weather_data.empty:
        logging.info(f"Weather dataframe empty: {weather_file}")
        return pd.DataFrame()

    if "Time" not in weather_data.columns:
        logging.info(f"Weather file missing Time column: {weather_file}")
        return pd.DataFrame()

    weather_data["Time"] = _to_seconds(weather_data["Time"])
    weather_data = (
        weather_data
        .dropna(subset=["Time"])
        .sort_values("Time")
        .reset_index(drop=True)
    )

    return weather_data


def _prepare_driver_race_data(
    race_year: int,
    race_location: str,
    driver: str,
    bronze_container: str,
    storage_options: dict,
    fs
) -> Optional[pd.DataFrame]:
    telemetry_file, laps_file, _ = _get_bronze_paths(
        bronze_container, race_year, race_location, SESSION_TYPE, driver
    )

    if not _exists(fs, telemetry_file) or not _exists(fs, laps_file):
        logging.info(f"Skipping {driver} in {race_location} - missing telemetry or laps file")
        return None

    try:
        tel_data = pd.read_parquet(
            telemetry_file,
            columns=[c for c in TELEMETRY],
            storage_options=storage_options
        )
        lap_data = pd.read_parquet(
            laps_file,
            columns=[c for c in LAP],
            storage_options=storage_options
        )
    except Exception as e:
        logging.exception(f"Error reading parquet for {driver} in {race_location}: {e}")
        return None

    tel_data = clean_gear_column(tel_data)

    if "session_time" not in tel_data.columns or "LapStartTime" not in lap_data.columns:
        logging.info(f"Skipping {driver} in {race_location} - missing required time columns")
        return None

    tel_data["session_time"] = _to_seconds(tel_data["session_time"])
    lap_data["LapStartTime"] = _to_seconds(lap_data["LapStartTime"])

    tel_data = tel_data.dropna(subset=["session_time"]).sort_values("session_time").reset_index(drop=True)
    lap_data = lap_data.dropna(subset=["LapStartTime"]).sort_values("LapStartTime").reset_index(drop=True)

    if tel_data.empty or lap_data.empty:
        logging.info(f"Skipping {driver} in {race_location} - telemetry or laps empty after cleaning")
        return None

    merged = pd.merge_asof(
        tel_data,
        lap_data,
        left_on="session_time",
        right_on="LapStartTime",
        direction="backward"
    )

    merged = merged.drop(columns=["LapStartTime"], errors="ignore")

    if merged.empty:
        logging.info(f"Skipping {driver} in {race_location} - merged dataframe empty")
        return None

    merged["driver_code"] = driver
    merged["race_year"] = race_year
    merged["race_location"] = race_location
    merged["session_type"] = SESSION_TYPE

    logging.info(f"{driver} merged shape in {race_location}: {merged.shape}")
    return merged


def load_race_data(
    race_year: int,
    race_location: str,
    race_id: int,
    bronze_container: str,
    storage_options: dict,
    fs
) -> pd.DataFrame:
    logging.info("=" * 70)
    logging.info(f"Processing {race_location} - {race_year}")
    logging.info("=" * 70)

    driver_frames = []

    _, _, weather_file = _get_bronze_paths(
        bronze_container, race_year, race_location, SESSION_TYPE, "LEC"
    )
    weather_data = _prepare_weather(weather_file, storage_options)

    if weather_data.empty:
        logging.info(f"Skipping race {race_location} - weather missing or empty")
        return pd.DataFrame()

    for driver in driver_abb.keys():
        driver_df = _prepare_driver_race_data(
            race_year=race_year,
            race_location=race_location,
            driver=driver,
            bronze_container=bronze_container,
            storage_options=storage_options,
            fs=fs
        )
        if driver_df is None or driver_df.empty:
            continue

        driver_df = pd.merge_asof(
            driver_df.sort_values("session_time"),
            weather_data.sort_values("Time"),
            left_on="session_time",
            right_on="Time",
            direction="nearest",
            tolerance=60.0
        ).drop(columns=["Time"], errors="ignore")

        driver_df["race_id"] = race_id
        driver_frames.append(driver_df)

    if not driver_frames:
        logging.info(f"Skipping race {race_location} - no driver data prepared")
        return pd.DataFrame()

    race_df = pd.concat(driver_frames, ignore_index=True, sort=False)
    race_df = add_race_date(race_df)

    min_time = race_df["session_time"].min()
    race_df = race_df[race_df["session_time"] > min_time]

    logging.info(f"Final silver race shape for {race_location}: {race_df.shape}")
    return race_df


# ── ONLY CHANGE FROM ORIGINAL: writes to Azure instead of local disk ───────────
def write_race_output(
    race_df: pd.DataFrame,
    race_year: int,
    race_location: str,
    silver_container: str,
    storage_options: dict
) -> str:
    safe_location = re.sub(r"[^\w]+", "_", race_location).strip("_")
    output_file = _abfs_path(
        silver_container,
        f"{race_year}/{SESSION_TYPE}/{safe_location}.parquet"
    )
    race_df.to_parquet(output_file, index=False, storage_options=storage_options)
    logging.info(f"Saved silver to Azure: {output_file}")
    return output_file


def run_silver_pipeline(
    year: Optional[int] = None,
    session_type: Optional[str] = None,
    race_location: Optional[str] = None,
    storage_account: Optional[str] = None,
    storage_key: Optional[str] = None,
    bronze_container: Optional[str] = None,
    silver_container: Optional[str] = None
):
    global SESSION_TYPE

    # -------- YEARS --------
    if year:
        if isinstance(year, list):
            selected_years = [int(y) for y in year]
        else:
            selected_years = [int(y.strip()) for y in str(year).split(",") if y.strip()]
    else:
        selected_years = YEARS

    # -------- RACES --------
    if race_location:
        if isinstance(race_location, list):
            selected_races = race_location
        else:
            selected_races = [r.strip() for r in str(race_location).split(",") if r.strip()]
    else:
        selected_races = RACE_LOCATIONS

    original_session_type = SESSION_TYPE
    SESSION_TYPE = session_type or SESSION_TYPE

    storage_options = get_storage_options(storage_account, storage_key)
    fs = get_filesystem(storage_account, storage_key)

    output_files = []
    race_id = 0

    try:
        for race_year in selected_years:
            for race in selected_races:
                try:
                    race_df = load_race_data(
                        race_year=race_year,
                        race_location=race,
                        race_id=race_id,
                        bronze_container=bronze_container,
                        storage_options=storage_options,
                        fs=fs
                    )

                    if race_df.empty:
                        continue

                    out_file = write_race_output(
                        race_df=race_df,
                        race_year=race_year,
                        race_location=race,
                        silver_container=silver_container,
                        storage_options=storage_options
                    )

                    output_files.append(out_file)
                    race_id += 1

                    del race_df
                    gc.collect()

                except Exception as e:
                    logging.exception(f"Error processing {race} - {race_year}: {e}")
                    gc.collect()
                    continue

    finally:
        SESSION_TYPE = original_session_type

    if not output_files:
        logging.warning("No silver files created")

    return output_files


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
    run_silver_pipeline()
