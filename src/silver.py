import os
import logging
from pathlib import Path
import pandas as pd


# --------------------------- Configuration ------------------------------
DATA_ROOT = Path(os.getenv("DATA_ROOT", "data"))
BRONZE_PATH = DATA_ROOT / "bronze"
SILVER_PATH = DATA_ROOT / "silver"

YEARS = [2024, 2025]
SESSION_TYPE = os.getenv("SESSION_TYPE", "R")
TARGET_DRIVER = os.getenv("TARGET_DRIVER", "LEC")

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


def ensure_dirs() -> None:
    SILVER_PATH.mkdir(parents=True, exist_ok=True)


def _to_seconds(series: pd.Series) -> pd.Series:
    td = pd.to_timedelta(series, errors="coerce")
    return td.dt.total_seconds()


def _get_bronze_paths(race_year: int, race_location: str, session_type: str, selected_driver: str):
    base = BRONZE_PATH / str(race_year) / race_location / session_type

    telemetry_file = base / selected_driver / f"{selected_driver}_telemetry.parquet"
    overall_file = base / selected_driver / f"{selected_driver}_laps.parquet"
    weather_file = base / "weather.parquet"

    return telemetry_file, overall_file, weather_file


def load_data_files(target_driver: str) -> pd.DataFrame:
    """
    This function will load all the data files and align them
    according to the desired target_driver. Since, this code is
    focused on building ML Models per driver.
    """
    race_id = 0
    prepared_data = pd.DataFrame()

    for race_year in YEARS:
        for race_location in race_locations:
            print(f"\n{'='*60}")
            print(f"Processing {race_location} - {race_year}")
            print(f"{'='*60}")

            # Dictionary to store all driver data and weather data for this race
            driver_data_dict = {}
            target_driver_times = None
            weather_data_dict = None

            for selected_driver in driver_abb.keys():
                try:
                    telemetry_file, overall_file, weather_file = _get_bronze_paths(
                        race_year, race_location, SESSION_TYPE, selected_driver
                    )

                    # Skip if files not found
                    if not telemetry_file.exists() or not overall_file.exists():
                        print(f"Skipping {selected_driver} - files not found")
                        continue

                    print(f"Processing {selected_driver}...")

                    # Read data
                    sector_data = pd.read_parquet(overall_file)
                    tel_data = pd.read_parquet(telemetry_file)

                    # Read weather data only once (not for each driver)
                    if weather_data_dict is None and weather_file.exists():
                        weather_data = pd.read_parquet(weather_file)
                        weather_data["Time"] = _to_seconds(weather_data["Time"])
                        weather_data = weather_data.dropna(subset=["Time"])
                        weather_data_dict = weather_data

                    print("Data Read complete!")

                    # ==================================================
                    # GATHERING, FORMATTING AND COMBINING DATA OF RACERS
                    # ==================================================

                    # Format the data
                    sector_data["LapStartTime"] = _to_seconds(sector_data["LapStartTime"])
                    tel_data["session_time"] = _to_seconds(tel_data["session_time"])

                    print("Time conversion complete!")

                    tel_data = tel_data[
                        ["session_time", "x", "y", "z", "speed", "gear", "rpm", "drs", "brake", "position"]
                    ]
                    sector_data = sector_data[
                        ["LapStartTime", "Driver", "DriverNumber", "LapNumber", "PitInTime", "PitOutTime",
                         "TyreLife", "Compound", "Team", "TrackStatus"]
                    ]

                    tel_data = tel_data.dropna(subset=["session_time"])
                    sector_data = sector_data.dropna(subset=["LapStartTime"])

                    # Merge desired data
                    driver_race_details = pd.merge_asof(
                        left=tel_data.sort_values("session_time"),
                        right=sector_data.sort_values("LapStartTime"),
                        left_on="session_time",
                        right_on="LapStartTime"
                    )

                    # Drop 'LapStartTime' column, since it is now unnecessary
                    driver_race_details = driver_race_details.drop(columns=["LapStartTime"])

                    # Store the session times for the target driver
                    if selected_driver == target_driver:
                        target_driver_times = driver_race_details[["session_time"]].copy()
                        target_driver_times = target_driver_times.reset_index(drop=True)

                    # Store driver data with appropriate prefix
                    prefix = f"Target_{target_driver}_" if selected_driver == target_driver else f"{selected_driver}_"
                    driver_race_details = driver_race_details.add_prefix(prefix)

                    # Add race data columns (race_id, race_year, race_location)
                    if selected_driver == target_driver:
                        driver_race_details["race_id"] = race_id
                        driver_race_details["race_year"] = race_year
                        driver_race_details["race_location"] = race_location

                    # Reset index for proper alignment
                    driver_race_details = driver_race_details.reset_index(drop=True)

                    # Store in dictionary
                    driver_data_dict[selected_driver] = driver_race_details

                    print(f"{selected_driver} data processed")

                except Exception as e:
                    print(f"Error processing {selected_driver}: {str(e)}")
                    continue

            # Skip this race if no data was collected
            if not driver_data_dict:
                print(f"No data collected for {race_location} - {race_year}, skipping...")
                continue

            # Skip if target driver data is missing
            if target_driver not in driver_data_dict:
                print(f"Target driver {target_driver} not found for {race_location} - {race_year}, skipping...")
                continue
            elif weather_data_dict is None or weather_data_dict.empty:
                print(f"No weather data for {race_location} - {race_year}, skipping...")
                continue

            print(f"\nMerging data for {len(driver_data_dict)} drivers...")

            # Build race_data by merging on target driver's session time
            race_data = target_driver_times.copy()

            for driver, driver_df in driver_data_dict.items():
                if driver == target_driver:
                    # Merge target driver data on session time
                    prefix = f"Target_{target_driver}_"
                    target_cols = [col for col in driver_df.columns if col != f"{prefix}session_time"]

                    race_data = pd.merge(
                        race_data,
                        driver_df[target_cols],
                        left_index=True,
                        right_index=True,
                        how="left"
                    )

                    # Merge weather data on session time using merge_asof
                    race_data = race_data.sort_values("session_time")
                    weather_data_sorted = weather_data_dict.sort_values("Time")

                    race_data = pd.merge_asof(
                        race_data,
                        weather_data_sorted,
                        left_on="session_time",
                        right_on="Time",
                        direction="nearest"
                    )

                    # Drop the 'Time' column as it's redundant with 'session_time'
                    race_data = race_data.drop(columns=["Time"])

                else:
                    # Merge other drivers using merge_asof on session time
                    prefix = f"{driver}_"
                    session_col = f"{prefix}session_time"
                    other_cols = [col for col in driver_df.columns if col != session_col]

                    temp_df = driver_df[[session_col] + other_cols].copy()
                    temp_df = temp_df.sort_values(session_col)

                    race_data = pd.merge_asof(
                        race_data.sort_values("session_time"),
                        temp_df,
                        left_on="session_time",
                        right_on=session_col,
                        direction="nearest"
                    )

                    # Drop the extra session time column
                    race_data = race_data.drop(columns=[session_col])

            print(f"Race data merged: {race_data.shape[0]} rows, {race_data.shape[1]} columns")

            # Append to master DataFrame
            prepared_data = pd.concat([prepared_data, race_data], axis=0, ignore_index=True, sort=False)

            print(f"Database updated: Total rows = {prepared_data.shape[0]}")

            # Increment the race_id
            race_id += 1

    return prepared_data


def run_silver_pipeline(target_driver: str = TARGET_DRIVER) -> Path:
    ensure_dirs()
    data = load_data_files(target_driver)

    output_file = SILVER_PATH / f"{target_driver}_silver.parquet"
    data.to_parquet(output_file, index=False)

    print(f"\n{'='*60}")
    print(f"FINAL DATASET: {data.shape[0]} rows × {data.shape[1]} columns")
    print(f"{'='*60}")

    for col in data.columns:
        print(col)

    print(f"\nSaved silver data to: {output_file}")
    return output_file


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run_silver_pipeline()

