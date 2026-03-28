# This file fetches data from FastF1 and OpenF1 API, formulates the data, converts it into appropriate form and returns it.

import os
import logging
import pandas as pd
import numpy as np
import fastf1
from pathlib import Path

# ------------------ Configuration ----------------------
DATA_ROOT = os.getenv("DATA_ROOT", "data")

BRONZE_PATH = Path(DATA_ROOT) / "bronze"
SILVER_PATH = Path(DATA_ROOT) / "silver"
GOLD_PATH = Path(DATA_ROOT) / "gold"
CACHE_PATH = Path(DATA_ROOT) / "f1_cache"

YEAR = int(os.getenv("YEAR", "2024"))
LOCATION = os.getenv("LOCATION", "Monaco")
SESSION_TYPE = os.getenv("SESSION_TYPE", "R")

# Enable cache 
def setup_cache():
    os.makedirs(CACHE_PATH, exist_ok=True)
    fastf1.Cache.enable_cache(CACHE_PATH)

def ensure_dirs():
    os.makedirs(BRONZE_PATH, exist_ok=True)
    os.makedirs(SILVER_PATH, exist_ok=True)
    os.makedirs(GOLD_PATH, exist_ok=True)
    os.makedirs(CACHE_PATH, exist_ok=True)

#---------------------- Data extraction- bronze layer ---------------------
def run_bronze_pipeline(year: int, location: str, session_type: str):
    logging.info(f"Starting Bronze pipeline: {year} {location} {session_type}")

    # Create directory structure
    base_path = BRONZE_PATH / str(year) / str(location) / str(session_type)
    base_path.mkdir(parents=True, exist_ok=True)

    try:
        # Load session
        logging.info("Loading session")
        session = fastf1.get_session(year, location, session_type)
        session.load()
        logging.info("Session loaded")

        list_of_drivers = session.drivers
        logging.info(f"Drivers: {list_of_drivers}")

        # Step-1
        #---------------------------- Extracting all drivers telemetry --------------------------
        logging.info("Collecting all driver telemetry...")
        all_telemetry_list = []

        for drv in list_of_drivers:
            try:
                drv_laps = session.laps.pick_drivers(drv)
                drv_tel = drv_laps.get_telemetry()

                drv_x_diff = drv_tel['X'].diff().fillna(0)
                drv_y_diff = drv_tel['Y'].diff().fillna(0)
                drv_z_diff = drv_tel['Z'].diff().fillna(0)

                drv_dist_inc = np.sqrt(drv_x_diff**2 + drv_y_diff**2 + drv_z_diff**2)

                drv_tel_subset = pd.DataFrame({
                    'SessionTime': drv_tel['SessionTime'],
                    'DriverNumber': drv,
                    'RelativeDistance': drv_dist_inc.cumsum()
                })

                all_telemetry_list.append(drv_tel_subset)

            except Exception as e:
                logging.warning(f"Telemetry failed for driver {drv}: {e}")

        if all_telemetry_list:
            all_drivers_telemetry = pd.concat(all_telemetry_list, ignore_index=True)
        else:
            all_drivers_telemetry = None

        # Step-2
        #---------------------------- Per driver data --------------------------
        for driver in list_of_drivers:
            logging.info(f"Processing driver {driver}")

            laps_data = session.laps
            lap_table = pd.DataFrame(laps_data)

            driver_lap_table = lap_table.loc[lap_table['DriverNumber'] == driver]
            driver_abb = driver_lap_table['Driver'].iloc[0]

            driver_path = base_path / driver_abb
            driver_path.mkdir(parents=True, exist_ok=True)

            # Save lap data as PARQUET
            lap_file = driver_path / f"{driver_abb}_laps.parquet"
            driver_lap_table.to_parquet(lap_file, index=False)

            # Telemetry 
            lap = session.laps.pick_drivers(driver_abb)
            telemetry = lap.get_telemetry()

            x = telemetry['X']
            y = telemetry['Y']
            z = telemetry['Z']

            speed = telemetry['Speed']
            gear = telemetry['nGear']
            session_time = telemetry['SessionTime']
            rpm = telemetry['RPM']
            drs = telemetry['DRS']
            brake = telemetry['Brake']

            # Calculate relative distance along the track
            x_diff = x.diff().fillna(0)
            y_diff = y.diff().fillna(0)
            z_diff = z.diff().fillna(0)

            # Calculate distance increments
            distance_increments = np.sqrt(x_diff**2 + y_diff**2 + z_diff**2)
            
            # Calculate cumulative relative distance
            relative_distance = distance_increments.cumsum()

            # Derive position based on relative distance at each session time
            if all_drivers_telemetry is not None:
                positions = []

                # Process in batches to improve performance
                for idx in range(len(telemetry)):
                    curr_time = session_time.iloc[idx]
                    curr_dist = relative_distance.iloc[idx]

                    # Get all drivers' relative distances at approximately the same time
                    time_window = pd.Timedelta(seconds=1)

                    nearby_data = all_drivers_telemetry[
                        (all_drivers_telemetry['SessionTime'] >= curr_time - time_window) &
                        (all_drivers_telemetry['SessionTime'] <= curr_time + time_window)
                    ]

                    if len(nearby_data) > 0:
                        # Get max distance for each driver in this time window
                        driver_max_dist = nearby_data.groupby('DriverNumber')['RelativeDistance'].max()
                        # Rank by distance (higher distance = better position/lower number)
                        position = (driver_max_dist > curr_dist).sum() + 1
                        positions.append(position)
                    else:
                        positions.append(np.nan)

                    # Progress indicator
                    if idx % 1000 == 0:
                        logging.info(f"{driver_abb}: {idx}/{len(telemetry)}")

            else:
                positions = [np.nan] * len(telemetry)

            # Create telemetry dataframe
            driver_lap_telemetry = pd.DataFrame({
                'x': x.values,
                'y': y.values,
                'z': z.values,
                'speed': speed.values,
                'gear': gear.values,
                'relative_distance': relative_distance.values,
                'position': positions,
                'session_time': session_time.values,
                'rpm': rpm.values,
                'drs': drs.values,
                'brake': brake.values
            })

            # Save telemetry as parquet
            telemetry_file = driver_path / f"{driver_abb}_telemetry.parquet"
            driver_lap_telemetry.to_parquet(telemetry_file, index=False)    
            logging.info(f"Saved telemetry for {driver_abb}")

        # Step-3
        #---------------------------- Weather data --------------------------
        weather_data = session.weather_data
        weather_file = base_path / "weather.parquet"
        weather_data.to_parquet(weather_file, index=False)

        logging.info("Bronze pipeline completed successfully")

    except Exception as e:
        logging.error(f"Pipeline failed: {e}")
        raise

def run_bronze():
    ensure_dirs()
    setup_cache()
    return run_bronze_pipeline(YEAR, LOCATION, SESSION_TYPE)

# --------------------------------------- Function call to fetch data -----------------------------------------------
def run_bronze_all():
    # Fetch years, session types, and locations from environment variables
    years = [int(y) for y in os.getenv("YEARS", "2025").split(",")]
    session_types = os.getenv("SESSION_TYPES", "R").split(",")
    race_locations = os.getenv("LOCATIONS", "").split(",")

    # If empty or not provided, use default list of race locations
    if race_locations == [""] or not race_locations:
        race_locations = [
            "Italian Grand Prix", "Azerbaijan Grand Prix", "Singapore Grand Prix",
            "Mexico City Grand Prix", "Brazilian Grand Prix", "Las Vegas Grand Prix",
            "Qatar Grand Prix", "Abu Dhabi Grand Prix", "United States Grand Prix",
            "Australian Grand Prix", "Austrian Grand Prix", "Bahrain Grand Prix",
            "Belgian Grand Prix", "British Grand Prix", "Canadian Grand Prix",
            "Chinese Grand Prix", "Dutch Grand Prix", "Emilia Romagna Grand Prix",
            "Hungarian Grand Prix", "Japanese Grand Prix", "Miami Grand Prix",
            "Monaco Grand Prix", "Saudi Arabian Grand Prix", "Spanish Grand Prix"
        ]

    ensure_dirs()
    setup_cache()

    for year in years:
        for session_type in session_types:
            for location in race_locations:
                try:
                    logging.info(f"Running for: {year} | {location} | {session_type}")
                    run_bronze_pipeline(year, location, session_type)
                    logging.info(f"Completed for: {year} | {location}")

                except Exception as e:
                    logging.error(f"Failed for {year} | {location}: {e}")
                    continue
