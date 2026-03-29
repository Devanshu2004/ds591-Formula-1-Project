import os
from pathlib import Path
import pandas as pd


DATA_ROOT = Path(os.getenv("DATA_ROOT", "data"))
SILVER_PATH = DATA_ROOT / "silver"
GOLD_PATH = DATA_ROOT / "gold"
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


def ensure_dirs():
    GOLD_PATH.mkdir(parents=True, exist_ok=True)


def preprocess_df(df, target_driver):
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

    # Make a temp dict to keep a track of everything
    all_row_data = []

    for row in df.itertuples():
        # All data of current row
        track_append_data = {}

        # Get all the attributes for the Target Driver
        target_driver_attrs = {}

        for value in var_attributes:
            # Get name of columns for the driver
            col_name = str(prefix + "_" + target_driver + "_" + value)
            target_driver_attrs[value] = col_name

        target_driver_attrs["TrackStatus"] = str(prefix + "_" + target_driver + "_TrackStatus")

        for value in const_attributes:
            # Get name of columns for weather data
            target_driver_attrs[value] = str(value)

        # Get session time
        track_append_data["session_time"] = row.session_time
        
        # Get Target Driver stats for the row and append these details to track_append_data dict
        track_append_data["target_x"] = getattr(row, target_driver_attrs["x"])
        track_append_data["target_y"] = getattr(row, target_driver_attrs["y"])
        track_append_data["target_z"] = getattr(row, target_driver_attrs["z"])
        track_append_data["target_speed"] = getattr(row, target_driver_attrs["speed"])
        track_append_data["target_gear"] = getattr(row, target_driver_attrs["gear"])
        track_append_data["target_rpm"] = getattr(row, target_driver_attrs["rpm"])
        track_append_data["target_drs"] = getattr(row, target_driver_attrs["drs"])
        track_append_data["target_brake"] = getattr(row, target_driver_attrs["brake"])
        track_append_data["target_pos"] = getattr(row, target_driver_attrs["position"])
        track_append_data["target_driver"] = getattr(row, target_driver_attrs["Driver"])
        track_append_data["target_driver_number"] = getattr(row, target_driver_attrs["DriverNumber"])
        track_append_data["target_lap_number"] = getattr(row, target_driver_attrs["LapNumber"])
        track_append_data["target_tyre_life"] = getattr(row, target_driver_attrs["TyreLife"])
        track_append_data["target_compound"] = getattr(row, target_driver_attrs["Compound"])
        track_append_data["target_team"] = getattr(row, target_driver_attrs["Team"])
        track_append_data["track_status"] = getattr(row, target_driver_attrs["TrackStatus"])

        # Get all the general data
        track_append_data["race_id"] = getattr(row, target_driver_attrs["race_id"])
        track_append_data["race_year"] = getattr(row, target_driver_attrs["race_year"])
        track_append_data["race_location"] = getattr(row, target_driver_attrs["race_location"])
        track_append_data["air_temp"] = getattr(row, target_driver_attrs["AirTemp"])
        track_append_data["humidity"] = getattr(row, target_driver_attrs["Humidity"])
        track_append_data["pressure"] = getattr(row, target_driver_attrs["Pressure"])
        track_append_data["rainfall"] = getattr(row, target_driver_attrs["Rainfall"])
        track_append_data["track_temp"] = getattr(row, target_driver_attrs["TrackTemp"])
        track_append_data["wind_direction"] = getattr(row, target_driver_attrs["WindDirection"])
        track_append_data["wind_speed"] = getattr(row, target_driver_attrs["WindSpeed"])

        target_pos = getattr(row, target_driver_attrs["position"])
        driver_ahead = target_pos - 1
        driver_behind = target_pos + 1

        # Set up flag checks for ahead and behind drivers
        found_driver_ahead = False
        found_driver_behind = False

        # Get stats per driver (inner loop)
        # Look for immediate drivers
        for driver in driver_abb.keys():
            try:
                # Check selected driver's position
                # If he is not eligible then go to next driver
                selected_driver_pos = getattr(row, str(driver + "_position"))

                if selected_driver_pos == driver_ahead:
                    # Update data of the driver ahead for selected row
                    track_append_data["driver_ahead_x"] = getattr(row, str(driver + "_x"))
                    track_append_data["driver_ahead_y"] = getattr(row, str(driver + "_y"))
                    track_append_data["driver_ahead_z"] = getattr(row, str(driver + "_z"))
                    track_append_data["driver_ahead_speed"] = getattr(row, str(driver + "_speed"))
                    track_append_data["driver_ahead_gear"] = getattr(row, str(driver + "_gear"))
                    track_append_data["driver_ahead_rpm"] = getattr(row, str(driver + "_rpm"))
                    track_append_data["driver_ahead_drs"] = getattr(row, str(driver + "_drs"))
                    track_append_data["driver_ahead_brake"] = getattr(row, str(driver + "_brake"))
                    track_append_data["driver_ahead_pos"] = getattr(row, str(driver + "_position"))
                    track_append_data["driver_ahead"] = getattr(row, str(driver + "_Driver"))
                    track_append_data["driver_ahead_number"] = getattr(row, str(driver + "_DriverNumber"))
                    track_append_data["driver_ahead_lap_number"] = getattr(row, str(driver + "_LapNumber"))
                    track_append_data["driver_ahead_tyre_life"] = getattr(row, str(driver + "_TyreLife"))
                    track_append_data["driver_ahead_compound"] = getattr(row, str(driver + "_Compound"))
                    track_append_data["driver_ahead_team"] = getattr(row, str(driver + "_Team"))
                    found_driver_ahead = True

                if selected_driver_pos == driver_behind:
                    track_append_data["driver_behind_x"] = getattr(row, str(driver + "_x"))
                    track_append_data["driver_behind_y"] = getattr(row, str(driver + "_y"))
                    track_append_data["driver_behind_z"] = getattr(row, str(driver + "_z"))
                    track_append_data["driver_behind_speed"] = getattr(row, str(driver + "_speed"))
                    track_append_data["driver_behind_gear"] = getattr(row, str(driver + "_gear"))
                    track_append_data["driver_behind_rpm"] = getattr(row, str(driver + "_rpm"))
                    track_append_data["driver_behind_drs"] = getattr(row, str(driver + "_drs"))
                    track_append_data["driver_behind_brake"] = getattr(row, str(driver + "_brake"))
                    track_append_data["driver_behind_pos"] = getattr(row, str(driver + "_position"))
                    track_append_data["driver_behind"] = getattr(row, str(driver + "_Driver"))
                    track_append_data["driver_behind_number"] = getattr(row, str(driver + "_DriverNumber"))
                    track_append_data["driver_behind_lap_number"] = getattr(row, str(driver + "_LapNumber"))
                    track_append_data["driver_behind_tyre_life"] = getattr(row, str(driver + "_TyreLife"))
                    track_append_data["driver_behind_compound"] = getattr(row, str(driver + "_Compound"))
                    track_append_data["driver_behind_team"] = getattr(row, str(driver + "_Team"))
                    found_driver_behind = True
                else:
                    continue
            except:
                continue

        try:
            # Closest driver variable
            closest_driver_ahead = str()
            closest_driver_behind = str()

            # Check for the next immediate drivers
            if found_driver_ahead == False and target_pos > 2:
                # Find closest driver ahead
                for driver in driver_abb.keys():
                    try:
                        selected_driver_pos = int(getattr(row, str(driver + "_position")))
                        maximum_dist = 21
                        rank_dist = target_pos - selected_driver_pos

                        if 0 < rank_dist < maximum_dist:
                            closest_driver_ahead = str(driver)

                    except:
                        continue

                # Update data of the driver ahead for selected row
                track_append_data["driver_ahead_x"] = getattr(row, str(closest_driver_ahead + "_x"))
                track_append_data["driver_ahead_y"] = getattr(row, str(closest_driver_ahead + "_y"))
                track_append_data["driver_ahead_z"] = getattr(row, str(closest_driver_ahead + "_z"))
                track_append_data["driver_ahead_speed"] = getattr(row, str(closest_driver_ahead + "_speed"))
                track_append_data["driver_ahead_gear"] = getattr(row, str(closest_driver_ahead + "_gear"))
                track_append_data["driver_ahead_rpm"] = getattr(row, str(closest_driver_ahead + "_rpm"))
                track_append_data["driver_ahead_drs"] = getattr(row, str(closest_driver_ahead + "_drs"))
                track_append_data["driver_ahead_brake"] = getattr(row, str(closest_driver_ahead + "_brake"))
                track_append_data["driver_ahead_pos"] = getattr(row, str(closest_driver_ahead + "_position"))
                track_append_data["driver_ahead"] = getattr(row, str(closest_driver_ahead + "_Driver"))
                track_append_data["driver_ahead_number"] = getattr(row, str(closest_driver_ahead + "_DriverNumber"))
                track_append_data["driver_ahead_lap_number"] = getattr(row, str(closest_driver_ahead + "_LapNumber"))
                track_append_data["driver_ahead_tyre_life"] = getattr(row, str(closest_driver_ahead + "_TyreLife"))
                track_append_data["driver_ahead_compound"] = getattr(row, str(closest_driver_ahead + "_Compound"))
                track_append_data["driver_ahead_team"] = getattr(row, str(closest_driver_ahead + "_Team"))
                found_driver_ahead = True

            if found_driver_behind == False:
                # Find closest driver behind
                for driver in driver_abb.keys():
                    try:
                        selected_driver_pos = int(getattr(row, str(driver + "_position")))
                        maximum_dist = 21
                        rank_dist = int(selected_driver_pos) - int(target_pos)

                        if 0 < rank_dist < maximum_dist:
                            closest_driver_ahead = str(driver)

                    except:
                        continue
                # Update data of the driver behind for selected row
                track_append_data["driver_behind_x"] = getattr(row, str(closest_driver_behind + "_x"))
                track_append_data["driver_behind_y"] = getattr(row, str(closest_driver_behind + "_y"))
                track_append_data["driver_behind_z"] = getattr(row, str(closest_driver_behind + "_z"))
                track_append_data["driver_behind_speed"] = getattr(row, str(closest_driver_behind + "_speed"))
                track_append_data["driver_behind_gear"] = getattr(row, str(closest_driver_behind + "_gear"))
                track_append_data["driver_behind_rpm"] = getattr(row, str(closest_driver_behind + "_rpm"))
                track_append_data["driver_behind_drs"] = getattr(row, str(closest_driver_behind + "_drs"))
                track_append_data["driver_behind_brake"] = getattr(row, str(closest_driver_behind + "_brake"))
                track_append_data["driver_behind_pos"] = getattr(row, str(closest_driver_behind + "_position"))
                track_append_data["driver_behind"] = getattr(row, str(closest_driver_behind + "_Driver"))
                track_append_data["driver_behind_number"] = getattr(row, str(closest_driver_behind + "_DriverNumber"))
                track_append_data["driver_behind_lap_number"] = getattr(row, str(closest_driver_behind + "_LapNumber"))
                track_append_data["driver_behind_tyre_life"] = getattr(row, str(closest_driver_behind + "_TyreLife"))
                track_append_data["driver_behind_compound"] = getattr(row, str(closest_driver_behind + "_Compound"))
                track_append_data["driver_behind_team"] = getattr(row, str(closest_driver_behind + "_Team"))
                found_driver_behind = True

        except:
            all_row_data.append(track_append_data)
            continue

        all_row_data.append(track_append_data)

    processed_data = pd.DataFrame(all_row_data)
    return processed_data


def run_gold_pipeline(target_driver=TARGET_DRIVER):
    ensure_dirs()

    silver_file = SILVER_PATH / f"{target_driver}_silver.parquet"
    df = pd.read_parquet(silver_file)

    gold_df = preprocess_df(df, target_driver)

    output_file = GOLD_PATH / f"{target_driver}_gold.parquet"
    gold_df.to_parquet(output_file, index=False)

    print(f"Saved gold data to: {output_file}")
    print(f"Final gold shape: {gold_df.shape}")

    return output_file


if __name__ == "__main__":
    run_gold_pipeline()