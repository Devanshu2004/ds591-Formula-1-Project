# """
# F1 Live Qualifying/Race Data Pipeline
# DS591 - Big Data Workloads, Boston University
# ----------------------------------------------
# Architecture:
#   - Thread 1 (record_live_session): SignalRClient streams raw live timing to a .txt file
#   - Thread 2 (poll_and_push): Periodically loads the growing file with LiveTimingData,
#     processes ALL available data, and sends to Azure Event Hub

# Azure Pipeline:
#   Python (this script) --> Azure Event Hub --> Azure Stream Analytics --> Power BI

# Environment variables are loaded from local.settings.json (Azure Functions local dev)
# or Azure App Service Application Settings in production. Keys used:
#     F1_USERNAME            - your formula1.com email
#     F1_PASSWORD            - your formula1.com password
#     EVENT_HUB_CONNECTION_STRING  - full Event Hub connection string (Endpoint=sb://...)
#     EVENT_HUB_NAME         - the Event Hub instance name (e.g. "f1-telemetry")
#     F1_YEAR                - e.g. "2026"
#     F1_GRAND_PRIX          - e.g. "Japan"
#     F1_SESSION_TYPE        - e.g. "Q" or "R"

# FastF1 IMPORTANT NOTE:
#   SignalRClient saves raw data that CAN'T be read in real-time.
#   We use a polling approach: reload the file every N seconds using LiveTimingData,
#   extract all available data, and push incremental updates to Event Hub.
# """

# import json
# import os
# import time
# import threading
# import logging
# from datetime import datetime
# from pathlib import Path

# import fastf1
# from fastf1.livetiming.client import SignalRClient
# from fastf1.livetiming.data import LiveTimingData

# from azure.eventhub import EventHubProducerClient, EventData

# import ssl
# import certifi
# ssl._create_default_https_context = ssl.create_default_context
# import os
# os.environ['SSL_CERT_FILE'] = certifi.where()

# # ==============================
# # LOGGING
# # ==============================
# logging.basicConfig(
#     level=logging.INFO,
#     format="%(asctime)s [%(levelname)s] %(message)s",
#     handlers=[
#         logging.StreamHandler(),
#         logging.FileHandler("f1_pipeline.log")
#     ]
# )
# log = logging.getLogger("F1Pipeline")

# # ==============================
# # CONFIGURATION — all read from local.settings.json / Azure App Settings
# # Never hardcode these values here
# # ==============================

# # F1 session config — matches local.settings.json keys exactly
# YEAR         = int(os.environ.get("F1_YEAR", "2026"))
# GRAND_PRIX   = os.environ.get("F1_GRAND_PRIX", "Japanese Grand Prix")   # FastF1 GP name
# SESSION_TYPE = os.environ.get("F1_SESSION_TYPE", "R")     # Q / R / FP1 / FP2 / FP3

# LIVE_DATA_FILE = "live_timing_data.txt"
# POLL_INTERVAL  = 10   # seconds between polls (keep 10–15 to avoid file contention)

# # Azure Event Hub — key name matches local.settings.json: EVENT_HUB_CONNECTION_STRING
# EVENT_HUB_CONNECTION_STR = os.environ.get("EVENT_HUB_CONNECTION_STRING", "")
# EVENT_HUB_NAME           = os.environ.get("EVENT_HUB_NAME", "")

# # FastF1 cache directory
# fastf1.Cache.enable_cache("./cache")

# # ==============================
# # HELPER — safe value converter
# # ==============================
# def safe(val):
#     """Convert pandas/numpy types to JSON-serializable Python types."""
#     if val is None:
#         return None
#     try:
#         import pandas as pd
#         import numpy as np
#         if pd.isna(val):
#             return None
#         if isinstance(val, np.integer):
#             return int(val)
#         if isinstance(val, np.floating):
#             return float(val)
#         if isinstance(val, pd.Timedelta):
#             return val.total_seconds()
#         if isinstance(val, pd.Timestamp):
#             return val.isoformat()
#         return val
#     except Exception:
#         return str(val)


# # ==============================
# # DATA EXTRACTION
# # ==============================
# def extract_laps_data(session) -> list[dict]:
#     """Extract ALL available lap-level fields for every driver's latest lap."""
#     try:
#         laps = session.laps
#         if laps is None or laps.empty:
#             log.warning("No lap data available yet.")
#             return []

#         latest_laps = laps.groupby("Driver").tail(1)
#         output = []
#         for _, row in latest_laps.iterrows():
#             record = {
#                 # Metadata
#                 "timestamp_utc": datetime.utcnow().isoformat(),
#                 "year": YEAR,
#                 "grand_prix": GRAND_PRIX,
#                 "session_type": SESSION_TYPE,

#                 # Driver
#                 "driver_code": safe(row.get("Driver")),
#                 "driver_number": safe(row.get("DriverNumber")),
#                 "team": safe(row.get("Team")),

#                 # Lap timing (seconds for Power BI compatibility)
#                 "lap_time_seconds": safe(row.get("LapTime")),
#                 "sector1_time_seconds": safe(row.get("Sector1Time")),
#                 "sector2_time_seconds": safe(row.get("Sector2Time")),
#                 "sector3_time_seconds": safe(row.get("Sector3Time")),

#                 # Session position & lap info
#                 "position": safe(row.get("Position")),
#                 "lap_number": safe(row.get("LapNumber")),
#                 "stint": safe(row.get("Stint")),

#                 # Tyre data
#                 "compound": safe(row.get("Compound")),
#                 "tyre_life": safe(row.get("TyreLife")),
#                 "fresh_tyre": safe(row.get("FreshTyre")),

#                 # Speed traps (km/h)
#                 "speed_i1": safe(row.get("SpeedI1")),
#                 "speed_i2": safe(row.get("SpeedI2")),
#                 "speed_fl": safe(row.get("SpeedFL")),
#                 "speed_st": safe(row.get("SpeedST")),

#                 # Lap validity
#                 "is_personal_best": safe(row.get("IsPersonalBest")),
#                 "deleted": safe(row.get("Deleted")),
#                 "deleted_reason": safe(row.get("DeletedReason")),
#                 "is_accurate": safe(row.get("IsAccurate")),

#                 # Pit stop
#                 "pit_out_time": safe(row.get("PitOutTime")),
#                 "pit_in_time": safe(row.get("PitInTime")),

#                 # Sector session times
#                 "sector1_session_time": safe(row.get("Sector1SessionTime")),
#                 "sector2_session_time": safe(row.get("Sector2SessionTime")),
#                 "sector3_session_time": safe(row.get("Sector3SessionTime")),

#                 # Lap start
#                 "lap_start_time": safe(row.get("LapStartTime")),
#                 "lap_start_date": safe(row.get("LapStartDate")),
#             }
#             output.append(record)

#         return output

#     except Exception as e:
#         log.error(f"Error extracting lap data: {e}", exc_info=True)
#         return []


# def extract_results_data(session) -> list[dict]:
#     """Extract session results / classification if available (end of session)."""
#     try:
#         results = session.results
#         if results is None or results.empty:
#             return []

#         output = []
#         for _, row in results.iterrows():
#             record = {
#                 "timestamp_utc": datetime.utcnow().isoformat(),
#                 "grand_prix": GRAND_PRIX,
#                 "session_type": SESSION_TYPE,
#                 "driver_number": safe(row.get("DriverNumber")),
#                 "driver_code": safe(row.get("Abbreviation")),
#                 "full_name": safe(row.get("FullName")),
#                 "team_name": safe(row.get("TeamName")),
#                 "grid_position": safe(row.get("GridPosition")),
#                 "classified_position": safe(row.get("ClassifiedPosition")),
#                 "position": safe(row.get("Position")),
#                 "q1_time_seconds": safe(row.get("Q1")),
#                 "q2_time_seconds": safe(row.get("Q2")),
#                 "q3_time_seconds": safe(row.get("Q3")),
#                 "time_seconds": safe(row.get("Time")),
#                 "status": safe(row.get("Status")),
#                 "points": safe(row.get("Points")),
#             }
#             output.append(record)
#         return output
#     except Exception as e:
#         log.error(f"Error extracting results: {e}", exc_info=True)
#         return []


# # ==============================
# # AZURE EVENT HUB PUSH
# # ==============================
# def push_to_event_hub(data: list[dict], label: str = "laps"):
#     """
#     Push a list of records to Azure Event Hub.
#     Each record is sent as a separate JSON EventData message.
#     Falls back to dry-run logging if credentials are not configured.
#     """
#     if not data:
#         return

#     # Dry-run mode if env vars not set
#     if not EVENT_HUB_CONNECTION_STR or not EVENT_HUB_NAME:
#         log.warning("EVENT_HUB_CONNECTION_STRING or EVENT_HUB_NAME not set — dry-run mode.")
#         log.info(f"[DRY RUN] Would push {len(data)} {label} records to Event Hub.")
#         log.debug(json.dumps(data[:2], indent=2, default=str))
#         return

#     try:
#         producer = EventHubProducerClient.from_connection_string(
#             conn_str=EVENT_HUB_CONNECTION_STR,
#             eventhub_name=EVENT_HUB_NAME
#         )
#         with producer:
#             # Azure Event Hub batch — automatically respects 1 MB batch limit
#             event_data_batch = producer.create_batch()
#             for record in data:
#                 try:
#                     event_data_batch.add(EventData(json.dumps(record, default=str)))
#                 except ValueError:
#                     # Batch is full — send current batch, start a new one
#                     producer.send_batch(event_data_batch)
#                     log.info(f"Batch full — sent intermediate batch, starting new one.")
#                     event_data_batch = producer.create_batch()
#                     event_data_batch.add(EventData(json.dumps(record, default=str)))

#             # Send the final (or only) batch
#             producer.send_batch(event_data_batch)

#         log.info(f"✅ Pushed {len(data)} {label} records to Azure Event Hub.")

#     except Exception as e:
#         log.error(f"Event Hub push failed: {e}", exc_info=True)


# # ==============================
# # PROCESS 1: LIVE RECORDING
# # ==============================
# def record_live_session():
#     """
#     Runs SignalRClient to record the live timing stream to a file.
#     Blocks until session ends or timeout. Reconnects automatically.

#     Requires env vars: F1_USERNAME, F1_PASSWORD
#     """
#     session_counter = 0
#     while True:
#         session_counter += 1
#         filename = LIVE_DATA_FILE if session_counter == 1 else \
#             f"live_timing_data_part{session_counter}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.txt"

#         log.info(f"Starting SignalRClient -> {filename}")
#         try:
#             client = SignalRClient(
#                 filename=filename,
#                 filemode="w",
#                 timeout=300,   # Stop if no data for 5 minutes (end of session)
#                 debug=False,
#                 no_auth=False  # Uses F1_USERNAME + F1_PASSWORD env vars automatically
#             )
#             client.start()     # BLOCKS until timeout or Ctrl+C
#             log.info("SignalRClient stopped — session likely ended.")
#             break
#         except KeyboardInterrupt:
#             log.info("Recording stopped by user (Ctrl+C).")
#             break
#         except Exception as e:
#             log.error(f"SignalRClient error: {e}. Reconnecting in 10s...", exc_info=True)
#             time.sleep(10)


# # ==============================
# # PROCESS 2: POLL & PUSH
# # ==============================
# def poll_and_push():
#     """
#     Periodically reloads the growing live data file using LiveTimingData,
#     extracts all available data, and pushes incremental updates to Azure Event Hub.
#     """
#     log.info("Polling loop started. Waiting for data file to appear...")

#     last_lap_count = 0
#     last_result_count = 0

#     while True:
#         time.sleep(POLL_INTERVAL)

#         try:
#             # Discover all data files (handles reconnection parts too)
#             data_files = sorted(Path(".").glob("live_timing_data*.txt"))
#             if not data_files:
#                 log.info("No data file found yet. Waiting...")
#                 continue

#             file_paths = [str(f) for f in data_files]
#             log.debug(f"Loading files: {file_paths}")

#             # Load all available live timing data
#             livedata = LiveTimingData(*file_paths)

#             session = fastf1.get_session(YEAR, GRAND_PRIX, SESSION_TYPE)
#             session.load(
#                 livedata=livedata,
#                 laps=True,
#                 telemetry=False,   # Disable telemetry — huge data, slows polling
#                 weather=True,
#                 messages=True
#             )

#             # --- Laps ---
#             laps_data = extract_laps_data(session)
#             if len(laps_data) > last_lap_count:
#                 new_records = laps_data[last_lap_count:]
#                 push_to_event_hub(new_records, label="laps")
#                 last_lap_count = len(laps_data)
#                 log.info(f"Pushed {len(new_records)} new lap records ({len(laps_data)} total).")
#             else:
#                 log.info(f"No new lap data since last poll ({last_lap_count} records total).")

#             # --- Results (available after session ends) ---
#             results_data = extract_results_data(session)
#             if len(results_data) > last_result_count:
#                 new_results = results_data[last_result_count:]
#                 push_to_event_hub(new_results, label="results")
#                 last_result_count = len(results_data)
#                 log.info(f"Pushed {len(new_results)} new result records.")

#         except FileNotFoundError:
#             log.info("Data file not ready yet. Waiting...")
#         except Exception as e:
#             log.error(f"Poll error: {e}", exc_info=True)


# # ==============================
# # MAIN
# # ==============================
# def main():
#     log.info("=" * 60)
#     log.info("F1 Live Timing Pipeline — Azure Event Hub Edition")
#     log.info(f"Session: {YEAR} {GRAND_PRIX} — {SESSION_TYPE}")
#     log.info(f"Event Hub configured: {'YES' if EVENT_HUB_CONNECTION_STR else 'NO (dry-run mode)'}")
#     log.info("=" * 60)

#     # Validate F1 auth
#     f1_user = os.environ.get("F1_USERNAME")
#     f1_pass = os.environ.get("F1_PASSWORD")
#     if not f1_user or not f1_pass:
#         log.warning(
#             "F1_USERNAME and/or F1_PASSWORD not set.\n"
#             "  export F1_USERNAME='your@email.com'\n"
#             "  export F1_PASSWORD='yourpassword'"
#         )
#     else:
#         log.info(f"F1 auth configured for: {f1_user}")

#     # Validate Event Hub config
#     if not EVENT_HUB_CONNECTION_STR or not EVENT_HUB_NAME:
#         log.warning(
#             "EVENT_HUB_CONNECTION_STRING and/or EVENT_HUB_NAME not set.\n"
#             "  Running in DRY-RUN mode — data will NOT be sent to Azure.\n"
#             "  Check local.settings.json has EVENT_HUB_CONNECTION_STRING filled in."
#         )

#     # Start polling thread (background)
#     poll_thread = threading.Thread(target=poll_and_push, daemon=True)
#     poll_thread.start()
#     log.info("Polling thread started.")

#     # Start live recording (blocks main thread — intentional)
#     log.info("Starting live F1 recording. Blocks until session ends.")
#     log.info("TIP: Start this script 2-3 minutes BEFORE the session begins.")
#     record_live_session()

#     # Final poll after recording finishes
#     log.info("Recording done. Running one final poll...")
#     time.sleep(POLL_INTERVAL + 5)
#     log.info("Pipeline complete.")


# if __name__ == "__main__":
#     main()

"""
F1 Live Qualifying/Race Data Pipeline
DS591 - Big Data Workloads, Boston University
----------------------------------------------
Architecture:
  - Thread 1 (record_live_session): SignalRClient streams raw live timing to a .txt file
  - Thread 2 (poll_and_push): A custom JSON parser reads the raw text file byte-by-byte,
    extracts telemetry updates continuously, and sends a full grid snapshot to Azure Event Hub every 2 seconds.
"""

import json
import os
import time
import threading
import logging
from datetime import datetime
from pathlib import Path

from fastf1.livetiming.client import SignalRClient
from azure.eventhub import EventHubProducerClient, EventData

# --- SSL Fix for macOS ---
import ssl
import certifi
ssl._create_default_https_context = ssl.create_default_context
os.environ['SSL_CERT_FILE'] = certifi.where()

# ==============================
# LOGGING
# ==============================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("f1_pipeline.log")
    ]
)
log = logging.getLogger("F1Pipeline")

# ==============================
# CONFIGURATION
# ==============================
YEAR         = int(os.environ.get("F1_YEAR", "2026"))
GRAND_PRIX   = os.environ.get("F1_GRAND_PRIX", "Japanese Grand Prix")   
SESSION_TYPE = os.environ.get("F1_SESSION_TYPE", "R")     

LIVE_DATA_FILE = "live_timing_data.txt"
POLL_INTERVAL  = 2   # Push updates every 2 seconds for a truly live dashboard

EVENT_HUB_CONNECTION_STR = os.environ.get("EVENT_HUB_CONNECTION_STRING", "")
EVENT_HUB_NAME           = os.environ.get("EVENT_HUB_NAME", "")


# ==============================
# DATA PARSING HELPER
# ==============================
def parse_f1_time(time_str):
    """Converts F1 time strings (e.g., '1:32.412') to seconds for Power BI."""
    if not time_str or not isinstance(time_str, str): 
        return None
    try:
        if ':' in time_str:
            m, s = time_str.split(':')
            return float(m) * 60 + float(s)
        return float(time_str)
    except ValueError:
        return None

class RawTelemetryParser:
    """A custom stream processor that extracts lap data directly from the raw SignalR JSON logs."""
    def __init__(self):
        self.file_positions = {}
        self.driver_state = {} # Holds the absolute latest value for every metric

    def find_lines_dict(self, payload):
        """Recursively hunt for the 'Lines' telemetry block, unpacking stringified JSONs."""
        if isinstance(payload, str):
            if '"Lines"' in payload:  # F1 hides the real data inside a string
                try:
                    return self.find_lines_dict(json.loads(payload))
                except json.JSONDecodeError:
                    pass
            return None
        elif isinstance(payload, dict):
            if 'Lines' in payload and isinstance(payload['Lines'], dict):
                return payload['Lines']
            for k, v in payload.items():
                res = self.find_lines_dict(v)
                if res: return res
        elif isinstance(payload, list):
            for item in payload:
                res = self.find_lines_dict(item)
                if res: return res
        return None

    def extract_value(self, data_dict, key):
        """Safely extracts 'Value' from nested F1 telemetry dicts."""
        if key in data_dict:
            val = data_dict[key]
            if isinstance(val, dict):
                return val.get('Value')
            return val
        return None

    def poll(self):
        data_files = sorted(Path(".").glob("live_timing_data*.txt"))
        lines_processed = 0
        
        # 1. Read all new incoming data and update the running state
        for filepath in data_files:
            path_str = str(filepath)
            if path_str not in self.file_positions:
                self.file_positions[path_str] = 0

            with open(path_str, 'r', encoding='utf-8') as f:
                f.seek(self.file_positions[path_str])
                for line in f:
                    idx = line.find('{')
                    if idx == -1: continue

                    try:
                        data = json.loads(line[idx:])
                        lines_dict = self.find_lines_dict(data)
                        if not lines_dict: continue

                        lines_processed += 1

                        # Update the driver's state with the new deltas
                        for driver_num, driver_data in lines_dict.items():
                            if driver_num not in self.driver_state:
                                self.driver_state[driver_num] = {
                                    "driver_number": str(driver_num),
                                    "lap_time_seconds": None,
                                    "sector1_time_seconds": None,
                                    "sector2_time_seconds": None,
                                    "sector3_time_seconds": None,
                                    "position": None,
                                    "gap_to_leader": None
                                }
                            
                            state = self.driver_state[driver_num]
                            
                            # Update fields if they exist in this specific JSON payload
                            lap = self.extract_value(driver_data, 'LastLapTime')
                            if lap: state["lap_time_seconds"] = parse_f1_time(lap)
                            
                            pos = self.extract_value(driver_data, 'Position')
                            if pos: state["position"] = pos
                            
                            gap = self.extract_value(driver_data, 'GapToLeader')
                            if gap: state["gap_to_leader"] = gap

                            # Sectors are inside a list
                            if 'Sectors' in driver_data and isinstance(driver_data['Sectors'], list):
                                sectors = driver_data['Sectors']
                                if len(sectors) > 0:
                                    s1 = self.extract_value(sectors[0], 'Value')
                                    if s1: state["sector1_time_seconds"] = parse_f1_time(s1)
                                if len(sectors) > 1:
                                    s2 = self.extract_value(sectors[1], 'Value')
                                    if s2: state["sector2_time_seconds"] = parse_f1_time(s2)
                                if len(sectors) > 2:
                                    s3 = self.extract_value(sectors[2], 'Value')
                                    if s3: state["sector3_time_seconds"] = parse_f1_time(s3)

                    except json.JSONDecodeError:
                        pass
                
                # Save position so we only read new bytes on the next poll
                self.file_positions[path_str] = f.tell()

        # 2. If we processed new data, yield a snapshot of the ENTIRE grid right now
        new_records = []
        if lines_processed > 0:
            timestamp = datetime.utcnow().isoformat()
            for driver_num, state in self.driver_state.items():
                # Create a copy of the state and attach the metadata
                record = state.copy()
                record["timestamp_utc"] = timestamp
                record["year"] = YEAR
                record["grand_prix"] = GRAND_PRIX
                record["session_type"] = SESSION_TYPE
                new_records.append(record)
                
        return new_records

# ==============================
# AZURE EVENT Hub PUSH
# ==============================
def push_to_event_hub(data: list[dict], label: str = "laps"):
    if not data:
        return

    if not EVENT_HUB_CONNECTION_STR or not EVENT_HUB_NAME:
        log.warning("EVENT_HUB_CONNECTION_STRING or EVENT_HUB_NAME not set — dry-run mode.")
        return

    try:
        producer = EventHubProducerClient.from_connection_string(
            conn_str=EVENT_HUB_CONNECTION_STR,
            eventhub_name=EVENT_HUB_NAME
        )
        with producer:
            event_data_batch = producer.create_batch()
            for record in data:
                try:
                    event_data_batch.add(EventData(json.dumps(record, default=str)))
                except ValueError:
                    producer.send_batch(event_data_batch)
                    event_data_batch = producer.create_batch()
                    event_data_batch.add(EventData(json.dumps(record, default=str)))

            producer.send_batch(event_data_batch)
        log.info(f"✅ Pushed {len(data)} {label} records to Azure Event Hub.")

    except Exception as e:
        log.error(f"Event Hub push failed: {e}", exc_info=True)

# ==============================
# PROCESS 1: LIVE RECORDING
# ==============================
def record_live_session():
    session_counter = 0
    while True:
        session_counter += 1
        filename = LIVE_DATA_FILE if session_counter == 1 else \
            f"live_timing_data_part{session_counter}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.txt"

        log.info(f"Starting SignalRClient -> {filename}")
        try:
            client = SignalRClient(
                filename=filename,
                filemode="w",
                timeout=300,
                debug=False,
                no_auth=False 
            )
            client.start()    
            log.info("SignalRClient stopped — session likely ended.")
            break
        except KeyboardInterrupt:
            log.info("Recording stopped by user (Ctrl+C).")
            break
        except Exception as e:
            log.error(f"SignalRClient error: {e}. Reconnecting in 10s...", exc_info=True)
            time.sleep(10)

# ==============================
# PROCESS 2: POLL & PUSH
# ==============================
def poll_and_push():
    log.info("Raw telemetry polling loop started. Waiting for data...")
    parser = RawTelemetryParser()

    while True:
        time.sleep(POLL_INTERVAL)
        try:
            new_lap_records = parser.poll()
            if new_lap_records:
                log.info(f"Parsed and pushing snapshot of {len(new_lap_records)} drivers directly from stream.")
                push_to_event_hub(new_lap_records, label="live snapshot")
        except Exception as e:
            log.error(f"Poll error: {e}", exc_info=True)

# ==============================
# MAIN
# ==============================
def main():
    log.info("=" * 60)
    log.info("F1 Real-Time Telemetry Stream — Azure Edition")
    log.info(f"Event Hub configured: {'YES' if EVENT_HUB_CONNECTION_STR else 'NO'}")
    log.info("=" * 60)

    poll_thread = threading.Thread(target=poll_and_push, daemon=True)
    poll_thread.start()

    record_live_session()

    log.info("Recording done. Running final poll...")
    time.sleep(POLL_INTERVAL + 2)
    log.info("Pipeline complete.")

if __name__ == "__main__":
    main()