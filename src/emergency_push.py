"""
Emergency push — uses 2025 Japan Race (confirmed available on FastF1)
and streams it lap-by-lap to Event Hub to simulate a live race feed.
"""
import json, os, time
from datetime import datetime
from azure.eventhub import EventHubProducerClient, EventData
import fastf1
import pandas as pd, numpy as np

_cache_dir = os.path.join(os.path.dirname(__file__), "..", "cache")
os.makedirs(_cache_dir, exist_ok=True)
fastf1.Cache.enable_cache(_cache_dir)

conn = os.environ["EVENT_HUB_CONNECTION_STRING"]
name = os.environ["EVENT_HUB_NAME"]

def safe(v):
    try:
        if pd.isna(v): return None
        if isinstance(v, np.integer): return int(v)
        if isinstance(v, np.floating): return float(v)
        if isinstance(v, pd.Timedelta): return v.total_seconds()
        if isinstance(v, pd.Timestamp): return v.isoformat()
        return v
    except: return str(v)

print("Loading 2025 Japan Race from FastF1...")
session = fastf1.get_session(2025, "Japan", "R")
session.load(laps=True, telemetry=False, weather=False, messages=False)
laps = session.laps
print(f"Loaded {len(laps)} laps for {laps['Driver'].nunique()} drivers.")

max_lap = int(laps["LapNumber"].max())
print(f"Total laps in race: {max_lap}. Streaming to Event Hub now...\n")

producer = EventHubProducerClient.from_connection_string(conn, eventhub_name=name)

for lap_num in range(1, max_lap + 1):
    current_laps = laps[laps["LapNumber"] <= lap_num]
    latest = current_laps.groupby("Driver").tail(1)

    records = []
    for _, row in latest.iterrows():
        records.append({
            "timestamp_utc": datetime.utcnow().isoformat(),
            "grand_prix": "Japan", "session_type": "R", "year": 2025,
            "driver_code":          safe(row.get("Driver")),
            "driver_number":        safe(row.get("DriverNumber")),
            "team":                 safe(row.get("Team")),
            "lap_time_seconds":     safe(row.get("LapTime")),
            "sector1_time_seconds": safe(row.get("Sector1Time")),
            "sector2_time_seconds": safe(row.get("Sector2Time")),
            "sector3_time_seconds": safe(row.get("Sector3Time")),
            "position":             safe(row.get("Position")),
            "lap_number":           safe(row.get("LapNumber")),
            "compound":             safe(row.get("Compound")),
            "tyre_life":            safe(row.get("TyreLife")),
            "speed_fl":             safe(row.get("SpeedFL")),
            "speed_st":             safe(row.get("SpeedST")),
            "is_personal_best":     safe(row.get("IsPersonalBest")),
            "stint":                safe(row.get("Stint")),
        })

    with producer:
        batch = producer.create_batch()
        for r in records:
            batch.add(EventData(json.dumps(r, default=str)))
        producer.send_batch(batch)
    
    # Re-open producer for next iteration
    producer = EventHubProducerClient.from_connection_string(conn, eventhub_name=name)

    print(f"Lap {lap_num:>2}/{max_lap} — pushed {len(records)} driver records to Event Hub")
    time.sleep(3)  # 3 second delay between laps = simulates live race feed

print("\nDone! All laps pushed. Check Power BI dashboard.")
