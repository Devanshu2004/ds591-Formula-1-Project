# Live Casting Pipeline

The live casting pipeline streams real-time Formula 1 timing data to Azure Event Hub during an active race or qualifying session. It is built around three concurrent threads and is designed to feed a live Power BI dashboard with optional model predictions.

## Files

- `src/live_casting.py` - connects to the F1 live timing feed, polls for grid snapshots, and pushes to Azure Event Hub.
- `src/live_inference.py` - runs model inference during the race and exposes predictions to the polling thread.

## Architecture: Three-Thread Design

### Thread 1 — Live Recording (main thread)

`record_live_session()` connects to the official F1 live timing feed via FastF1's `SignalRClient` using `F1_USERNAME` and `F1_PASSWORD`. It writes the raw timing stream to `live_timing_data.txt`, blocking until the session ends (timeout after 5 minutes of no data) or a reconnection limit is hit. On disconnect it automatically retries.

If the session produces multiple part files (e.g. due to reconnection), they are named `live_timing_data_part{N}_{timestamp}.txt` and all are read together by the other threads.

Note: `SignalRClient` output cannot be parsed in real-time. Threads 2 and 3 reload the full file each cycle using `LiveTimingData`.

### Thread 2 — Poll and Push (background)

`poll_and_push()` fires every 5 seconds. Each cycle:

1. Discovers all `live_timing_data*.txt` files on disk.
2. Reloads them all via `LiveTimingData` and calls `fastf1.get_session(...).load(laps=True)`.
3. Extracts the latest lap record for every driver — one record per driver, 20 records total.
4. Attaches model predictions from Thread 3 if available.
5. Pushes all records as a batch to Azure Event Hub.

Each record contains:

- Session metadata: timestamp, year, Grand Prix, session type.
- Driver identity: driver code, driver number, team.
- Lap timing: lap time, sector 1/2/3 times (seconds).
- Race state: position, lap number, stint.
- Tyre data: compound, tyre life, fresh tyre flag.
- Speed traps: speed at intermediate 1/2, finish line, speed trap (km/h).
- Lap validity: personal best flag, deleted flag, deleted reason, accuracy flag.
- Pit stop times: pit in time, pit out time.
- Predictions: `predicted_position`, `prediction_confidence` (null if inference unavailable).

Every cycle pushes unconditionally — no deduplication — because sector times, positions, and speed traps change continuously within a lap.

### Thread 3 — Inference Loop (background, optional)

`inference_loop()` fires every 90 seconds (approximately one lap). Each cycle:

1. Loads the full timing data with telemetry and weather enabled.
2. Calls `run_inference_cycle(session)` from `live_inference.py`, which builds feature matrices for all drivers and runs the Bi-LSTM + Random Forest model.
3. Writes predictions to an in-memory store read by Thread 2 via `get_predictions()`.

This thread only starts if `torch` is installed. If unavailable, Thread 2 sets `predicted_position` and `prediction_confidence` to null.

**Note:** Inference is not fully wired in yet. A feature column mismatch exists between the gold training schema (e.g. `target_driver_number`, OHE `compound_SOFT`) and the live snapshot schema (e.g. `compound`, `team`). Verify `ckpt["feature_cols"]` from a saved `.pt` checkpoint before enabling.

## Azure Event Hub Push

`push_to_event_hub()` batches records into Azure Event Hub's 1 MB batch limit, automatically splitting into multiple batches when needed.

If `EVENT_HUB_CONNECTION_STRING` or `EVENT_HUB_NAME` are not set, the function runs in **dry-run mode** — it logs what it would push without sending anything to Azure.

## Downstream Flow

```
live_casting.py → Azure Event Hub → Azure Stream Analytics → Power BI
```

## Run Commands

Install dependencies:

```bash
pip install -r requirements.txt
```

Set required environment variables in `local.settings.json`:

```
F1_USERNAME, F1_PASSWORD
EVENT_HUB_CONNECTION_STRING, EVENT_HUB_NAME
F1_YEAR, F1_GRAND_PRIX, F1_SESSION_TYPE
```

Run locally (start 2–3 minutes before the session begins):

```bash
python src/live_casting.py
```

Optional Azure Function endpoint (blocks for the full session duration — set `functionTimeout` to `02:30:00` in `host.json`):

```bash
POST /api/run_live
```
