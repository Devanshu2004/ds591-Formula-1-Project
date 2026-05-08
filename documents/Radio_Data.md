# Radio Data Pipeline

The radio data pipeline processes Formula 1 team radio communications for downstream modeling. It is built around three stages: Bronze ingestion, Silver classification, and Live streaming.

## Files

- `src/radio_data.py` - fetches, transcribes, classifies, and feature-engineers team radio data.

## Bronze: Raw Radio Recordings

`run_radio_bronze()` fetches raw team radio recordings from the OpenF1 API and stores them to ADLS.

Data collected:

- Recording URL, driver number, session key, and meeting key for each transmission.
- Filtered to race sessions only when no specific session key is provided.
- Stored as `bronze/radio_bronze.json`.

Transcription of the raw MP3 audio files is performed separately using OpenAI Whisper (run offline on SCC/Colab) and stored as `bronze/radio_transcripts.json`.

## Silver: Classified Radio Events

`run_radio_silver()` reads the pre-transcribed JSON from Bronze, classifies each transmission, and writes a flat Parquet file to ADLS.

Classification assigns each transmission:

- **Primary event type**: one of 12 categories — pit call, tire strategy, pace management, safety, weather, damage issue, mechanical issue, overtaking, defending, traffic, celebration, or information only.
- **Secondary event types**: additional matching categories beyond the primary.
- **Action type**: the specific instruction implied — pit now, pit soon, stay out, push, conserve, manage tires, defend, overtake, report issue, or acknowledge info.
- **Urgency**: high, medium, or low based on event type and keyword signals.
- **Sentiment**: positive, negative, urgent, or neutral.
- **Confidence score**: 0.35–0.85 based on transcript quality and keyword density.

Additional fields derived per record:

- `grand_prix_name` extracted from the recording URL.
- `radio_session_time`: seconds elapsed since race start, computed from OpenF1 session start times.
- `transcript_quality`: low, medium, or high based on word count.

Silver output is stored as `silver/radio.parquet`, one row per transmission, partitioned by session.

## Feature Engineering

`engineer_features()` aggregates classified radio records into per-driver, per-session model-ready features.

Features produced:

- Transmission counts: total transmissions, action-required count, action-required ratio.
- Issue signals: issue count and mean issue severity.
- Strategy signals: pit-related, tire-related, fuel-saving, pace-change, weather-related, and safety-related transmission counts.
- Racecraft signals: traffic, overtake, defend, DRS, and gap management mention counts.
- Event type counts: one count column per primary event type.
- Action type counts: one count column per action type.
- Secondary event type counts: one count column per secondary event type.

All columns are schema-fixed so that every session partition shares the same structure regardless of which event types appear.

## Live Streaming

`run_radio_live()` is called by an Azure Timer Trigger every 15 seconds during a race.

Each firing:

1. Fetches a valid OpenF1 OAuth2 token, refreshing automatically when near expiry.
2. Opens an MQTT connection to `mqtt.openf1.org:8883` and subscribes to `v1/team_radio`.
3. Listens for 13 seconds, collecting incoming radio payloads.
4. Disconnects, then for each new recording: downloads the MP3, converts to WAV via ffmpeg, transcribes using Azure Speech Services, and classifies.
5. Sends classified events to Azure Event Hub for real-time Power BI streaming.
6. Appends new rows to `silver/radio_live.parquet` on ADLS.

The live output schema matches `silver/radio.parquet` so live and historical data can be merged after the race. Previously processed recording URLs are tracked to avoid duplicate processing across firings.

## Output

Silver radio data is saved as a flat Parquet file:

```text
silver/radio.parquet
```

Live radio data is appended incrementally during a race:

```text
silver/radio_live.parquet
```

Both files are merged into the Gold layer via driver abbreviation, race year, and Grand Prix name.

## Run Commands

Install dependencies:

```bash
pip install -r requirements.txt
```

Run Bronze ingestion:

```bash
python src/radio_data.py --stage bronze
```

Run Silver processing:

```bash
python src/radio_data.py --stage silver
```

Optional Azure Function endpoints:

```bash
POST /api/run_radio_bronze
POST /api/run_radio_silver
```

The live timer trigger fires automatically every 15 seconds when deployed and enabled in `function_app.py`.
