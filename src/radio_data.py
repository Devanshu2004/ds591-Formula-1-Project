"""
F1 Team Radio Pipeline — Batch & Live Modes

Two operating modes:

  BATCH MODE (default):
    Fetches historical team radio from OpenF1, transcribes with Whisper locally,
    classifies, and saves to JSON/CSV.

  LIVE MODE (--live):
    Polls OpenF1 API for new radio recordings during a live session,
    transcribes with Azure Speech Services (real-time, low latency),
    classifies, and pushes events to Azure Event Hub.

Architecture (live):
  OpenF1 API (poll) → Azure Speech Services → Classify → Azure Event Hub
                                                        → Stream Analytics → Power BI

Environment variables (loaded from local.settings.json or Azure App Settings):
    F1_STORAGE_CONNECTION_STRING - ADLS Gen2 connection string (batch → bronze)
    ADLS_CONTAINER_NAME          - ADLS container name (default: "bronze")
    EVENT_HUB_CONNECTION_STRING  - Azure Event Hub connection string
    EVENT_HUB_NAME               - Event Hub instance name
    AZURE_SPEECH_KEY             - Azure Speech Services subscription key
    AZURE_SPEECH_REGION          - Azure Speech Services region (e.g. "eastus")

Usage:
    python radio_data.py                                # Batch: all race sessions
    python radio_data.py --session-key 9158             # Batch: specific session
    python radio_data.py --live --session-key 9158      # Live: poll a live session
    python radio_data.py --live --poll-interval 10      # Live: custom poll interval
"""

import argparse
import json
import logging
import os
import re
import tempfile
import time

import pandas as pd
import requests

BASE_URL = "https://api.openf1.org/v1"
REQUEST_TIMEOUT = 60

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

# ==============================
# CONFIGURATION — load local.settings.json, then read env vars
# ==============================
_settings_path = os.path.join(os.path.dirname(__file__), "..", "local.settings.json")
if os.path.exists(_settings_path):
    with open(_settings_path) as _f:
        _settings = json.load(_f)
    for _key, _val in _settings.get("Values", {}).items():
        os.environ.setdefault(_key, str(_val))
    log.info("Loaded settings from local.settings.json")

EVENT_HUB_CONNECTION_STR = os.environ.get("EVENT_HUB_CONNECTION_STRING", "")
EVENT_HUB_NAME = os.environ.get("EVENT_HUB_NAME", "")
AZURE_SPEECH_KEY = os.environ.get("AZURE_SPEECH_KEY", "")
AZURE_SPEECH_REGION = os.environ.get("AZURE_SPEECH_REGION", "")
ADLS_CONNECTION_STR = os.environ.get("F1_STORAGE_CONNECTION_STRING", "")
ADLS_CONTAINER_NAME = os.environ.get("ADLS_CONTAINER_NAME", "bronze")

# ---------------------------------------------------------------------------
# Keyword patterns for classification
# ---------------------------------------------------------------------------
PIT_KW = re.compile(r"\b(box|pit|pitting|pit\s*stop|pit\s*lane|pit\s*wall|pit\s*now|box\s*box)\b", re.I)
TIRE_KW = re.compile(r"\b(tyre|tire|soft|medium|hard|inter|wet|compound|graining|degradation|deg|blistering|wear)\b", re.I)
SAFETY_KW = re.compile(r"\b(yellow|safety\s*car|vsc|virtual|red\s*flag|caution|incident|crash|off|barrier|debris)\b", re.I)
PACE_KW = re.compile(r"\b(push|pushing|attack|lift|coast|lift\s*and\s*coast|delta|target|pace|manage|conserve|save)\b", re.I)
DAMAGE_KW = re.compile(r"\b(damage|wing|floor|puncture|broken|loose|endplate)\b", re.I)
MECH_KW = re.compile(r"\b(engine|power\s*unit|gearbox|brake|battery|ers|mgu|overheating|temperature|cooling|hydraulic|steering)\b", re.I)
WEATHER_KW = re.compile(r"\b(rain|wet|dry|drizzle|shower|spray|standing\s*water)\b", re.I)
OVERTAKE_KW = re.compile(r"\b(overtake|pass|passing|move|sent\s*it|inside|outside|late\s*brake)\b", re.I)
DEFEND_KW = re.compile(r"\b(defend|defending|cover|position|hold|protect)\b", re.I)
DRS_KW = re.compile(r"\b(drs|detection|activation)\b", re.I)
TRAFFIC_KW = re.compile(r"\b(traffic|blue\s*flag|backmarker|lapped)\b", re.I)
GAP_KW = re.compile(r"\b(gap|interval|behind|ahead|margin|undercut|overcut)\b", re.I)
POSITIVE_KW = re.compile(r"\b(well\s*done|great|nice|perfect|brilliant|fantastic|good\s*job|p[1-3]|win|winner|podium|congratulations)\b", re.I)

ALL_PATTERNS = [PIT_KW, TIRE_KW, SAFETY_KW, PACE_KW, DAMAGE_KW, MECH_KW,
                WEATHER_KW, OVERTAKE_KW, DEFEND_KW, DRS_KW, TRAFFIC_KW]


# ---------------------------------------------------------------------------
# Data fetching
# ---------------------------------------------------------------------------
def fetch_sessions():
    log.info("Fetching sessions from OpenF1 API...")
    resp = requests.get(f"{BASE_URL}/sessions", timeout=REQUEST_TIMEOUT)
    resp.raise_for_status()
    df = pd.DataFrame(resp.json())
    log.info("  %d sessions fetched", len(df))
    return df


def fetch_team_radio(session_key=None):
    url = f"{BASE_URL}/team_radio"
    params = {}
    if session_key:
        params["session_key"] = session_key
    log.info("Fetching team radio%s...", f" for session {session_key}" if session_key else "")
    resp = requests.get(url, params=params, timeout=120)
    resp.raise_for_status()
    df = pd.DataFrame(resp.json())
    log.info("  %d recordings fetched", len(df))
    return df


def filter_race_radio(radio_df, sessions_df):
    race_keys = sessions_df[sessions_df["session_type"] == "Race"]["session_key"]
    filtered = radio_df[radio_df["session_key"].isin(race_keys)].reset_index(drop=True)
    log.info("  %d race-only recordings after filtering", len(filtered))
    return filtered


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------
def classify_radio(transcript, record):
    text = transcript.strip()
    text_lower = text.lower()

    hits = {
        "pit": bool(PIT_KW.search(text)),
        "tire": bool(TIRE_KW.search(text)),
        "safety": bool(SAFETY_KW.search(text)),
        "pace": bool(PACE_KW.search(text)),
        "damage": bool(DAMAGE_KW.search(text)),
        "mech": bool(MECH_KW.search(text)),
        "weather": bool(WEATHER_KW.search(text)),
        "overtake": bool(OVERTAKE_KW.search(text)),
        "defend": bool(DEFEND_KW.search(text)),
        "drs": bool(DRS_KW.search(text)),
        "traffic": bool(TRAFFIC_KW.search(text)),
        "gap": bool(GAP_KW.search(text)),
        "positive": bool(POSITIVE_KW.search(text)),
    }

    # Primary event type
    primary = "information_only"
    if hits["pit"] and ("box" in text_lower or "pit" in text_lower):
        primary = "pit_call"
    elif hits["damage"]:
        primary = "damage_issue"
    elif hits["mech"]:
        primary = "mechanical_issue"
    elif hits["safety"]:
        primary = "safety"
    elif hits["weather"]:
        primary = "weather"
    elif hits["tire"] and not hits["pit"]:
        primary = "tire_strategy"
    elif hits["pace"]:
        primary = "pace_management"
    elif hits["overtake"]:
        primary = "overtaking"
    elif hits["defend"]:
        primary = "defending"
    elif hits["traffic"]:
        primary = "traffic"
    elif hits["positive"]:
        primary = "celebration"

    # Secondary labels
    secondary = []
    label_map = [
        ("tire", "tire_strategy"), ("pit", "pit_call"), ("safety", "safety"),
        ("pace", "pace_management"), ("damage", "damage_issue"),
        ("mech", "mechanical_issue"), ("overtake", "overtaking"),
        ("defend", "defending"), ("traffic", "traffic"), ("positive", "celebration"),
    ]
    for key, label in label_map:
        if hits[key] and label != primary and label not in secondary:
            secondary.append(label)

    # Action type
    action_required = True
    if primary == "pit_call" and "box" in text_lower:
        action_type = "pit_now"
    elif primary == "pit_call":
        action_type = "pit_soon"
    elif "stay out" in text_lower:
        action_type = "stay_out"
    elif "push" in text_lower or "attack" in text_lower:
        action_type = "push"
    elif "conserve" in text_lower or "save" in text_lower or "lift and coast" in text_lower:
        action_type = "conserve"
    elif hits["tire"] and ("manage" in text_lower or "look after" in text_lower):
        action_type = "manage_tires"
    elif hits["defend"]:
        action_type = "defend"
    elif hits["overtake"]:
        action_type = "overtake"
    elif hits["damage"] or hits["mech"]:
        action_type = "report_issue"
    elif primary in ("information_only", "celebration", "safety"):
        action_type = "acknowledge_info"
        action_required = False
    else:
        action_type = "unknown"
        action_required = False

    # Urgency
    if primary in ("pit_call", "damage_issue", "safety") or "now" in text_lower:
        urgency = "high"
    elif primary in ("mechanical_issue", "pace_management", "tire_strategy", "defending"):
        urgency = "medium"
    else:
        urgency = "low"

    # Sentiment
    if hits["positive"]:
        sentiment = "positive"
    elif hits["damage"] or hits["mech"]:
        sentiment = "negative"
    elif urgency == "high":
        sentiment = "urgent"
    else:
        sentiment = "neutral"

    # Quality & confidence
    word_count = len(text.split())
    quality = "low" if word_count < 3 else ("medium" if word_count < 8 else "high")
    keyword_hits = sum(1 for v in hits.values() if v)
    if keyword_hits >= 2 and quality == "high":
        confidence = 0.85
    elif keyword_hits >= 1 and quality != "low":
        confidence = 0.7
    elif keyword_hits >= 1:
        confidence = 0.5
    else:
        confidence = 0.35

    # Evidence phrases
    evidence = []
    for pattern in ALL_PATTERNS:
        for match in pattern.finditer(text):
            start = max(0, match.start() - 15)
            end = min(len(text), match.end() + 15)
            snippet = text[start:end].strip()
            if snippet and snippet not in evidence:
                evidence.append(snippet)

    # Car issue signal
    issue_type, severity = "none", "none"
    if hits["damage"]:
        if "wing" in text_lower:
            issue_type = "wing_damage"
        elif "floor" in text_lower:
            issue_type = "floor_damage"
        else:
            issue_type = "unknown"
        severity = "moderate"
    elif hits["mech"]:
        for kw, it in [("engine", "engine"), ("power unit", "engine"), ("brake", "brakes"),
                       ("gearbox", "gearbox"), ("battery", "battery"), ("ers", "battery"),
                       ("overheat", "overheating"), ("temperature", "overheating"),
                       ("steering", "steering")]:
            if kw in text_lower:
                issue_type = it
                break
        else:
            issue_type = "unknown"
        severity = "minor"

    # Summary
    if primary == "pit_call":
        summary = "Pit call communicated."
    elif primary == "celebration":
        summary = "Positive message or celebration."
    elif primary == "information_only" and not secondary:
        summary = "General information exchanged."
    else:
        parts = [primary.replace("_", " ").title()] + [s.replace("_", " ") for s in secondary[:2]]
        summary = f"{'. '.join(parts)} related communication."

    return {
        "date": record.get("date"),
        "driver_number": int(record["driver_number"]) if pd.notna(record.get("driver_number")) else None,
        "meeting_key": int(record["meeting_key"]) if pd.notna(record.get("meeting_key")) else None,
        "session_key": int(record["session_key"]) if pd.notna(record.get("session_key")) else None,
        "recording_url": record.get("recording_url"),
        "transcript_cleaned": text,
        "transcript_quality": quality,
        "summary": summary,
        "primary_event_type": primary,
        "secondary_event_types": secondary,
        "speaker_direction": "unknown",
        "urgency_level": urgency,
        "sentiment_tone": sentiment,
        "action_required": action_required,
        "action_type": action_type,
        "strategy_signal": {
            "pit_related": hits["pit"],
            "tire_related": hits["tire"],
            "fuel_saving": "fuel" in text_lower or "lift and coast" in text_lower,
            "pace_change": hits["pace"],
            "weather_related": hits["weather"],
            "safety_related": hits["safety"],
        },
        "car_issue_signal": {
            "has_issue": hits["damage"] or hits["mech"],
            "issue_type": issue_type,
            "severity": severity,
        },
        "racecraft_signal": {
            "traffic_mentioned": hits["traffic"],
            "overtake_mentioned": hits["overtake"],
            "defend_mentioned": hits["defend"],
            "drs_mentioned": hits["drs"],
            "gap_management_mentioned": hits["gap"],
        },
        "confidence": confidence,
        "evidence_phrases": evidence[:5],
        "notes": "" if confidence >= 0.7 else "Low confidence — transcript may be noisy or too short.",
    }


# ===========================================================================
# BATCH MODE — Whisper (local transcription)
# ===========================================================================
def load_whisper_model(model_size="base"):
    import torch
    import whisper
    device = "cuda" if torch.cuda.is_available() else "cpu"
    log.info("Loading Whisper '%s' model on %s...", model_size, device)
    model = whisper.load_model(model_size, device=device)
    return model


def transcribe_and_classify_batch(radio_df, model):
    events = []
    failed = 0
    total = len(radio_df)

    for idx, (_, row) in enumerate(radio_df.iterrows()):
        url = row["recording_url"]

        try:
            r = requests.get(url, timeout=30)
            r.raise_for_status()
        except Exception as e:
            log.debug("Download failed for %s: %s", url, e)
            failed += 1
            continue

        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
            f.write(r.content)
            tmppath = f.name

        try:
            result = model.transcribe(tmppath)
            transcript = result["text"].strip()
        except Exception as e:
            log.debug("Transcription failed for %s: %s", url, e)
            failed += 1
            os.unlink(tmppath)
            continue

        os.unlink(tmppath)

        if not transcript:
            failed += 1
            continue

        event = classify_radio(transcript, row.to_dict())
        events.append(event)

        if (idx + 1) % 100 == 0:
            log.info("  Processed %d/%d (failed: %d)", idx + 1, total, failed)

    log.info("Completed: %d events from %d recordings (%d failed/skipped)", len(events), total, failed)
    return events


def save_results(events, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    json_path = os.path.join(output_dir, "team_radio_events.json")
    with open(json_path, "w") as f:
        json.dump(events, f, indent=2)
    log.info("Saved %d events to %s", len(events), json_path)

    csv_path = os.path.join(output_dir, "team_radio_events.csv")
    df = pd.json_normalize(events)
    df.to_csv(csv_path, index=False)
    log.info("Saved CSV to %s", csv_path)

    return json_path, csv_path


def upload_to_adls_bronze(events, session_key=None, race_name=None):
    """Upload classified radio events to ADLS Gen2 bronze/radio/ as parquet."""
    if not ADLS_CONNECTION_STR:
        log.warning("F1_STORAGE_CONNECTION_STRING not set — skipping ADLS upload.")
        return None

    from azure.storage.filedatalake import DataLakeServiceClient
    import io

    try:
        service_client = DataLakeServiceClient.from_connection_string(ADLS_CONNECTION_STR)
        fs_client = service_client.get_file_system_client(ADLS_CONTAINER_NAME)

        # Build path: bronze/radio/{race_name}-{session_key}/team_radio_events.parquet
        if race_name and session_key:
            folder_name = f"{race_name}-{session_key}".replace(" ", "_")
        elif session_key:
            folder_name = str(session_key)
        else:
            folder_name = "all_sessions"
        folder = f"radio/{folder_name}"

        # Create directory if it doesn't exist
        dir_client = fs_client.get_directory_client(folder)
        dir_client.create_directory()

        # Convert events to parquet bytes
        df = pd.json_normalize(events)
        buffer = io.BytesIO()
        df.to_parquet(buffer, index=False)
        parquet_bytes = buffer.getvalue()

        # Upload parquet file
        file_client = dir_client.get_file_client("team_radio_events.parquet")
        file_client.upload_data(parquet_bytes, overwrite=True)

        adls_path = f"{ADLS_CONTAINER_NAME}/{folder}/team_radio_events.parquet"
        log.info("Uploaded %d events to ADLS: %s", len(events), adls_path)
        return adls_path

    except Exception as e:
        log.error("ADLS upload failed: %s", e)
        return None


def print_summary(events):
    df = pd.json_normalize(events)
    log.info("--- Summary ---")
    log.info("Event type distribution:\n%s", df["primary_event_type"].value_counts().to_string())
    log.info("Urgency distribution:\n%s", df["urgency_level"].value_counts().to_string())

    strategy_cols = [c for c in df.columns if c.startswith("strategy_signal.")]
    for col in strategy_cols:
        label = col.split(".")[1]
        log.info("  Strategy signal %-20s %5d occurrences", label, df[col].sum())


def run_batch(session_key=None, output_dir=None, whisper_model="base"):
    log.info("=" * 60)
    log.info("F1 Team Radio Pipeline — BATCH MODE (Whisper)")
    log.info("=" * 60)

    if output_dir is None:
        output_dir = os.path.join(os.path.dirname(__file__), "..", "data")

    sessions = fetch_sessions()
    radio = fetch_team_radio(session_key=session_key)

    if session_key is None:
        radio = filter_race_radio(radio, sessions)

    if radio.empty:
        log.warning("No radio recordings found. Exiting.")
        return []

    model = load_whisper_model(whisper_model)
    events = transcribe_and_classify_batch(radio, model)

    if not events:
        log.warning("No events produced. Exiting.")
        return []

    save_results(events, output_dir)

    # Look up race name from sessions for ADLS folder naming
    race_name = None
    if session_key is not None:
        match = sessions[sessions["session_key"] == session_key]
        if not match.empty:
            race_name = match.iloc[0].get("location", None)

    upload_to_adls_bronze(events, session_key=session_key, race_name=race_name)
    print_summary(events)
    return events


# ===========================================================================
# LIVE MODE — Azure Speech Services + Event Hub
# ===========================================================================
def transcribe_with_azure_speech(audio_content):
    """Transcribe audio bytes using Azure Speech Services REST API."""
    if not AZURE_SPEECH_KEY or not AZURE_SPEECH_REGION:
        log.error("AZURE_SPEECH_KEY or AZURE_SPEECH_REGION not set.")
        return None

    endpoint = (
        f"https://{AZURE_SPEECH_REGION}.stt.speech.microsoft.com"
        f"/speech/recognition/conversation/cognitiveservices/v1"
        f"?language=en-US"
    )
    headers = {
        "Ocp-Apim-Subscription-Key": AZURE_SPEECH_KEY,
        "Content-Type": "audio/mpeg",
        "Accept": "application/json",
    }

    try:
        resp = requests.post(endpoint, headers=headers, data=audio_content, timeout=30)
        resp.raise_for_status()
        result = resp.json()

        if result.get("RecognitionStatus") == "Success":
            return result.get("DisplayText", "").strip()
        else:
            log.debug("Azure Speech status: %s", result.get("RecognitionStatus"))
            return None
    except Exception as e:
        log.error("Azure Speech transcription failed: %s", e)
        return None


def push_to_event_hub(data, label="radio"):
    """Push a list of records to Azure Event Hub as JSON EventData messages."""
    if not data:
        return

    if not EVENT_HUB_CONNECTION_STR or not EVENT_HUB_NAME:
        log.warning("Event Hub not configured — dry-run mode.")
        log.info("[DRY RUN] Would push %d %s records.", len(data), label)
        for record in data[:2]:
            log.debug(json.dumps(record, indent=2, default=str))
        return

    from azure.eventhub import EventHubProducerClient, EventData

    try:
        producer = EventHubProducerClient.from_connection_string(
            conn_str=EVENT_HUB_CONNECTION_STR,
            eventhub_name=EVENT_HUB_NAME,
        )
        with producer:
            batch = producer.create_batch()
            for record in data:
                try:
                    batch.add(EventData(json.dumps(record, default=str)))
                except ValueError:
                    producer.send_batch(batch)
                    log.info("Batch full — sent intermediate batch.")
                    batch = producer.create_batch()
                    batch.add(EventData(json.dumps(record, default=str)))
            producer.send_batch(batch)

        log.info("Pushed %d %s records to Azure Event Hub.", len(data), label)

    except Exception as e:
        log.error("Event Hub push failed: %s", e, exc_info=True)


def run_live(session_key, poll_interval=10):
    """
    Live mode: poll OpenF1 for new radio recordings, transcribe with Azure Speech
    Services, classify, and push to Event Hub in near-real-time.
    """
    log.info("=" * 60)
    log.info("F1 Team Radio Pipeline — LIVE MODE (Azure Speech Services)")
    log.info("  Session key : %s", session_key)
    log.info("  Poll interval: %ds", poll_interval)
    log.info("  Event Hub    : %s", "configured" if EVENT_HUB_CONNECTION_STR else "NOT SET (dry-run)")
    log.info("  Speech Svc   : %s", "configured" if AZURE_SPEECH_KEY else "NOT SET")
    log.info("=" * 60)

    if not AZURE_SPEECH_KEY:
        log.error("AZURE_SPEECH_KEY is required for live mode. Exiting.")
        return

    seen_urls = set()
    total_pushed = 0

    log.info("Starting live polling loop. Press Ctrl+C to stop.")

    try:
        while True:
            try:
                radio = fetch_team_radio(session_key=session_key)
            except Exception as e:
                log.error("Failed to fetch radio data: %s", e)
                time.sleep(poll_interval)
                continue

            if radio.empty:
                log.info("No radio data yet. Waiting...")
                time.sleep(poll_interval)
                continue

            # Filter to only new recordings
            new_rows = radio[~radio["recording_url"].isin(seen_urls)]
            if new_rows.empty:
                log.info("No new recordings. (%d total seen)", len(seen_urls))
                time.sleep(poll_interval)
                continue

            log.info("Found %d new recordings to process.", len(new_rows))
            events = []

            for _, row in new_rows.iterrows():
                url = row["recording_url"]
                seen_urls.add(url)

                # Download audio
                try:
                    r = requests.get(url, timeout=30)
                    r.raise_for_status()
                except Exception as e:
                    log.debug("Download failed for %s: %s", url, e)
                    continue

                # Transcribe with Azure Speech Services
                transcript = transcribe_with_azure_speech(r.content)
                if not transcript:
                    continue

                # Classify
                event = classify_radio(transcript, row.to_dict())
                events.append(event)
                log.info("  Driver %s | %s | %s",
                         event.get("driver_number"), event["primary_event_type"],
                         transcript[:60])

            if events:
                push_to_event_hub(events, label="live-radio")
                total_pushed += len(events)
                log.info("Total events pushed this session: %d", total_pushed)

            time.sleep(poll_interval)

    except KeyboardInterrupt:
        log.info("Live polling stopped by user. Total events pushed: %d", total_pushed)


# ===========================================================================
# MAIN
# ===========================================================================
def main():
    parser = argparse.ArgumentParser(description="F1 Team Radio Pipeline")
    parser.add_argument("--live", action="store_true",
                        help="Run in live mode (Azure Speech + Event Hub)")
    parser.add_argument("--session-key", type=int, default=None,
                        help="Process a specific session key (required for live mode)")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Output directory for batch mode (default: data/)")
    parser.add_argument("--whisper-model", type=str, default="base",
                        choices=["tiny", "base", "small", "medium", "large"],
                        help="Whisper model size for batch mode (default: base)")
    parser.add_argument("--poll-interval", type=int, default=10,
                        help="Seconds between polls in live mode (default: 10)")
    args = parser.parse_args()

    if args.live:
        if not args.session_key:
            parser.error("--session-key is required for live mode")
        run_live(session_key=args.session_key, poll_interval=args.poll_interval)
    else:
        run_batch(session_key=args.session_key, output_dir=args.output_dir,
                  whisper_model=args.whisper_model)


if __name__ == "__main__":
    main()
