"""
F1 Team Radio Pipeline — Medallion Architecture (Bronze → Silver)

BRONZE: Fetch raw team radio recordings from OpenF1 API → ADLS bronze/radio_bronze.json
SILVER: Transcribe (Whisper) + Classify → ADLS silver/radio_silver.parquet
        Partitioned by year / month / driver_abb
LIVE:   Poll OpenF1 → Azure Speech → Event Hub (commented out)

Environment variables:
    STORAGE_ACCOUNT_NAME         - Azure Storage account name
    STORAGE_ACCOUNT_KEY          - Azure Storage account key
    BRONZE_CONTAINER             - Bronze container (default: "bronze")
    SILVER_CONTAINER             - Silver container (default: "silver")
    EVENT_HUB_CONNECTION_STRING  - Azure Event Hub connection string  (live only)
    EVENT_HUB_NAME               - Event Hub instance name            (live only)
    AZURE_SPEECH_KEY             - Azure Speech Services key          (live only)
    AZURE_SPEECH_REGION          - Azure Speech Services region       (live only)
"""

import json
import logging
import os
import re
import tempfile

import fsspec
import pandas as pd
import requests

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

BASE_URL = "https://api.openf1.org/v1"
REQUEST_TIMEOUT = 60


# --- 1. Configuration ---
def get_storage_options():
    return {
        "account_name": os.getenv("STORAGE_ACCOUNT_NAME"),
        "account_key": os.getenv("STORAGE_ACCOUNT_KEY"),
    }


def _abfs_path(container, filename):
    return f"abfs://{container}@{os.getenv('STORAGE_ACCOUNT_NAME')}.dfs.core.windows.net/{filename}"


# --- 2. Driver Number → Abbreviation ---
DRIVER_NUMBER_TO_ABB = {
    1: "VER",  4: "NOR",  5: "VET",  6: "HAD",  7: "DOO",
    10: "GAS", 11: "PER", 12: "ANT", 14: "ALO", 16: "LEC",
    18: "STR", 20: "MAG", 22: "TSU", 23: "ALB", 24: "ZHO",
    27: "HUL", 30: "LAW", 31: "OCO", 43: "COL", 44: "HAM",
    50: "BOR", 55: "SAI", 63: "RUS", 77: "BOT", 81: "PIA",
    87: "BEA",
}


# --- 3. Keyword Patterns ---
PIT_KW      = re.compile(r"\b(box|pit|pitting|pit\s*stop|pit\s*lane|pit\s*wall|pit\s*now|box\s*box)\b", re.I)
TIRE_KW     = re.compile(r"\b(tyre|tire|soft|medium|hard|inter|wet|compound|graining|degradation|deg|blistering|wear)\b", re.I)
SAFETY_KW   = re.compile(r"\b(yellow|safety\s*car|vsc|virtual|red\s*flag|caution|incident|crash|off|barrier|debris)\b", re.I)
PACE_KW     = re.compile(r"\b(push|pushing|attack|lift|coast|lift\s*and\s*coast|delta|target|pace|manage|conserve|save)\b", re.I)
DAMAGE_KW   = re.compile(r"\b(damage|wing|floor|puncture|broken|loose|endplate)\b", re.I)
MECH_KW     = re.compile(r"\b(engine|power\s*unit|gearbox|brake|battery|ers|mgu|overheating|temperature|cooling|hydraulic|steering)\b", re.I)
WEATHER_KW  = re.compile(r"\b(rain|wet|dry|drizzle|shower|spray|standing\s*water)\b", re.I)
OVERTAKE_KW = re.compile(r"\b(overtake|pass|passing|move|sent\s*it|inside|outside|late\s*brake)\b", re.I)
DEFEND_KW   = re.compile(r"\b(defend|defending|cover|position|hold|protect)\b", re.I)
DRS_KW      = re.compile(r"\b(drs|detection|activation)\b", re.I)
TRAFFIC_KW  = re.compile(r"\b(traffic|blue\s*flag|backmarker|lapped)\b", re.I)
GAP_KW      = re.compile(r"\b(gap|interval|behind|ahead|margin|undercut|overcut)\b", re.I)
POSITIVE_KW = re.compile(r"\b(well\s*done|great|nice|perfect|brilliant|fantastic|good\s*job|p[1-3]|win|winner|podium|congratulations)\b", re.I)

ALL_PATTERNS = [PIT_KW, TIRE_KW, SAFETY_KW, PACE_KW, DAMAGE_KW, MECH_KW,
                WEATHER_KW, OVERTAKE_KW, DEFEND_KW, DRS_KW, TRAFFIC_KW]


# --- 4. OpenF1 Fetching ---
def fetch_sessions():
    log.info("Fetching sessions from OpenF1...")
    resp = requests.get(f"{BASE_URL}/sessions", timeout=REQUEST_TIMEOUT)
    resp.raise_for_status()
    df = pd.DataFrame(resp.json())
    log.info("  %d sessions fetched", len(df))
    return df


def fetch_team_radio(session_key=None):
    params = {"session_key": session_key} if session_key else {}
    log.info("Fetching team radio%s...", f" for session {session_key}" if session_key else "")
    resp = requests.get(f"{BASE_URL}/team_radio", params=params, timeout=120)
    resp.raise_for_status()
    df = pd.DataFrame(resp.json())
    log.info("  %d recordings fetched", len(df))
    return df


# --- 5. Bronze Layer ---
def run_radio_bronze(session_key=None):
    """
    Fetch team radio from OpenF1 and store raw JSON to ADLS bronze/radio_bronze.json.
    Returns the ADLS path written to.
    """
    storage_options = get_storage_options()
    bronze_container = os.getenv("BRONZE_CONTAINER", "bronze")

    sessions = fetch_sessions()
    radio = fetch_team_radio(session_key=session_key)

    if session_key is None:
        race_keys = sessions[sessions["session_type"] == "Race"]["session_key"]
        radio = radio[radio["session_key"].isin(race_keys)].reset_index(drop=True)
        log.info("  %d race-only recordings after filtering", len(radio))

    if radio.empty:
        log.warning("No radio recordings found.")
        return None

    output_path = _abfs_path(bronze_container, "radio_bronze.json")
    records = radio.to_dict(orient="records")

    with fsspec.open(output_path, "w", **storage_options) as f:
        json.dump(records, f)

    log.info("Stored %d raw recordings to %s", len(records), output_path)
    return output_path


# --- 6. Classification ---
def classify_radio(transcript, record):
    text = transcript.strip()
    text_lower = text.lower()

    hits = {
        "pit":      bool(PIT_KW.search(text)),
        "tire":     bool(TIRE_KW.search(text)),
        "safety":   bool(SAFETY_KW.search(text)),
        "pace":     bool(PACE_KW.search(text)),
        "damage":   bool(DAMAGE_KW.search(text)),
        "mech":     bool(MECH_KW.search(text)),
        "weather":  bool(WEATHER_KW.search(text)),
        "overtake": bool(OVERTAKE_KW.search(text)),
        "defend":   bool(DEFEND_KW.search(text)),
        "drs":      bool(DRS_KW.search(text)),
        "traffic":  bool(TRAFFIC_KW.search(text)),
        "gap":      bool(GAP_KW.search(text)),
        "positive": bool(POSITIVE_KW.search(text)),
    }

    primary = "information_only"
    if hits["pit"] and ("box" in text_lower or "pit" in text_lower): primary = "pit_call"
    elif hits["damage"]:                    primary = "damage_issue"
    elif hits["mech"]:                      primary = "mechanical_issue"
    elif hits["safety"]:                    primary = "safety"
    elif hits["weather"]:                   primary = "weather"
    elif hits["tire"] and not hits["pit"]:  primary = "tire_strategy"
    elif hits["pace"]:                      primary = "pace_management"
    elif hits["overtake"]:                  primary = "overtaking"
    elif hits["defend"]:                    primary = "defending"
    elif hits["traffic"]:                   primary = "traffic"
    elif hits["positive"]:                  primary = "celebration"

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

    if primary in ("pit_call", "damage_issue", "safety") or "now" in text_lower:
        urgency = "high"
    elif primary in ("mechanical_issue", "pace_management", "tire_strategy", "defending"):
        urgency = "medium"
    else:
        urgency = "low"

    if hits["positive"]:                 sentiment = "positive"
    elif hits["damage"] or hits["mech"]: sentiment = "negative"
    elif urgency == "high":              sentiment = "urgent"
    else:                                sentiment = "neutral"

    word_count = len(text.split())
    quality = "low" if word_count < 3 else "medium" if word_count < 8 else "high"
    keyword_hits = sum(1 for v in hits.values() if v)
    if keyword_hits >= 2 and quality == "high":   confidence = 0.85
    elif keyword_hits >= 1 and quality != "low":  confidence = 0.7
    elif keyword_hits >= 1:                       confidence = 0.5
    else:                                         confidence = 0.35

    evidence = []
    for pattern in ALL_PATTERNS:
        for match in pattern.finditer(text):
            start = max(0, match.start() - 15)
            end = min(len(text), match.end() + 15)
            snippet = text[start:end].strip()
            if snippet and snippet not in evidence:
                evidence.append(snippet)

    if hits["damage"]:
        issue_type = "wing_damage" if "wing" in text_lower else "floor_damage" if "floor" in text_lower else "unknown"
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
    else:
        issue_type = "none"
        severity = "none"

    if primary == "pit_call":
        summary = "Pit call communicated."
    elif primary == "celebration":
        summary = "Positive message or celebration."
    elif primary == "information_only" and not secondary:
        summary = "General information exchanged."
    else:
        parts = [primary.replace("_", " ").title()] + [s.replace("_", " ") for s in secondary[:2]]
        summary = f"{'. '.join(parts)} related communication."

    driver_num = record.get("driver_number")
    driver_num_int = int(driver_num) if driver_num is not None and pd.notna(driver_num) else None
    driver_abb = DRIVER_NUMBER_TO_ABB.get(driver_num_int, "UNK")

    return {
        "date": record.get("date"),
        "driver_number": driver_num_int,
        "driver_abb": driver_abb,
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
        "notes": "" if confidence >= 0.7 else "Low confidence — transcript may be noisy or too short for reliable classification.",
    }


# --- 7. Silver Layer ---
def run_radio_silver(session_key=None, whisper_model_size="base"):
    """
    Read radio_bronze.json from ADLS, transcribe with Whisper, classify,
    and write partitioned parquet to ADLS silver/radio_silver.parquet
    (partitioned by year / month / driver_abb).

    Returns nested JSON:
        { "Status": "Success", "Year": { year: { "Month": { month: { driver_abb: { event_type: count } } } } } }
    """
    storage_options = get_storage_options()
    bronze_container = os.getenv("BRONZE_CONTAINER", "bronze")
    silver_container = os.getenv("SILVER_CONTAINER", "silver")

    input_path = _abfs_path(bronze_container, "radio_bronze.json")
    log.info("Reading bronze data from %s", input_path)

    with fsspec.open(input_path, "r", **storage_options) as f:
        raw_data = json.load(f)

    if session_key:
        raw_data = [r for r in raw_data if r.get("session_key") == session_key]
        log.info("Filtered to %d recordings for session %s", len(raw_data), session_key)

    if not raw_data:
        log.warning("No records found in bronze.")
        return {"Status": "No data", "Year": {}}

    import torch
    import whisper
    device = "cuda" if torch.cuda.is_available() else "cpu"
    log.info("Loading Whisper '%s' on %s...", whisper_model_size, device)
    model = whisper.load_model(whisper_model_size, device=device)

    events = []
    failed = 0

    for i, record in enumerate(raw_data):
        url = record.get("recording_url")
        if not url:
            failed += 1
            continue

        try:
            r = requests.get(url, timeout=30)
            r.raise_for_status()
        except Exception as e:
            log.debug("Download failed for %s: %s", url, e)
            failed += 1
            continue

        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp:
            tmp.write(r.content)
            tmppath = tmp.name

        try:
            transcript = model.transcribe(tmppath)["text"].strip()
        except Exception as e:
            log.debug("Transcription failed: %s", e)
            failed += 1
            continue
        finally:
            if os.path.exists(tmppath):
                os.unlink(tmppath)

        if not transcript:
            failed += 1
            continue

        events.append(classify_radio(transcript, record))

        if (i + 1) % 100 == 0:
            log.info("  Processed %d/%d (failed: %d)", i + 1, len(raw_data), failed)

    log.info("Classified %d events (%d failed/skipped)", len(events), failed)

    if not events:
        return {"Status": "No events produced", "Year": {}}

    df = pd.json_normalize(events)
    df["date"] = pd.to_datetime(df["date"], utc=True, errors="coerce")
    df["year"] = df["date"].dt.year.astype(str)
    df["month"] = df["date"].dt.month.astype(str)

    output_path = _abfs_path(silver_container, "radio_silver.parquet")
    df.to_parquet(
        output_path,
        index=False,
        storage_options=storage_options,
        partition_cols=["year", "month", "driver_abb"],
    )
    log.info("Written parquet to %s", output_path)

    grouped = (
        df.groupby(["year", "month", "driver_abb", "primary_event_type"])
        .size()
        .reset_index(name="count")
    )

    nested = {}
    for _, row in grouped.iterrows():
        year, month, driver, event_type, count = (
            row["year"], row["month"], row["driver_abb"],
            row["primary_event_type"], row["count"],
        )
        nested.setdefault(year, {}).setdefault("Month", {}).setdefault(month, {}).setdefault(driver, {})[event_type] = int(count)

    return {"Status": "Success", "Year": nested}


# --- 8. Live Layer (commented out) ---
# def run_radio_live(session_key, poll_interval=10):
#     """
#     Live mode: poll OpenF1 for new radio, transcribe with Azure Speech Services,
#     classify, and push events to Azure Event Hub.
#     Requires: AZURE_SPEECH_KEY, AZURE_SPEECH_REGION,
#               EVENT_HUB_CONNECTION_STRING, EVENT_HUB_NAME
#     """
#     import time
#     speech_key    = os.getenv("AZURE_SPEECH_KEY", "")
#     speech_region = os.getenv("AZURE_SPEECH_REGION", "")
#     eh_conn       = os.getenv("EVENT_HUB_CONNECTION_STRING", "")
#     eh_name       = os.getenv("EVENT_HUB_NAME", "")
#
#     if not speech_key:
#         log.error("AZURE_SPEECH_KEY is required for live mode.")
#         return
#
#     def _transcribe_azure(audio_bytes):
#         endpoint = (
#             f"https://{speech_region}.stt.speech.microsoft.com"
#             "/speech/recognition/conversation/cognitiveservices/v1?language=en-US"
#         )
#         headers = {
#             "Ocp-Apim-Subscription-Key": speech_key,
#             "Content-Type": "audio/mpeg",
#             "Accept": "application/json",
#         }
#         resp = requests.post(endpoint, headers=headers, data=audio_bytes, timeout=30)
#         resp.raise_for_status()
#         result = resp.json()
#         return result.get("DisplayText", "").strip() if result.get("RecognitionStatus") == "Success" else None
#
#     def _push_to_event_hub(events):
#         if not eh_conn or not eh_name:
#             log.warning("[DRY RUN] Would push %d events.", len(events))
#             return
#         from azure.eventhub import EventHubProducerClient, EventData
#         producer = EventHubProducerClient.from_connection_string(eh_conn, eventhub_name=eh_name)
#         with producer:
#             batch = producer.create_batch()
#             for ev in events:
#                 try:
#                     batch.add(EventData(json.dumps(ev, default=str)))
#                 except ValueError:
#                     producer.send_batch(batch)
#                     batch = producer.create_batch()
#                     batch.add(EventData(json.dumps(ev, default=str)))
#             producer.send_batch(batch)
#         log.info("Pushed %d events to Event Hub.", len(events))
#
#     seen_urls = set()
#     log.info("Starting live radio polling for session %s (interval: %ds)...", session_key, poll_interval)
#     try:
#         while True:
#             try:
#                 radio = fetch_team_radio(session_key=session_key)
#             except Exception as e:
#                 log.error("Fetch failed: %s", e)
#                 time.sleep(poll_interval)
#                 continue
#
#             new_rows = radio[~radio["recording_url"].isin(seen_urls)]
#             events = []
#             for _, row in new_rows.iterrows():
#                 url = row["recording_url"]
#                 seen_urls.add(url)
#                 try:
#                     r = requests.get(url, timeout=30)
#                     r.raise_for_status()
#                 except Exception:
#                     continue
#                 transcript = _transcribe_azure(r.content)
#                 if transcript:
#                     events.append(classify_radio(transcript, row.to_dict()))
#             if events:
#                 _push_to_event_hub(events)
#             time.sleep(poll_interval)
#     except KeyboardInterrupt:
#         log.info("Live polling stopped.")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="F1 Radio Medallion Pipeline")
    parser.add_argument("--stage", choices=["bronze", "silver"], default="bronze",
                        help="Pipeline stage to run")
    parser.add_argument("--session-key", type=int, default=None,
                        help="Specific OpenF1 session key (optional)")
    parser.add_argument("--whisper-model", type=str, default="base",
                        choices=["tiny", "base", "small", "medium", "large"],
                        help="Whisper model size for silver transcription")
    args = parser.parse_args()

    if args.stage == "bronze":
        run_radio_bronze(session_key=args.session_key)
    elif args.stage == "silver":
        result = run_radio_silver(session_key=args.session_key, whisper_model_size=args.whisper_model)
        print(json.dumps(result, indent=2))
