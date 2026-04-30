'''
import azure.functions as func
import logging
import os
# Import your existing logic from the src folder
from src.telemetry_analysis import your_main_analysis_function 

app = func.FunctionApp()

# This decorator defines the Timer Trigger (Running every hour)
@app.timer_trigger(schedule="0 0 * * * *", arg_name="mytimer", run_on_startup=False)
def f1_live_process(mytimer: func.TimerRequest) -> None:
    if mytimer.past_due:
        logging.info('The timer is past due!') [cite: 125]

    logging.info("Starting F1 Live Telemetry Process...")
    
    try:
        # Call your existing analytics code here
        # Example: passing a specific GP or year
        result = your_main_analysis_function(year=2024, gp='Monaco')
        logging.info(f"Analysis Complete: {result}")
        
    except Exception as e:
        logging.error(f"Error during F1 Process: {e}")
'''

import azure.functions as func
import logging
import json
import os 
# from src.live_casting import main as live_casting_main
# from src.social_media_analysis import run_social_processor

app = func.FunctionApp()


# This decorator defines the Timer Trigger (Running every hour)
# @app.timer_trigger(schedule="*/10 * * * * *", arg_name="mytimer", run_on_startup=False)
# def f1_live_process(mytimer: func.TimerRequest) -> None:
#     from src.live_casting import main as live_casting_main
#     if mytimer.past_due:
#         logging.info('The timer is past due!')

#     logging.info("Starting F1 Live Telemetry Process...")
    
#     try:
#         # Call your existing analytics code here
#         # Example: passing a specific GP or year
#         result = live_casting_main()
#         logging.info(f"Analysis Complete: {result}")
        
#     except Exception as e:
#         logging.error(f"Error during F1 Process: {e}")

def get_env(name: str, default: str = None, required: bool = False) -> str:
    """
    Safely read environment variables.
    If required=True and value is missing, raise a clear error.
    """
    value = os.getenv(name, default)
    if required and (value is None or str(value).strip() == ""):
        raise ValueError(f"Missing required environment variable: {name}")
    return value


def json_response(payload: dict, status_code: int = 200) -> func.HttpResponse:
    return func.HttpResponse(
        json.dumps(payload),
        status_code=status_code,
        mimetype="application/json"
    )


# ── LIVE TELEMETRY: EVENT HUB TRIGGER ──────────────────────────────────────────
# @app.event_hub_message_trigger(
#     arg_name="event",
#     event_hub_name="%EVENT_HUB_NAME%",
#     connection="EVENT_HUB_CONNECTION_STRING"
# )
# def f1_live_ingest(event: func.EventHubEvent):
#     """
#     Triggered automatically whenever a message arrives on the Event Hub.
#     Writes raw telemetry to ADLS bronze/live layer.
#     """
#     logging.info("Event Hub trigger fired — new F1 telemetry received")

#     try:
#         raw = event.get_body().decode("utf-8")
#         logging.info(f"Raw event received, length={len(raw)}")

#         storage_account = get_env("STORAGE_ACCOUNT_NAME", required=True)
#         storage_key = get_env("STORAGE_ACCOUNT_KEY", required=True)
#         container = get_env("ADLS_CONTAINER", required=True)
#         directory = get_env("ADLS_DIRECTORY", required=True)

#         from src.fetch_data import store_bronze
#         store_bronze(
#             raw_data=raw,
#             storage_account=storage_account,
#             storage_key=storage_key,
#             container=container,
#             directory=directory
#         )

#         logging.info("Bronze layer write successful")

#     except Exception as e:
#         logging.exception(f"Live ingest failed: {e}")
#         raise


# ── LIVE TRANSMISSION: HTTP TRIGGER ────────────────────────────────────────────
@app.route(route="run_live", methods=["POST"], auth_level=func.AuthLevel.FUNCTION)
def run_live(req: func.HttpRequest) -> func.HttpResponse:
    """
    Manually triggers a live data fetch.
    POST /api/run_live

    Optional JSON body:
    {
      "year": "2026",
      "gp": "Japanese Grand Prix",
      "session": "R"
    }
    """
    logging.info("run_live HTTP trigger fired")

    try:
        try:
            body = req.get_json()
        except ValueError:
            body = {}

        year = body.get("year", get_env("F1_YEAR", "2026"))
        gp = body.get("gp", get_env("F1_GRAND_PRIX", "Japanese Grand Prix"))
        session = body.get("session", get_env("F1_SESSION_TYPE", "R"))

        username = get_env("F1_USERNAME", required=True)
        password = get_env("F1_PASSWORD", required=True)
        storage_account = get_env("STORAGE_ACCOUNT_NAME", required=True)
        storage_key = get_env("STORAGE_ACCOUNT_KEY", required=True)
        container = get_env("ADLS_CONTAINER", required=True)
        directory = get_env("ADLS_DIRECTORY", required=True)

        from src.fetch_data import run_live_fetch
        run_live_fetch(
            year=year,
            gp=gp,
            session=session,
            username=username,
            password=password,
            storage_account=storage_account,
            storage_key=storage_key,
            container=container,
            directory=directory
        )

        return json_response({
            "status": "success",
            "message": "Live fetch completed",
            "year": year,
            "gp": gp,
            "session": session
        }, 200)

    except Exception as e:
        logging.exception(f"run_live failed: {e}")
        return json_response({
            "status": "error",
            "message": str(e)
        }, 500)


# ── HISTORICAL: SILVER ─────────────────────────────────────────────────────────
@app.route(route="run_silver", methods=["POST"], auth_level=func.AuthLevel.FUNCTION)
def run_silver(req: func.HttpRequest) -> func.HttpResponse:
    logging.info("Silver pipeline triggered")

    try:
        try:
            body = req.get_json()
        except ValueError:
            body = {}

        year = req.params.get("year") or body.get("year")
        session_type = req.params.get("session_type") or body.get("session_type")
        race_location = req.params.get("race_location") or body.get("race_location")

        storage_account = get_env("STORAGE_ACCOUNT_NAME", required=True)
        storage_key = get_env("STORAGE_ACCOUNT_KEY", required=True)
        bronze_container = "bronze"
        silver_container = "silver"

        from src.silver import run_silver_pipeline

        output_path = run_silver_pipeline(
            year=year,                         # can be single, comma string, or list
            session_type=session_type,
            race_location=race_location,       # can be single, comma string, or list
            storage_account=storage_account,
            storage_key=storage_key,
            bronze_container=bronze_container,
            silver_container=silver_container
        )

        return json_response({
            "status": "success",
            "message": "Silver pipeline completed",
            "input": {
                "year": year,
                "session_type": session_type,
                "race_location": race_location
            },
            "output_path": output_path
        }, 200)

    except Exception as e:
        logging.exception(f"Silver failed: {e}")
        return json_response({
            "status": "error",
            "message": str(e)
        }, 500)

@app.route(route="health", methods=["GET","POST"], auth_level=func.AuthLevel.ANONYMOUS)
def health(req: func.HttpRequest) -> func.HttpResponse:
    import sys
    pkgs = [m for m in sys.modules.keys()]
    return func.HttpResponse(f"OK - Python {sys.version}", status_code=200)

# ── HISTORICAL: GOLD ───────────────────────────────────────────────────────────
import azure.durable_functions as df

# ── GOLD: HTTP Starter ─────────────────────────────────────────
@app.route(route="run_gold", methods=["POST"], auth_level=func.AuthLevel.FUNCTION)
@app.durable_client_input(client_name="client")
async def run_gold(req: func.HttpRequest, client) -> func.HttpResponse:
    try:
        body = req.get_json()
    except ValueError:
        body = {}

    driver = req.params.get("target_driver") or body.get("target_driver", "LEC")

    instance_id = await client.start_new("run_gold_orchestrator", None, driver)
    return client.create_check_status_response(req, instance_id)


# ── GOLD: Orchestrator ─────────────────────────────────────────
@app.orchestration_trigger(context_name="context")
def run_gold_orchestrator(context: df.DurableOrchestrationContext):
    driver = context.get_input()
    result = yield context.call_activity("run_gold_activity", driver)
    return result


# ── GOLD: Activity ─────────────────────────────────────────────
@app.activity_trigger(input_name="driver")
def run_gold_activity(driver: str) -> str:
    from src.gold import run_gold_pipeline
    output_path = run_gold_pipeline(target_driver=driver)
    return str(output_path)

# ── RADIO: BRONZE ─────────────────────────────────────────────────────────────
@app.route(route="run_radio_bronze", methods=["POST"], auth_level=func.AuthLevel.FUNCTION)
def run_radio_bronze(req: func.HttpRequest) -> func.HttpResponse:
    """
    Fetch team radio from OpenF1 and store raw JSON to ADLS bronze.
    POST /api/run_radio_bronze

    Optional JSON body:
    {
      "session_key": 9158
    }
    """
    logging.info("Radio bronze pipeline triggered")

    try:
        try:
            body = req.get_json()
        except ValueError:
            body = {}

        session_key = body.get("session_key", None)

        from src.radio_data import run_radio_bronze as _run_radio_bronze
        output_path = _run_radio_bronze(session_key=session_key)

        return json_response({
            "status": "success",
            "message": "Radio bronze pipeline completed",
            "session_key": session_key,
            "output_path": output_path,
        }, 200)

    except Exception as e:
        logging.exception(f"Radio bronze failed: {e}")
        return json_response({"status": "error", "message": str(e)}, 500)


# ── RADIO: SILVER ──────────────────────────────────────────────────────────────
@app.route(route="run_radio_silver", methods=["POST"], auth_level=func.AuthLevel.FUNCTION)
def run_radio_silver(req: func.HttpRequest) -> func.HttpResponse:
    """
    Transcribe + classify radio from bronze, write partitioned parquet to silver.
    POST /api/run_radio_silver

    Optional JSON body:
    {
      "session_key": 9158,
      "whisper_model": "base"
    }
    """
    logging.info("Radio silver pipeline triggered")

    try:
        try:
            body = req.get_json()
        except ValueError:
            body = {}

        session_key   = body.get("session_key", None)
        whisper_model = body.get("whisper_model", "base")

        from src.radio_data import run_radio_silver as _run_radio_silver
        result = _run_radio_silver(session_key=session_key, whisper_model_size=whisper_model)

        return json_response({
            "status": "success",
            "message": "Radio silver pipeline completed",
            "session_key": session_key,
            "radio_events": result,
        }, 200)

    except Exception as e:
        logging.exception(f"Radio silver failed: {e}")
        return json_response({"status": "error", "message": str(e)}, 500)


# ── SOCIAL MEDIA ANALYSIS ──────────────────────────────────────────────────────
@app.route(route="process_social", methods=["POST"], auth_level=func.AuthLevel.FUNCTION)
def run_social_media_pipeline(req: func.HttpRequest) -> func.HttpResponse:
    from src.social_media_analysis import run_social_processor
    """
    HTTP Trigger to run the Social Media Analysis.
    Moves data from social_media_bronze.json -> social_media_analysis.parquet
    """
    logging.info("Social Media Analysis pipeline triggered via HTTP.")

    try:
        # Calls the script that handles mapping, life scores, and parquet export
        final_scores = run_social_processor()

        return func.HttpResponse(
            json.dumps({
                "status": "success",
                "message": "Social media silver pipeline completed",
                "life_scores": final_scores
            }),
            mimetype="application/json",
            status_code=200
        )

    except Exception as e:
        logging.exception(f"Social Media Pipeline failed: {e}")
        return func.HttpResponse(
            json.dumps({
                "status": "error",
                "message": str(e)
            }),
            mimetype="application/json",
            status_code=500
        )

# ── MODEL: TRAIN ───────────────────────────────────────────────────────────────
@app.route(route="run_model", methods=["POST"], auth_level=func.AuthLevel.FUNCTION)
def run_model(req: func.HttpRequest) -> func.HttpResponse:
    """
    Trains the F1 race prediction model on historic gold data.
    Called right after run_gold in the ADF pipeline.

    Pipeline order:
        run_radio_bronze
               ↓
        run_silver | run_radio_silver | process_social  (parallel)
               ↓
            run_gold
               ↓
            run_model  ← this function

    What it does:
        1. Loads gold parquet for all drivers from Azure gold container
        2. Trains one Bi-LSTM channel per driver
        3. Extracts embeddings from all channels
        4. Trains Random Forest aggregator
        5. Saves all models to platinum/models/ on Azure

    Saved artifacts:
        platinum/models/channel_{DRIVER}.pt     (one per driver)
        platinum/models/random_forest.joblib

    POST /api/run_model
    No request body required.
    """
    logging.info("run_model HTTP trigger fired — historic training pipeline")

    try:
        import torch
        from src.model import train_all_channels, train_random_forest

        save_dir = "/tmp/model_weights"
        device   = torch.device("cpu")

        logging.info("Step 1: Training driver channels...")
        train_all_channels(save_dir, device)

        logging.info("Step 2: Training Random Forest on frozen embeddings...")
        train_random_forest(save_dir, device)

        return json_response({
            "status":  "success",
            "message": "Historic model training complete",
            "artifacts": {
                "driver_channels": "platinum/models/channel_{DRIVER}.pt",
                "random_forest":   "platinum/models/random_forest.joblib",
            },
            "config": {
                "sequence_length_timesteps":    300,
                "sequence_length_seconds":      30,
                "prediction_horizon_timesteps": 3000,
                "prediction_horizon_seconds":   300,
                "sampling_rate":                0.1,
            }
        }, 200)

    except Exception as e:
        logging.exception(f"run_model failed: {e}")
        return json_response({"status": "error", "message": str(e)}, 500)
