import azure.functions as func
import logging
import json
import os

app = func.FunctionApp()


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
@app.event_hub_message_trigger(
    arg_name="event",
    event_hub_name="%EVENT_HUB_NAME%",
    connection="EVENT_HUB_CONNECTION_STRING"
)
def f1_live_ingest(event: func.EventHubEvent):
    """
    Triggered automatically whenever a message arrives on the Event Hub.
    Writes raw telemetry to ADLS bronze/live layer.
    """
    logging.info("Event Hub trigger fired — new F1 telemetry received")

    try:
        raw = event.get_body().decode("utf-8")
        logging.info(f"Raw event received, length={len(raw)}")

        storage_account = get_env("STORAGE_ACCOUNT_NAME", required=True)
        storage_key = get_env("STORAGE_ACCOUNT_KEY", required=True)
        container = get_env("ADLS_CONTAINER", required=True)
        directory = get_env("ADLS_DIRECTORY", required=True)

        from src.fetch_data import store_bronze
        store_bronze(
            raw_data=raw,
            storage_account=storage_account,
            storage_key=storage_key,
            container=container,
            directory=directory
        )

        logging.info("Bronze layer write successful")

    except Exception as e:
        logging.exception(f"Live ingest failed: {e}")
        raise


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
    """
    Runs the silver processing pipeline.
    POST /api/run_silver?target_driver=VER
    """
    logging.info("Silver pipeline triggered")

    try:
        driver = req.params.get("target_driver", get_env("TARGET_DRIVER", "LEC"))

        storage_account = get_env("STORAGE_ACCOUNT_NAME", required=True)
        storage_key = get_env("STORAGE_ACCOUNT_KEY", required=True)
        bronze_container = get_env("BRONZE_CONTAINER", required=True)
        silver_container = get_env("SILVER_CONTAINER", required=True)

        from src.silver import run_silver_pipeline
        output_path = run_silver_pipeline(
            target_driver=driver,
            storage_account=storage_account,
            storage_key=storage_key,
            bronze_container=bronze_container,
            silver_container=silver_container
        )

        return json_response({
            "status": "success",
            "message": "Silver pipeline completed",
            "driver": driver,
            "output_path": output_path
        }, 200)

    except Exception as e:
        logging.exception(f"Silver failed: {e}")
        return json_response({
            "status": "error",
            "message": str(e)
        }, 500)


# ── HISTORICAL: GOLD ───────────────────────────────────────────────────────────
@app.route(route="run_gold", methods=["POST"], auth_level=func.AuthLevel.FUNCTION)
def run_gold(req: func.HttpRequest) -> func.HttpResponse:
    """
    Runs the gold processing pipeline.
    POST /api/run_gold?target_driver=VER
    """
    logging.info("Gold pipeline triggered")

    try:
        driver = req.params.get("target_driver", get_env("TARGET_DRIVER", "LEC"))

        storage_account = get_env("STORAGE_ACCOUNT_NAME", required=True)
        storage_key = get_env("STORAGE_ACCOUNT_KEY", required=True)
        silver_container = get_env("SILVER_CONTAINER", required=True)
        gold_container = get_env("GOLD_CONTAINER", required=True)

        from src.gold import run_gold_pipeline
        output_path = run_gold_pipeline(
            target_driver=driver,
            storage_account=storage_account,
            storage_key=storage_key,
            silver_container=silver_container,
            gold_container=gold_container
        )

        return json_response({
            "status": "success",
            "message": "Gold pipeline completed",
            "driver": driver,
            "output_path": output_path
        }, 200)

    except Exception as e:
        logging.exception(f"Gold failed: {e}")
        return json_response({
            "status": "error",
            "message": str(e)
        }, 500)
