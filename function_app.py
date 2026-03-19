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
