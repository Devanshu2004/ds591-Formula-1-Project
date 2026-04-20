import json
import os
import logging
import fsspec
import pandas as pd
import numpy as np
from textblob import TextBlob
from datetime import datetime

# --- 1. Configuration ---
def get_storage_options():
    return {
        "account_name": os.getenv("STORAGE_ACCOUNT_NAME"),
        "account_key": os.getenv("STORAGE_ACCOUNT_KEY")
    }

def _abfs_path(container, filename):
    return f"abfs://{container}@{os.getenv('STORAGE_ACCOUNT_NAME')}.dfs.core.windows.net/{filename}"

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

# Reverse lookup: Name -> Abbreviation
NAME_TO_ABB = {name.lower(): abb for abb, names in driver_abb.items() for name in names}

# --- 2. Cleaning Function ---
def get_clean_sentiment(text):

    if not text or not isinstance(text, str): return 0.0

    # Basic cleaning to remove non-ascii characters
    clean_text = text.encode('ascii', 'ignore').decode('ascii')

    return TextBlob(clean_text).sentiment.polarity

# --- 3. The Scoring Function ---
def calculate_life_score(sentiment, likes):
    """
    Converts sentiment and engagement into a 1(worst)-10(best) score.
    Weights: 70% Sentiment, 30% Engagement.
    """

    # Normalize sentiment (-1 to 1) to (1 to 10)
    sentiment_base = ((sentiment + 1) / 2) * 10
    
    # Scale likes so high-follower accounts don't skew the data
    engagement_base = np.clip(np.log1p(likes) / 2, 1, 10)
    
    return round((sentiment_base * 0.7) + (engagement_base * 0.3), 1)

# --- 4. The Main Processor---
def run_social_processor():
    storage_options = get_storage_options()
    bronze_container = os.getenv("BRONZE_CONTAINER", "bronze")
    silver_container = os.getenv("SILVER_CONTAINER", "silver")
    
    # Define paths in Azure
    input_path = _abfs_path(bronze_container, "social_media_bronze.json")
    logging.info(f"Accessing Bronze Data: {input_path}")
    
    try:
        with fsspec.open(input_path, **storage_options) as f:
            raw_data = json.load(f)
        
        processed_rows = []
        for entry in raw_data:
            full_name = entry.get('fullName', 'Unknown')
            abb = NAME_TO_ABB.get(full_name.lower(), "UNK")
            if abb == "UNK": continue

            posts = entry.get('latestPosts', [])
            for post in posts:
                ts_str = post.get('timestamp')
                if not ts_str: continue
                
                dt = datetime.fromisoformat(ts_str.replace('Z', '+00:00'))
                
                processed_rows.append({
                    "driver_abb": abb,
                    "year": str(dt.year),
                    "month": str(dt.month), 
                    "life_score": calculate_life_score(get_clean_sentiment(post.get('caption', '')), post.get('likesCount', 0))
                })

        df_silver = pd.DataFrame(processed_rows)

        # Save partitioned Parquet to Silver
        output_parquet = _abfs_path(silver_container, "social_media_analysis.parquet")
        df_silver.to_parquet(output_parquet, index=False, storage_options=storage_options, partition_cols=['year', 'month'])

        # Build Nested Dictionary Output
        grouped = df_silver.groupby(['year', 'month', 'driver_abb'])['life_score'].mean().round(1)
        
        nested_output = {}
        for (year, month, driver), score in grouped.items():
            if year not in nested_output:
                nested_output[year] = {"Month": {}}
            if month not in nested_output[year]["Month"]:
                nested_output[year]["Month"][month] = {}
            
            nested_output[year]["Month"][month][driver] = score

        final_response = {
            "Status": "Success",
            "Year": nested_output
        }

        return final_response

    except Exception as e:
        logging.error(f"Pipeline Error: {e}")
        raise

if __name__ == "__main__":
    run_social_processor()
