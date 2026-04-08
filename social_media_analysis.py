import json
import os
import pandas as pd
import numpy as np
from pathlib import Path
from textblob import TextBlob

# --- 1. Configuration ---
DATA_ROOT = os.getenv("DATA_ROOT", "data")
BRONZE_PATH = Path(DATA_ROOT) / "bronze"
SILVER_PATH = Path(DATA_ROOT) / "silver"

# --- 2. Cleaning & NLP Function ---
def get_clean_sentiment(text):
    """
    Cleans text and analyzes sentiment. Fulfills the 'Semantic Analysis' requirement.
    """

    if not text: return 0.0

    # Basic cleaning to remove non-ascii characters
    clean_text = text.encode('ascii', 'ignore').decode('ascii')

    return TextBlob(clean_text).sentiment.polarity

# --- 3. The Scoring Function ---
def calculate_life_score(sentiment, likes):
    """
    Heuristic Model: Converts sentiment and engagement into a 1-10 score.
    Weights: 70% Sentiment, 30% Engagement.
    """

    # Normalize sentiment (-1 to 1) to (1 to 10)
    sentiment_base = ((sentiment + 1) / 2) * 10
    
    # Scale likes so high-follower accounts don't skew the data
    engagement_base = np.clip(np.log1p(likes) / 2, 1, 10)
    
    return round((sentiment_base * 0.7) + (engagement_base * 0.3), 1)

# --- 4. The Main Processor---
def run_social_processor():
    input_file = BRONZE_PATH / "social-scraper.json"
    
    if not input_file.exists():
        print(f"Error: Missing {input_file}")
        return

    with open(input_file, 'r') as f:
        data = json.load(f)

    processed_rows = []
    for driver in data:
        username = driver.get('username')
        for post in driver.get('latestPosts', []):
            likes = post.get('likesCount', 0)
            sentiment = get_clean_sentiment(post.get('caption', ''))
            
            # Applying the Model
            score = calculate_life_score(sentiment, likes)
            
            processed_rows.append({
                "driver": username,
                "sentiment": sentiment,
                "life_score": score,
                "likes": likes
            })

    # Convert to DataFrame (The 'Appropriate Ingestion' into Silver)
    df_silver = pd.DataFrame(processed_rows)
    df_silver.to_csv(SILVER_PATH / "social_analysis.csv", index=False)
    df_silver.to_parquet(SILVER_PATH / "social_analysis.parquet", index=False)
    
    # Output the final Dictionary for the team
    final_dict = df_silver.groupby('driver')['life_score'].mean().to_dict()
    print(f"Final Driver Scores: {final_dict}")
    return final_dict

if __name__ == "__main__":
    run_social_processor()
