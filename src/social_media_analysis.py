import json
import os
import pandas as pd
from pathlib import Path
from textblob import TextBlob

# --- 1. Configuration ---
DATA_ROOT = os.getenv("DATA_ROOT", "data")
BRONZE_PATH = Path(DATA_ROOT) / "bronze"
SILVER_PATH = Path(DATA_ROOT) / "silver"

# Ensure the silver directory exists before we save to it
os.makedirs(SILVER_PATH, exist_ok=True)

# --- 2. Semantic Analysis Functions ---
def get_sentiment(text):
    if not text: return 0.0
    # Clean text slightly for better analysis
    clean_text = text.encode('ascii', 'ignore').decode('ascii')
    return TextBlob(clean_text).sentiment.polarity

# --- 3. Process Bronze to Silver ---
def run_semantic_silver_pipeline():
    # Path to the raw file in the bronze layer
    input_file = BRONZE_PATH / "social-scraper.json"
    
    if not input_file.exists():
        print(f"Error: Could not find {input_file}")
        return

    print(f"Reading raw data from {BRONZE_PATH}...")
    with open(input_file, 'r') as f:
        data = json.load(f)

    processed_posts = []

    for driver in data:
        username = driver.get('username')
        posts = driver.get('latestPosts', [])
        
        for post in posts:
            caption = post.get('caption', '')
            sentiment = get_sentiment(caption)
            
            processed_posts.append({
                "driver": username,
                "timestamp": post.get('timestamp'),
                "likes": post.get('likesCount'),
                "sentiment_score": sentiment,
                "is_positive": sentiment > 0.1,
                "caption_snippet": caption[:50]
            })

    # Convert to DataFrame
    df_silver = pd.DataFrame(processed_posts)
    
    # Save to the Team's Silver Path
    output_path = SILVER_PATH / "instagram_sentiment_analysis.csv"
    df_silver.to_csv(output_path, index=False)
    
    print(f"Semantic Analysis saved to: {output_path}")

if __name__ == "__main__":
    run_semantic_silver_pipeline()
