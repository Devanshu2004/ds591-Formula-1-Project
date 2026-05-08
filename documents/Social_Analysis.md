# Social Media Analysis Pipeline

The social media analysis pipeline converts raw driver social media posts into a sentiment-based engagement score for downstream modeling. It reads from Bronze and writes a cleaned, aggregated Silver output.

## Files

- `src/social_media_analysis.py` - reads bronze social data, scores each post, and writes aggregated silver output.

## Bronze: Raw Social Media Data

`run_social_processor()` reads `bronze/social_media_bronze.json` from ADLS.

Data expected per entry:

- `fullName` — driver's full name (matched to driver abbreviation via internal lookup).
- `latestPosts` — list of posts, each containing:
  - `caption` — post text used for sentiment analysis.
  - `likesCount` — engagement count used for scaling.
  - `timestamp` — ISO 8601 timestamp used to derive year and month.

Drivers not matched in the abbreviation lookup are skipped.

## Silver: Scored and Aggregated Data

`run_social_processor()` computes a `life_score` for each post and aggregates to a per-driver, per-month mean.

### Sentiment Scoring

Each post's caption is cleaned of non-ASCII characters and passed to TextBlob to produce a polarity score in the range `[-1, 1]`.

### Life Score Calculation

Each post is assigned a `life_score` on a 1–10 scale:

```
sentiment_base  = ((polarity + 1) / 2) * 10        # maps [-1, 1] → [1, 10]
engagement_base = clip(log1p(likesCount) / 2, 1, 10) # log-scaled to avoid skew
life_score      = (sentiment_base × 0.7) + (engagement_base × 0.3)
```

Weights: 70% sentiment, 30% engagement. Log scaling on likes prevents high-follower accounts from dominating.

### Aggregation

Scores are grouped by `(year, month, driver_abb)` and averaged. The output is a nested JSON dictionary:

```json
{
  "Status": "Success",
  "<year>": {
    "<month>": {
      "<DRIVER_ABB>": <mean_life_score>
    }
  }
}
```

Silver output is stored as `silver/social_media_silver.json`.

## Output

```text
silver/social_media_silver.json
```

This file is merged into the Gold layer in `gold.py` as the `social_life_score` column, keyed by `(race_year, race_month, driver_abbreviation)`.

## Run Commands

Install dependencies:

```bash
pip install -r requirements.txt
```

Run social media processing:

```bash
python src/social_media_analysis.py
```

Optional Azure Function endpoint:

```bash
POST /api/process_social
```
