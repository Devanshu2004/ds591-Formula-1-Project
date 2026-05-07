# Historical Analysis Pipeline

The historical analysis pipeline prepares Formula 1 race data for downstream modeling. It is built around three stages: Bronze ingestion, Silver cleaning, and Gold feature engineering.

## Files

- `src/fetch_data.py` - collects historical FastF1 race data.
- `src/silver.py` - cleans and aligns telemetry, lap, and weather data.
- `src/gold.py` - creates target-driver model features and merges contextual data.

## Bronze: Raw Historical Data

`fetch_data.py` loads historical FastF1 sessions and stores raw race data by year, race, session, and driver.

Data collected:

- Driver telemetry: speed, gear, RPM, DRS, brake, position, coordinates, and relative distance.
- Lap data: lap number, tire compound, tire life, team, and track status.
- Weather data: air temperature, track temperature, humidity, rainfall, pressure, wind speed, and wind direction.

## Silver: Cleaned Race Data

`silver.py` converts raw Bronze files into race-level Silver datasets.

Main steps:

- Converts session times into seconds.
- Cleans invalid gear values.
- Aligns telemetry with lap data using time-based joins.
- Merges nearest weather data.
- Adds race metadata such as race year, race location, race ID, and race date.
- Removes unreliable initial timestamps where position data may be invalid.

Silver race files are stored by year and session type.

## Gold: Model-Ready Features

`gold.py` reads Silver race files and builds a driver-specific Gold dataset.

Main features created:

- Target driver telemetry, position, tire, lap, team, and weather fields.
- Driver-ahead and driver-behind features.
- Distance gap features between the target driver and nearby competitors.
- Encoded categorical features for tire compound, team, gear, and track status.
- Boolean indicators such as `has_driver_ahead` and `has_driver_behind`.

The Gold layer also includes fallback logic to reduce missing values when exact ahead/behind joins fail due to timing differences.

## Social Media and Radio Merge

After Gold feature engineering, `gold.py` adds contextual data from the Silver layer:

- `silver/social_media_silver.json` is merged into Gold as `social_life_score`.
- `silver/radio.parquet` is aggregated and merged into Gold as radio feature columns.
- `radio_data_available` indicates whether radio data was matched for a driver, year, and race.
- Radio features are matched using driver abbreviation, race year, and Grand Prix name.

This gives the final Gold dataset both on-track race features and off-track/contextual signals.

## Output

The final Gold dataset is saved per target driver:

```text
model_weights/gold/{session_type}/{target_driver}_gold.parquet
```

This dataset is used by the predictive model to learn how recent race conditions and contextual signals influence a driver's future race position.

## Run Commands

Install dependencies:

```bash
pip install -r requirements.txt
```

Run Bronze ingestion:

```bash
python src/fetch_data.py
```

Run Silver processing:

```bash
python src/silver.py
```

Run Gold processing:

```bash
python src/gold.py
```

Optional Azure Function endpoints:

```bash
POST /api/run_silver
POST /api/run_gold
```
