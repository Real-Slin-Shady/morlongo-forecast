# Morlongo Weather Forecast

ML-debiased weather forecast with real-time observations for Morlongo (46.02°N, 8.24°E, Ticino, Switzerland).

**Live:** [https://real-slin-shady.github.io/morlongo-forecast](https://real-slin-shady.github.io/morlongo-forecast)

## Features

- **Real-time observations** from Netatmo weather station (hourly)
- **5-day forecast** from MeteoSwiss ICON-CH2 via Open-Meteo
- **ML debiasing** using XGBoost trained on local station data
- **Auto-updates** via GitHub Actions (hourly)

## Model Performance

| Variable | Test MAE | Improvement vs Raw |
|----------|----------|-------------------|
| Temperature | 0.73°C | 40% |
| Humidity | 5.0% | 10% |
| Wind speed | 0.98 km/h | 67% |

## Setup Netatmo Integration

### 1. Create Netatmo App

1. Go to [dev.netatmo.com/apps](https://dev.netatmo.com/apps)
2. Create a new app
3. Copy `client_id` and `client_secret`

### 2. Get Refresh Token

```bash
pip install requests
python get_netatmo_token.py
```

Follow the prompts - it will open your browser for authorization.

### 3. Add GitHub Secrets

Go to: `github.com/Real-Slin-Shady/morlongo-forecast/settings/secrets/actions`

Add these secrets:
- `NETATMO_CLIENT_ID` → your client_id
- `NETATMO_CLIENT_SECRET` → your client_secret
- `NETATMO_REFRESH_TOKEN` → from step 2

Done! Observations will appear within an hour.

## Files

```
├── generate_forecast.py     # Fetch & debias forecast
├── fetch_observations.py    # Fetch Netatmo data
├── get_netatmo_token.py     # One-time token setup
├── models_v2/               # XGBoost models
├── docs/
│   ├── index.html           # Website
│   ├── forecast.json        # Current forecast
│   └── observations.json    # Observation history (7 days)
└── .github/workflows/       # Hourly automation
```

## Local Development

```bash
# Install dependencies
pip install requests xgboost numpy pandas scikit-learn

# Generate forecast
python generate_forecast.py

# Fetch observations (requires env vars)
export NETATMO_CLIENT_ID=...
export NETATMO_CLIENT_SECRET=...
export NETATMO_REFRESH_TOKEN=...
python fetch_observations.py

# View site
open docs/index.html
```

## Retrain Models

```bash
# Add new station data to station_data_combined.csv
python prepare_data_v2.py
python train_model_v2.py
```

## Data Sources

- **Observations**: [Netatmo](https://www.netatmo.com) weather station
- **Forecast**: [Open-Meteo](https://open-meteo.com) (MeteoSwiss ICON-CH2)
