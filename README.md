# Morlongo Weather Forecast

ML-debiased weather forecast for Morlongo (46.02°N, 8.24°E, Ticino, Switzerland).

**Live forecast:** [https://real-slin-shady.github.io/morlongo-forecast](https://real-slin-shady.github.io/morlongo-forecast)

## How it works

1. **Raw forecast**: MeteoSwiss ICON-CH2 model via [Open-Meteo API](https://open-meteo.com)
2. **Debiasing**: XGBoost models trained on local station observations
3. **Updates**: Twice daily via GitHub Actions (06:00 & 18:00 CET)

## Model Performance

Validation: 10% daily hold-out (randomly held-out days)

| Variable | Test MAE | R² | Improvement |
|----------|----------|-----|-------------|
| Temperature | 0.73°C | 0.97 | 40% |
| Humidity | 5.0% | 0.82 | 10% |
| Wind speed | 0.98 km/h | 0.34 | 67% |
| Gust speed | 1.94 km/h | 0.39 | 67% |

Training period: July 2025 - March 2026 (~8 months)

## Files

```
├── generate_forecast.py   # Fetch & debias forecast
├── models_v2/             # Trained XGBoost models
├── docs/
│   ├── index.html         # Webpage
│   └── forecast.json      # Current forecast data
└── .github/workflows/     # Automated updates
```

## Local usage

```bash
# Generate forecast
pip install requests xgboost numpy
python generate_forecast.py

# View at docs/index.html
```

## Retrain models

```bash
# 1. Add new station data to station_data_combined.csv
# 2. Prepare data
python prepare_data_v2.py

# 3. Train models
python train_model_v2.py
```

## Data sources

- **Forecast**: [Open-Meteo](https://open-meteo.com) (MeteoSwiss ICON-CH2, 2km, 5-day)
- **Observations**: Local weather station (Netatmo)
