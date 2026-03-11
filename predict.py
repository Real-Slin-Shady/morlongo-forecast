#!/usr/bin/env python3
"""
Apply trained XGBoost models to debias current MeteoSwiss forecasts.
Fetches the latest forecast and applies local corrections.
"""

import pandas as pd
import numpy as np
import requests
import xgboost as xgb
import json
from pathlib import Path
from datetime import datetime

# Configuration
LAT = 46.021245
LON = 8.239861
MODEL_DIR = Path('/Users/slin/Documents/Privat/Morlongo_forecast/models')

# Forecast features (must match training)
FORECAST_FEATURES = [
    'fc_temperature_2m',
    'fc_relative_humidity_2m',
    'fc_dew_point_2m',
    'fc_precipitation',
    'fc_rain',
    'fc_pressure_msl',
    'fc_surface_pressure',
    'fc_wind_speed_10m',
    'fc_wind_direction_10m',
    'fc_wind_gusts_10m',
    'fc_cloud_cover',
    'fc_shortwave_radiation'
]

# API variable names (without fc_ prefix)
API_VARS = [f.replace('fc_', '') for f in FORECAST_FEATURES]

def fetch_current_forecast(days=5):
    """Fetch current MeteoSwiss ICON forecast from Open-Meteo."""

    print(f"Fetching {days}-day forecast for location ({LAT}, {LON})...")

    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": LAT,
        "longitude": LON,
        "hourly": ",".join(API_VARS),
        "models": "meteoswiss_icon_ch2",  # CH2 has 5-day forecast
        "forecast_days": days,
        "timezone": "Europe/Zurich"
    }

    response = requests.get(url, params=params)

    if response.status_code != 200:
        print(f"Error fetching forecast: {response.status_code}")
        return None

    data = response.json()

    # Convert to DataFrame
    hourly = data.get("hourly", {})
    df = pd.DataFrame(hourly)
    df['time'] = pd.to_datetime(df['time'])
    df = df.rename(columns={'time': 'datetime'})

    # Add fc_ prefix to match model features
    for var in API_VARS:
        if var in df.columns:
            df[f'fc_{var}'] = df[var]
            df = df.drop(columns=[var])

    print(f"Forecast data: {len(df)} hours")
    print(f"Period: {df['datetime'].min()} to {df['datetime'].max()}")

    return df

def add_time_features(df):
    """Add cyclical time features."""
    df = df.copy()
    df['hour'] = df['datetime'].dt.hour
    df['month'] = df['datetime'].dt.month

    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)

    return df

def load_models():
    """Load all trained XGBoost models."""
    models = {}

    targets = ['temperature', 'humidity', 'rain', 'wind_speed', 'gust_speed']

    for target in targets:
        model_path = MODEL_DIR / f'xgb_{target}.json'
        if model_path.exists():
            model = xgb.XGBRegressor()
            model.load_model(str(model_path))
            models[target] = model
            print(f"Loaded model: {target}")
        else:
            print(f"Model not found: {target}")

    return models

def apply_debiasing(df, models):
    """Apply debiasing models to forecast data."""

    df = add_time_features(df)

    # Prepare features
    time_features = ['hour_sin', 'hour_cos', 'month_sin', 'month_cos']
    all_features = FORECAST_FEATURES + time_features

    # Check for missing features
    missing = [f for f in all_features if f not in df.columns]
    if missing:
        print(f"Warning: Missing features: {missing}")
        for f in missing:
            df[f] = 0

    X = df[all_features].values

    # Apply each model
    results = df[['datetime']].copy()

    # Add raw forecast values
    results['raw_temperature'] = df['fc_temperature_2m']
    results['raw_humidity'] = df['fc_relative_humidity_2m']
    results['raw_precipitation'] = df['fc_precipitation']
    results['raw_wind_speed'] = df['fc_wind_speed_10m']
    results['raw_gust_speed'] = df['fc_wind_gusts_10m']

    # Add debiased values
    for target, model in models.items():
        pred = model.predict(X)

        # Clip to reasonable ranges
        if target == 'temperature':
            pred = np.clip(pred, -30, 50)
        elif target == 'humidity':
            pred = np.clip(pred, 0, 100)
        elif target in ['rain', 'wind_speed', 'gust_speed']:
            pred = np.clip(pred, 0, None)

        results[f'debiased_{target}'] = pred

    return results

def main():
    print("=" * 60)
    print("MORLONGO FORECAST DEBIASING")
    print(f"Location: {LAT}, {LON}")
    print(f"Time: {datetime.now()}")
    print("=" * 60)

    # Fetch current forecast
    forecast_df = fetch_current_forecast(days=5)

    if forecast_df is None:
        print("Failed to fetch forecast")
        return

    # Load models
    print("\nLoading models...")
    models = load_models()

    if not models:
        print("No models found!")
        return

    # Apply debiasing
    print("\nApplying debiasing...")
    results = apply_debiasing(forecast_df, models)

    # Save results
    output_path = Path('/Users/slin/Documents/Privat/Morlongo_forecast/debiased_forecast.csv')
    results.to_csv(output_path, index=False)
    print(f"\nSaved to: {output_path}")

    # Display summary
    print("\n" + "=" * 60)
    print("DEBIASED FORECAST SUMMARY")
    print("=" * 60)

    print("\nNext 24 hours:")
    print(results.head(24).to_string(index=False))

    # Show correction statistics
    print("\n" + "=" * 60)
    print("CORRECTION STATISTICS (mean absolute difference)")
    print("=" * 60)

    if 'debiased_temperature' in results.columns:
        temp_diff = (results['debiased_temperature'] - results['raw_temperature']).abs().mean()
        print(f"Temperature: {temp_diff:.2f}°C average correction")

    if 'debiased_humidity' in results.columns:
        hum_diff = (results['debiased_humidity'] - results['raw_humidity']).abs().mean()
        print(f"Humidity: {hum_diff:.2f}% average correction")

    if 'debiased_wind_speed' in results.columns:
        wind_diff = (results['debiased_wind_speed'] - results['raw_wind_speed']).abs().mean()
        print(f"Wind speed: {wind_diff:.2f} km/h average correction")

if __name__ == '__main__':
    main()
