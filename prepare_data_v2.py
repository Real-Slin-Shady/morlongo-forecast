#!/usr/bin/env python3
"""
Prepare training data V2:
- Uses ALL available forecast variables
- Aggregates 30-min station data to hourly
"""

import pandas as pd
import numpy as np
import requests
from datetime import datetime, timedelta
import time

# Configuration
LAT = 46.021245
LON = 8.239861
START_DATE = "2025-07-10"
END_DATE = "2026-03-11"

# ALL available forecast variables
FORECAST_VARS = [
    "temperature_2m",
    "relative_humidity_2m",
    "dew_point_2m",
    "apparent_temperature",
    "precipitation",
    "rain",
    "snowfall",
    "snow_depth",
    "weather_code",
    "pressure_msl",
    "surface_pressure",
    "cloud_cover",
    "cloud_cover_low",
    "cloud_cover_mid",
    "cloud_cover_high",
    "et0_fao_evapotranspiration",
    "vapour_pressure_deficit",
    "wind_speed_10m",
    "wind_direction_10m",
    "wind_gusts_10m",
    "shortwave_radiation",
    "direct_radiation",
    "diffuse_radiation",
    "direct_normal_irradiance",
    "global_tilted_irradiance",
    "terrestrial_radiation",
    "cape",
    "convective_inhibition",
    "freezing_level_height",
    "is_day"
]

def aggregate_station_to_hourly(df):
    """Aggregate 30-min station data to hourly using mean."""
    df = df[df['datetime'] >= '2025-07-10'].copy()
    df['hour'] = df['datetime'].dt.floor('h')

    agg_cols = ['temperature', 'humidity', 'rain', 'wind_speed',
                'wind_direction', 'gust_speed', 'gust_direction']

    hourly = df.groupby('hour').agg({
        col: 'mean' for col in agg_cols
    }).reset_index()
    hourly = hourly.rename(columns={'hour': 'datetime'})

    # Rain should be summed
    rain_sum = df.groupby('hour')['rain'].sum().reset_index()
    rain_sum = rain_sum.rename(columns={'hour': 'datetime', 'rain': 'rain_sum'})
    hourly = hourly.merge(rain_sum, on='datetime', how='left')
    hourly['rain'] = hourly['rain_sum']
    hourly = hourly.drop(columns=['rain_sum'])

    return hourly

def download_forecast_data():
    """Download ALL MeteoSwiss ICON forecast variables."""
    all_data = []

    start = datetime.strptime(START_DATE, "%Y-%m-%d")
    end = datetime.strptime(END_DATE, "%Y-%m-%d")

    chunk_days = 60
    current = start

    while current < end:
        chunk_end = min(current + timedelta(days=chunk_days), end)
        print(f"Downloading {current.date()} to {chunk_end.date()}...")

        url = "https://historical-forecast-api.open-meteo.com/v1/forecast"
        params = {
            "latitude": LAT,
            "longitude": LON,
            "start_date": current.strftime("%Y-%m-%d"),
            "end_date": chunk_end.strftime("%Y-%m-%d"),
            "hourly": ",".join(FORECAST_VARS),
            "models": "meteoswiss_icon_ch1",
            "timezone": "UTC"
        }

        response = requests.get(url, params=params)

        if response.status_code == 200:
            data = response.json()
            hourly_data = data.get("hourly", {})
            if hourly_data:
                chunk_df = pd.DataFrame(hourly_data)
                chunk_df['time'] = pd.to_datetime(chunk_df['time'])
                all_data.append(chunk_df)
                print(f"  Got {len(chunk_df)} rows, {len(chunk_df.columns)} columns")
        else:
            print(f"  Error: {response.status_code}")

        current = chunk_end + timedelta(days=1)
        time.sleep(0.5)

    if all_data:
        forecast_df = pd.concat(all_data, ignore_index=True)
        forecast_df = forecast_df.drop_duplicates(subset=['time']).sort_values('time')
        return forecast_df
    return None

def main():
    print("="*60)
    print("PREPARE DATA V2 - ALL FORECAST VARIABLES")
    print("="*60)

    # Load and aggregate station data
    print("\nStep 1: Aggregate station data to hourly...")
    station_df = pd.read_csv('/Users/slin/Documents/Privat/Morlongo_forecast/station_data_combined.csv')
    station_df['datetime'] = pd.to_datetime(station_df['datetime'])
    station_hourly = aggregate_station_to_hourly(station_df)
    print(f"Hourly station data: {len(station_hourly)} rows")

    # Download forecast data
    print("\nStep 2: Download ALL forecast variables...")
    forecast_df = download_forecast_data()

    if forecast_df is None:
        print("Failed to download forecast data")
        return

    print(f"\nForecast data: {len(forecast_df)} rows, {len(forecast_df.columns)} columns")
    print(f"Variables: {list(forecast_df.columns)}")

    # Save forecast data
    forecast_df.to_csv('/Users/slin/Documents/Privat/Morlongo_forecast/forecast_data_v2.csv', index=False)

    # Merge
    print("\nStep 3: Merge datasets...")
    forecast_df = forecast_df.rename(columns={'time': 'datetime'})

    # Add prefixes
    forecast_cols = [c for c in forecast_df.columns if c != 'datetime']
    forecast_df = forecast_df.rename(columns={c: f'fc_{c}' for c in forecast_cols})

    station_cols = [c for c in station_hourly.columns if c != 'datetime']
    station_hourly = station_hourly.rename(columns={c: f'obs_{c}' for c in station_cols})

    merged = pd.merge(station_hourly, forecast_df, on='datetime', how='inner')

    print(f"Merged dataset: {len(merged)} rows, {len(merged.columns)} columns")

    # Save
    merged.to_csv('/Users/slin/Documents/Privat/Morlongo_forecast/training_data_v2.csv', index=False)
    print(f"Saved to training_data_v2.csv")

    print(f"\nForecast features: {len(forecast_cols)}")
    for col in forecast_cols:
        print(f"  fc_{col}")

if __name__ == '__main__':
    main()
