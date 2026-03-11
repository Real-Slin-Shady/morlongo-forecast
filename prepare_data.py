#!/usr/bin/env python3
"""
Prepare training data:
1. Aggregate 30-min station data to hourly
2. Download MeteoSwiss ICON forecast data from Open-Meteo
3. Align observations with forecasts
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

def aggregate_station_to_hourly(df):
    """Aggregate 30-min station data to hourly using mean."""

    # Filter to July 2025 onwards
    df = df[df['datetime'] >= '2025-07-10'].copy()

    # Create hour column (floor to hour)
    df['hour'] = df['datetime'].dt.floor('h')

    # Aggregate by hour - use mean, which handles missing values correctly
    agg_cols = ['temperature', 'humidity', 'rain', 'wind_speed',
                'wind_direction', 'gust_speed', 'gust_direction']

    hourly = df.groupby('hour').agg({
        col: 'mean' for col in agg_cols
    }).reset_index()

    hourly = hourly.rename(columns={'hour': 'datetime'})

    # For rain, we should sum rather than mean (accumulation)
    rain_sum = df.groupby('hour')['rain'].sum().reset_index()
    rain_sum = rain_sum.rename(columns={'hour': 'datetime', 'rain': 'rain_sum'})
    hourly = hourly.merge(rain_sum, on='datetime', how='left')
    hourly['rain'] = hourly['rain_sum']
    hourly = hourly.drop(columns=['rain_sum'])

    return hourly

def download_forecast_data():
    """Download MeteoSwiss ICON forecast data from Open-Meteo Historical Forecast API."""

    # Variables to download
    hourly_vars = [
        "temperature_2m",
        "relative_humidity_2m",
        "dew_point_2m",
        "precipitation",
        "rain",
        "pressure_msl",
        "surface_pressure",
        "wind_speed_10m",
        "wind_direction_10m",
        "wind_gusts_10m",
        "cloud_cover",
        "shortwave_radiation"
    ]

    # Download in chunks to avoid API limits
    all_data = []

    start = datetime.strptime(START_DATE, "%Y-%m-%d")
    end = datetime.strptime(END_DATE, "%Y-%m-%d")

    chunk_days = 60  # Download 60 days at a time
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
            "hourly": ",".join(hourly_vars),
            "models": "meteoswiss_icon_ch1",
            "timezone": "UTC"
        }

        response = requests.get(url, params=params)

        if response.status_code == 200:
            data = response.json()

            # Convert to DataFrame
            hourly_data = data.get("hourly", {})
            if hourly_data:
                chunk_df = pd.DataFrame(hourly_data)
                chunk_df['time'] = pd.to_datetime(chunk_df['time'])
                all_data.append(chunk_df)
                print(f"  Got {len(chunk_df)} rows")
        else:
            print(f"  Error: {response.status_code}")

        current = chunk_end + timedelta(days=1)
        time.sleep(0.5)  # Be nice to the API

    if all_data:
        forecast_df = pd.concat(all_data, ignore_index=True)
        forecast_df = forecast_df.drop_duplicates(subset=['time']).sort_values('time')
        return forecast_df

    return None

def main():
    print("=" * 60)
    print("Step 1: Load and aggregate station data to hourly")
    print("=" * 60)

    station_df = pd.read_csv('/Users/slin/Documents/Privat/Morlongo_forecast/station_data_combined.csv')
    station_df['datetime'] = pd.to_datetime(station_df['datetime'])

    print(f"Raw station data: {len(station_df)} rows")
    print(f"Date range: {station_df['datetime'].min()} to {station_df['datetime'].max()}")

    station_hourly = aggregate_station_to_hourly(station_df)

    print(f"\nHourly station data (from July 2025): {len(station_hourly)} rows")
    print(f"Date range: {station_hourly['datetime'].min()} to {station_hourly['datetime'].max()}")

    # Save hourly station data
    station_hourly.to_csv('/Users/slin/Documents/Privat/Morlongo_forecast/station_hourly.csv', index=False)
    print("Saved to station_hourly.csv")

    print("\nStation hourly data summary:")
    print(station_hourly.describe())

    print("\nMissing values in hourly data:")
    print(station_hourly.isnull().sum())

    print("\n" + "=" * 60)
    print("Step 2: Download MeteoSwiss ICON forecast data")
    print("=" * 60)

    forecast_df = download_forecast_data()

    if forecast_df is not None:
        print(f"\nForecast data: {len(forecast_df)} rows")
        print(f"Date range: {forecast_df['time'].min()} to {forecast_df['time'].max()}")

        # Save forecast data
        forecast_df.to_csv('/Users/slin/Documents/Privat/Morlongo_forecast/forecast_data.csv', index=False)
        print("Saved to forecast_data.csv")

        print("\nForecast data summary:")
        print(forecast_df.describe())

        print("\nMissing values in forecast data:")
        print(forecast_df.isnull().sum())

    print("\n" + "=" * 60)
    print("Step 3: Merge station observations with forecasts")
    print("=" * 60)

    # Rename forecast columns for clarity
    forecast_df = forecast_df.rename(columns={'time': 'datetime'})

    # Add prefix to forecast columns
    forecast_cols = [c for c in forecast_df.columns if c != 'datetime']
    forecast_df = forecast_df.rename(columns={c: f'fc_{c}' for c in forecast_cols})

    # Add prefix to station columns
    station_cols = [c for c in station_hourly.columns if c != 'datetime']
    station_hourly = station_hourly.rename(columns={c: f'obs_{c}' for c in station_cols})

    # Merge on datetime
    merged = pd.merge(station_hourly, forecast_df, on='datetime', how='inner')

    print(f"Merged dataset: {len(merged)} rows")
    print(f"Date range: {merged['datetime'].min()} to {merged['datetime'].max()}")

    # Save merged data
    merged.to_csv('/Users/slin/Documents/Privat/Morlongo_forecast/training_data.csv', index=False)
    print("Saved to training_data.csv")

    print("\nMerged data columns:")
    print(merged.columns.tolist())

    print("\nMissing values in merged data:")
    print(merged.isnull().sum())

if __name__ == '__main__':
    main()
