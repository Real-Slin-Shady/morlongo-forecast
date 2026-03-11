#!/usr/bin/env python3
"""
Combine station data from multiple CSV files into a single clean dataset.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import re

def parse_weather_csv(filepath):
    """Parse a Weather.csv file with variable column structure."""

    with open(filepath, 'r') as f:
        lines = f.readlines()

    # Find the header row (contains 'Timestamp')
    header_idx = None
    for i, line in enumerate(lines):
        if 'Timestamp' in line:
            header_idx = i
            break

    if header_idx is None:
        print(f"Could not find header in {filepath}")
        return None

    # Read the data starting from the header
    df = pd.read_csv(filepath, skiprows=header_idx, skipinitialspace=True)

    # Clean column names (remove extra spaces and special chars)
    df.columns = [str(c).strip() for c in df.columns]

    # Identify columns we care about
    cols_mapping = {}
    for col in df.columns:
        col_lower = col.lower()
        if 'timestamp' in col_lower and 'timezone' not in col_lower:
            cols_mapping['timestamp'] = col
        elif 'timezone' in col_lower:
            cols_mapping['datetime'] = col
        elif 'temperature' in col_lower:
            cols_mapping['temperature'] = col
        elif 'humidity' in col_lower:
            cols_mapping['humidity'] = col
        elif 'rain' in col_lower:
            cols_mapping['rain'] = col
        elif 'wind strength' in col_lower:
            cols_mapping['wind_speed'] = col
        elif 'wind angle' in col_lower:
            cols_mapping['wind_direction'] = col
        elif 'gust strength' in col_lower:
            cols_mapping['gust_speed'] = col
        elif 'gust angle' in col_lower:
            cols_mapping['gust_direction'] = col

    # Create standardized dataframe
    result = pd.DataFrame()

    if 'timestamp' in cols_mapping:
        result['timestamp'] = pd.to_numeric(df[cols_mapping['timestamp']], errors='coerce')

    if 'datetime' in cols_mapping:
        result['datetime_str'] = df[cols_mapping['datetime']]

    for var in ['temperature', 'humidity', 'rain', 'wind_speed', 'wind_direction', 'gust_speed', 'gust_direction']:
        if var in cols_mapping:
            result[var] = pd.to_numeric(df[cols_mapping[var]], errors='coerce')
        else:
            result[var] = np.nan

    # Drop rows without valid timestamp
    result = result.dropna(subset=['timestamp'])
    result['timestamp'] = result['timestamp'].astype(int)

    print(f"  Parsed {filepath.name}: {len(result)} rows, cols found: {list(cols_mapping.keys())}")

    return result

def main():
    base_path = Path('/Users/slin/Documents/Privat/Morlongo_forecast/extracted')

    all_data = []

    for i in range(6):
        part_path = base_path / f'part_{i}' / 'Weather.csv'
        if part_path.exists():
            df = parse_weather_csv(part_path)
            if df is not None:
                all_data.append(df)

    if not all_data:
        print("No data found!")
        return

    # Combine all data
    combined = pd.concat(all_data, ignore_index=True)

    # Remove duplicates based on timestamp
    combined = combined.drop_duplicates(subset=['timestamp'], keep='first')

    # Sort by timestamp
    combined = combined.sort_values('timestamp').reset_index(drop=True)

    # Convert timestamp to datetime
    combined['datetime'] = pd.to_datetime(combined['timestamp'], unit='s')

    # Drop the string datetime column if it exists
    if 'datetime_str' in combined.columns:
        combined = combined.drop(columns=['datetime_str'])

    # Reorder columns
    cols_order = ['timestamp', 'datetime', 'temperature', 'humidity', 'rain',
                  'wind_speed', 'wind_direction', 'gust_speed', 'gust_direction']
    combined = combined[cols_order]

    # Save combined data
    output_path = Path('/Users/slin/Documents/Privat/Morlongo_forecast/station_data_combined.csv')
    combined.to_csv(output_path, index=False)

    print(f"\nCombined dataset:")
    print(f"  Total rows: {len(combined)}")
    print(f"  Date range: {combined['datetime'].min()} to {combined['datetime'].max()}")
    print(f"  Saved to: {output_path}")

    # Show data summary
    print("\nData summary:")
    print(combined.describe())

    print("\nMissing values:")
    print(combined.isnull().sum())

if __name__ == '__main__':
    main()
