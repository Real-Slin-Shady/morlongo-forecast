#!/usr/bin/env python3
"""
Train XGBoost models to debias MeteoSwiss forecasts for local station.
Each observed variable gets its own model.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb
import joblib
import json
from pathlib import Path

# Output directory for models
MODEL_DIR = Path('/Users/slin/Documents/Privat/Morlongo_forecast/models')
MODEL_DIR.mkdir(exist_ok=True)

# Forecast features to use as predictors
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

# Target variables (observed)
TARGET_VARS = {
    'obs_temperature': 'temperature',
    'obs_humidity': 'humidity',
    'obs_rain': 'rain',
    'obs_wind_speed': 'wind_speed',
    'obs_gust_speed': 'gust_speed',
}

# Add time features
def add_time_features(df):
    """Add cyclical time features."""
    df = df.copy()
    df['hour'] = df['datetime'].dt.hour
    df['month'] = df['datetime'].dt.month
    df['day_of_year'] = df['datetime'].dt.dayofyear

    # Cyclical encoding
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)

    return df

def train_model_for_target(df, target_col, target_name, features):
    """Train XGBoost model for a single target variable."""

    print(f"\n{'='*60}")
    print(f"Training model for: {target_name}")
    print(f"{'='*60}")

    # Filter rows where target is not missing
    valid_mask = df[target_col].notna()

    # Also filter rows where all forecast features are not missing
    for feat in FORECAST_FEATURES:
        if feat in df.columns:
            valid_mask &= df[feat].notna()

    df_valid = df[valid_mask].copy()

    print(f"Valid rows: {len(df_valid)} / {len(df)}")

    if len(df_valid) < 100:
        print(f"Not enough data for {target_name}, skipping...")
        return None

    # Prepare features and target
    X = df_valid[features].values
    y = df_valid[target_col].values

    # Split data (80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    print(f"Training set: {len(X_train)}, Test set: {len(X_test)}")

    # Train XGBoost model
    model = xgb.XGBRegressor(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        min_child_weight=3,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1
    )

    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=False
    )

    # Evaluate
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)

    train_mae = mean_absolute_error(y_train, y_pred_train)
    test_mae = mean_absolute_error(y_test, y_pred_test)
    train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
    test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
    test_r2 = r2_score(y_test, y_pred_test)

    print(f"\nResults:")
    print(f"  Train MAE: {train_mae:.3f}")
    print(f"  Test MAE:  {test_mae:.3f}")
    print(f"  Train RMSE: {train_rmse:.3f}")
    print(f"  Test RMSE:  {test_rmse:.3f}")
    print(f"  Test R²:    {test_r2:.3f}")

    # Feature importance
    importance = dict(zip(features, model.feature_importances_))
    sorted_importance = sorted(importance.items(), key=lambda x: x[1], reverse=True)

    print(f"\nTop 5 features:")
    for feat, imp in sorted_importance[:5]:
        print(f"  {feat}: {imp:.3f}")

    # Compare with raw forecast (if applicable)
    if target_name == 'temperature':
        raw_fc = df_valid['fc_temperature_2m'].values
        raw_mae = mean_absolute_error(y, raw_fc)
        print(f"\nRaw forecast MAE: {raw_mae:.3f}")
        print(f"Improvement: {(raw_mae - test_mae) / raw_mae * 100:.1f}%")
    elif target_name == 'humidity':
        raw_fc = df_valid['fc_relative_humidity_2m'].values
        raw_mae = mean_absolute_error(y, raw_fc)
        print(f"\nRaw forecast MAE: {raw_mae:.3f}")
        print(f"Improvement: {(raw_mae - test_mae) / raw_mae * 100:.1f}%")
    elif target_name == 'wind_speed':
        raw_fc = df_valid['fc_wind_speed_10m'].values
        raw_mae = mean_absolute_error(y, raw_fc)
        print(f"\nRaw forecast MAE: {raw_mae:.3f}")
        print(f"Improvement: {(raw_mae - test_mae) / raw_mae * 100:.1f}%")
    elif target_name == 'gust_speed':
        raw_fc = df_valid['fc_wind_gusts_10m'].values
        raw_mae = mean_absolute_error(y, raw_fc)
        print(f"\nRaw forecast MAE: {raw_mae:.3f}")
        print(f"Improvement: {(raw_mae - test_mae) / raw_mae * 100:.1f}%")

    # Save model
    model_path = MODEL_DIR / f'xgb_{target_name}.json'
    model.save_model(str(model_path))
    print(f"\nModel saved to: {model_path}")

    # Return metrics
    return {
        'target': target_name,
        'train_samples': len(X_train),
        'test_samples': len(X_test),
        'train_mae': train_mae,
        'test_mae': test_mae,
        'train_rmse': train_rmse,
        'test_rmse': test_rmse,
        'test_r2': test_r2,
        'feature_importance': dict(sorted_importance)
    }

def main():
    print("Loading training data...")
    df = pd.read_csv('/Users/slin/Documents/Privat/Morlongo_forecast/training_data.csv')
    df['datetime'] = pd.to_datetime(df['datetime'])

    print(f"Total rows: {len(df)}")

    # Add time features
    df = add_time_features(df)

    # Define all features
    time_features = ['hour_sin', 'hour_cos', 'month_sin', 'month_cos']
    all_features = FORECAST_FEATURES + time_features

    # Store results
    all_results = {}

    # Train model for each target
    for target_col, target_name in TARGET_VARS.items():
        result = train_model_for_target(df, target_col, target_name, all_features)
        if result:
            all_results[target_name] = result

    # Save feature list and results
    config = {
        'features': all_features,
        'targets': list(TARGET_VARS.values()),
        'location': {
            'lat': 46.021245,
            'lon': 8.239861,
            'name': 'Morlongo'
        },
        'training_period': {
            'start': str(df['datetime'].min()),
            'end': str(df['datetime'].max())
        }
    }

    with open(MODEL_DIR / 'config.json', 'w') as f:
        json.dump(config, f, indent=2)

    with open(MODEL_DIR / 'metrics.json', 'w') as f:
        json.dump(all_results, f, indent=2, default=float)

    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"\nModels trained: {len(all_results)}")
    for name, result in all_results.items():
        print(f"\n{name}:")
        print(f"  Test MAE: {result['test_mae']:.3f}")
        print(f"  Test R²:  {result['test_r2']:.3f}")

    print(f"\nAll models saved to: {MODEL_DIR}")

if __name__ == '__main__':
    main()
