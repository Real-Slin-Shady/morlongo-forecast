#!/usr/bin/env python3
"""
Train XGBoost models V2:
- Uses ALL forecast variables
- 10% DAILY hold-out validation (not random hours)
"""

import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb
import json
from pathlib import Path

MODEL_DIR = Path('/Users/slin/Documents/Privat/Morlongo_forecast/models_v2')
MODEL_DIR.mkdir(exist_ok=True)

# Target variables
TARGET_VARS = {
    'obs_temperature': 'temperature',
    'obs_humidity': 'humidity',
    'obs_rain': 'rain',
    'obs_wind_speed': 'wind_speed',
    'obs_gust_speed': 'gust_speed',
}

def add_time_features(df):
    """Add cyclical time features."""
    df = df.copy()
    df['hour'] = df['datetime'].dt.hour
    df['month'] = df['datetime'].dt.month
    df['day_of_year'] = df['datetime'].dt.dayofyear

    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    df['doy_sin'] = np.sin(2 * np.pi * df['day_of_year'] / 365)
    df['doy_cos'] = np.cos(2 * np.pi * df['day_of_year'] / 365)

    return df

def daily_holdout_split(df, holdout_fraction=0.10, seed=42):
    """
    Split data by randomly holding out 10% of DAYS.
    All hours from a held-out day go to test set.
    """
    df = df.copy()
    df['date'] = df['datetime'].dt.date

    # Get unique dates
    unique_dates = df['date'].unique()
    n_holdout = max(1, int(len(unique_dates) * holdout_fraction))

    # Randomly select holdout dates
    np.random.seed(seed)
    holdout_dates = np.random.choice(unique_dates, size=n_holdout, replace=False)

    # Split
    test_mask = df['date'].isin(holdout_dates)
    train_df = df[~test_mask].copy()
    test_df = df[test_mask].copy()

    print(f"  Total days: {len(unique_dates)}")
    print(f"  Holdout days: {len(holdout_dates)} ({len(holdout_dates)/len(unique_dates)*100:.1f}%)")
    print(f"  Train hours: {len(train_df)}, Test hours: {len(test_df)}")

    return train_df, test_df, holdout_dates

def train_model_for_target(df, target_col, target_name, feature_cols):
    """Train XGBoost model with daily hold-out validation."""

    print(f"\n{'='*60}")
    print(f"Training model for: {target_name}")
    print(f"{'='*60}")

    # Filter valid rows
    valid_mask = df[target_col].notna()
    for feat in feature_cols:
        if feat in df.columns:
            valid_mask &= df[feat].notna()

    df_valid = df[valid_mask].copy()
    print(f"Valid rows: {len(df_valid)} / {len(df)}")

    if len(df_valid) < 100:
        print(f"Not enough data, skipping...")
        return None

    # Daily hold-out split
    train_df, test_df, holdout_dates = daily_holdout_split(df_valid, holdout_fraction=0.10)

    X_train = train_df[feature_cols].values
    y_train = train_df[target_col].values
    X_test = test_df[feature_cols].values
    y_test = test_df[target_col].values

    # Train model
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

    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)

    # Evaluate
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)

    train_mae = mean_absolute_error(y_train, y_pred_train)
    test_mae = mean_absolute_error(y_test, y_pred_test)
    train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
    test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
    test_r2 = r2_score(y_test, y_pred_test)

    print(f"\nResults (10% daily hold-out):")
    print(f"  Train MAE:  {train_mae:.3f}")
    print(f"  Test MAE:   {test_mae:.3f}")
    print(f"  Train RMSE: {train_rmse:.3f}")
    print(f"  Test RMSE:  {test_rmse:.3f}")
    print(f"  Test R²:    {test_r2:.3f}")

    # Feature importance (top 10)
    importance = dict(zip(feature_cols, model.feature_importances_))
    sorted_importance = sorted(importance.items(), key=lambda x: x[1], reverse=True)

    print(f"\nTop 10 features:")
    for feat, imp in sorted_importance[:10]:
        print(f"  {feat}: {imp:.4f}")

    # Compare with raw forecast
    raw_fc_map = {
        'temperature': 'fc_temperature_2m',
        'humidity': 'fc_relative_humidity_2m',
        'wind_speed': 'fc_wind_speed_10m',
        'gust_speed': 'fc_wind_gusts_10m',
        'rain': 'fc_precipitation'
    }

    if target_name in raw_fc_map and raw_fc_map[target_name] in test_df.columns:
        raw_fc = test_df[raw_fc_map[target_name]].values
        raw_mae = mean_absolute_error(y_test, raw_fc)
        improvement = (raw_mae - test_mae) / raw_mae * 100
        print(f"\nRaw forecast MAE: {raw_mae:.3f}")
        print(f"Improvement: {improvement:.1f}%")

    # Save model
    model_path = MODEL_DIR / f'xgb_{target_name}.json'
    model.save_model(str(model_path))
    print(f"\nModel saved to: {model_path}")

    return {
        'target': target_name,
        'train_samples': len(X_train),
        'test_samples': len(X_test),
        'holdout_days': len(holdout_dates),
        'train_mae': float(train_mae),
        'test_mae': float(test_mae),
        'train_rmse': float(train_rmse),
        'test_rmse': float(test_rmse),
        'test_r2': float(test_r2),
        'top_features': dict(sorted_importance[:10])
    }

def main():
    print("Loading training data V2...")
    df = pd.read_csv('/Users/slin/Documents/Privat/Morlongo_forecast/training_data_v2.csv')
    df['datetime'] = pd.to_datetime(df['datetime'])

    print(f"Total rows: {len(df)}")
    print(f"Total columns: {len(df.columns)}")

    # Add time features
    df = add_time_features(df)

    # Get all forecast feature columns
    fc_features = [c for c in df.columns if c.startswith('fc_')]
    time_features = ['hour_sin', 'hour_cos', 'month_sin', 'month_cos', 'doy_sin', 'doy_cos']
    all_features = fc_features + time_features

    print(f"\nForecast features: {len(fc_features)}")
    print(f"Time features: {len(time_features)}")
    print(f"Total features: {len(all_features)}")

    # Train models
    all_results = {}
    for target_col, target_name in TARGET_VARS.items():
        result = train_model_for_target(df, target_col, target_name, all_features)
        if result:
            all_results[target_name] = result

    # Save config and metrics
    config = {
        'features': all_features,
        'n_features': len(all_features),
        'targets': list(TARGET_VARS.values()),
        'validation': '10% daily hold-out',
        'location': {'lat': 46.021245, 'lon': 8.239861, 'name': 'Morlongo'},
        'training_period': {'start': str(df['datetime'].min()), 'end': str(df['datetime'].max())}
    }

    with open(MODEL_DIR / 'config.json', 'w') as f:
        json.dump(config, f, indent=2)

    with open(MODEL_DIR / 'metrics.json', 'w') as f:
        json.dump(all_results, f, indent=2, default=float)

    # Summary
    print("\n" + "="*60)
    print("SUMMARY - 10% Daily Hold-out Validation")
    print("="*60)
    print(f"\nTotal features used: {len(all_features)}")
    print(f"Models trained: {len(all_results)}")

    for name, result in all_results.items():
        print(f"\n{name}:")
        print(f"  Test MAE: {result['test_mae']:.3f}")
        print(f"  Test R²:  {result['test_r2']:.3f}")
        print(f"  Holdout days: {result['holdout_days']}")

    print(f"\nModels saved to: {MODEL_DIR}")

if __name__ == '__main__':
    main()
