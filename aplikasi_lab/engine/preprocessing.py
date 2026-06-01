"""
Preprocessing Service — Pipeline Final
=========================================
Pipeline preprocessing yang identik dengan hasil evaluasi.
Digunakan hanya sementara saat training (tidak persist file).

Steps:
  1. Resample ke 1 jam
  2. Interpolasi linear (limit=6h)
  3. Drop NaN
  4. Feature Engineering: temporal, lag, rolling, diff, interaction
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List

from core.config import PARAMETERS, TARGET, EXCLUDED_CURRENT_FEATURES, TEST_RATIO


# ==============================================================================
# Pipeline Preprocessing Lengkap
# ==============================================================================
def preprocess_for_training(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Pipeline preprocessing lengkap untuk training.

    Args:
        df: DataFrame dengan kolom datetime + PARAMETERS

    Returns:
        dict: dataframe (processed), feature_columns, log
    """
    log_steps = []

    raw_records = len(df)
    
    # --- Step 1: Resample ke 1 jam ---
    df = df.copy()
    df["datetime"] = pd.to_datetime(df["datetime"])
    df = df.set_index("datetime")
    df = df.resample("1h").mean()
    log_steps.append({"step": "Resample 1h", "records_after": len(df)})

    # --- Step 2: Interpolasi missing values ---
    n_before = df[PARAMETERS].isna().sum().sum()
    df[PARAMETERS] = df[PARAMETERS].interpolate(method="linear", limit=6)
    n_after = df[PARAMETERS].isna().sum().sum()
    log_steps.append({"step": "Interpolasi linear", "values_filled": int(n_before - n_after)})

    # --- Step 3: Drop NaN yang tersisa ---
    n_before_drop = len(df)
    df = df.dropna(subset=PARAMETERS)
    df = df.reset_index()
    log_steps.append({"step": "Drop remaining NaN", "dropped": n_before_drop - len(df), "records_after": len(df)})


    # --- Step 4: Temporal features ---
    hour = df["datetime"].dt.hour
    df["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    df["hour_cos"] = np.cos(2 * np.pi * hour / 24)

    dow = df["datetime"].dt.dayofweek
    df["dow_sin"] = np.sin(2 * np.pi * dow / 7)
    df["dow_cos"] = np.cos(2 * np.pi * dow / 7)
    df["is_weekend"] = (dow >= 5).astype(int)

    # Rush hour features
    df["rush_hour_pagi"] = ((hour >= 6) & (hour <= 9)).astype(int)
    df["rush_hour_sore"] = ((hour >= 16) & (hour <= 20)).astype(int)

    # --- Step 5: Lag features PM2.5 ---
    for lag in [1, 2, 3, 6, 12, 24]:
        df[f"pm25_lag_{lag}h"] = df[TARGET].shift(lag)

    # Lag features parameter lain
    for param in ["pm1", "temperature", "relativehumidity", "um003"]:
        for lag in [1, 3]:
            df[f"{param}_lag_{lag}h"] = df[param].shift(lag)

    # --- Step 6: Rolling features ---
    for window in [3, 6, 12, 24]:
        df[f"pm25_rolling_mean_{window}h"] = (
            df[TARGET].shift(1).rolling(window=window, min_periods=1).mean()
        )
        df[f"pm25_rolling_std_{window}h"] = (
            df[TARGET].shift(1).rolling(window=window, min_periods=1).std()
        )

    df["pm25_rolling_min_24h"] = df[TARGET].shift(1).rolling(24, min_periods=1).min()
    df["pm25_rolling_max_24h"] = df[TARGET].shift(1).rolling(24, min_periods=1).max()

    # --- Step 7: Diff features ---
    df["pm25_diff_1h"] = df[TARGET].shift(1).diff(1)
    df["pm25_diff_3h"] = df[TARGET].shift(1).diff(3)
    df["temperature_diff_1h"] = df["temperature"].shift(1).diff(1)
    df["humidity_diff_1h"] = df["relativehumidity"].shift(1).diff(1)

    # --- Step 8: Interaction features ---
    df["pm1_pm25_ratio"] = df["pm1"] / (df["pm25"] + 1e-6)

    # --- Step 9: Drop NaN dari feature engineering ---
    n_before_final = len(df)
    df = df.dropna().reset_index(drop=True)
    n_dropped = n_before_final - len(df)
    log_steps.append({"step": "Drop NaN from features", "dropped": n_dropped, "records_after": len(df)})

    # Tentukan feature columns
    exclude_cols = {"datetime", TARGET} | set(EXCLUDED_CURRENT_FEATURES)
    feature_cols = [c for c in df.columns if c not in exclude_cols]

    return {
        "dataframe": df,
        "feature_columns": feature_cols,
        "log": log_steps,
        "raw_records": raw_records,
        "clean_records": len(df),
    }


def build_features_from_history(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build fitur dari data historis untuk prediksi real-time.
    Memerlukan minimal 25 baris data berurutan.

    Args:
        df: DataFrame dengan kolom recorded_at/datetime + PARAMETERS

    Returns:
        DataFrame dengan fitur lengkap, atau None jika data kurang
    """
    if len(df) < 25:
        return None

    df = df.copy()

    # Normalisasi kolom datetime
    if "recorded_at" in df.columns and "datetime" not in df.columns:
        df = df.rename(columns={"recorded_at": "datetime"})
    df["datetime"] = pd.to_datetime(df["datetime"])
    df = df.sort_values("datetime").reset_index(drop=True)

    # Lag features PM2.5
    for lag in [1, 2, 3, 6, 12, 24]:
        df[f"pm25_lag_{lag}h"] = df["pm25"].shift(lag)

    # Lag features parameter lain
    for param in ["pm1", "temperature", "relativehumidity", "um003"]:
        for lag in [1, 3]:
            df[f"{param}_lag_{lag}h"] = df[param].shift(lag)

    # Rolling statistics PM2.5
    for window in [3, 6, 12, 24]:
        df[f"pm25_rolling_mean_{window}h"] = (
            df["pm25"].shift(1).rolling(window=window, min_periods=1).mean()
        )
        df[f"pm25_rolling_std_{window}h"] = (
            df["pm25"].shift(1).rolling(window=window, min_periods=1).std()
        )

    df["pm25_rolling_min_24h"] = df["pm25"].shift(1).rolling(24, min_periods=1).min()
    df["pm25_rolling_max_24h"] = df["pm25"].shift(1).rolling(24, min_periods=1).max()

    # Temporal features
    hour = df["datetime"].dt.hour
    df["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    df["hour_cos"] = np.cos(2 * np.pi * hour / 24)

    dow = df["datetime"].dt.dayofweek
    df["dow_sin"] = np.sin(2 * np.pi * dow / 7)
    df["dow_cos"] = np.cos(2 * np.pi * dow / 7)
    df["is_weekend"] = (dow >= 5).astype(int)

    # Rush hour
    df["rush_hour_pagi"] = ((hour >= 6) & (hour <= 9)).astype(int)
    df["rush_hour_sore"] = ((hour >= 16) & (hour <= 20)).astype(int)

    # Diff features
    df["pm25_diff_1h"] = df[TARGET].shift(1).diff(1)
    df["pm25_diff_3h"] = df[TARGET].shift(1).diff(3)
    df["temperature_diff_1h"] = df["temperature"].shift(1).diff(1)
    df["humidity_diff_1h"] = df["relativehumidity"].shift(1).diff(1)

    # Interaction
    df["pm1_pm25_ratio"] = df["pm1"] / (df["pm25"] + 1e-6)

    return df

def analyze_data_quality(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Analisis kualitas data secara menyeluruh.

    Returns:
        dict dengan info missing values, outliers, gaps, distribusi
    """
    result = {}

    # --- Missing Values ---
    missing = {}
    for col in PARAMETERS:
        if col in df.columns:
            n_miss = int(df[col].isna().sum())
            pct = round(n_miss / len(df) * 100, 2) if len(df) > 0 else 0
            missing[col] = {"count": n_miss, "percentage": pct}
    result["missing_values"] = missing

    # --- Duplikasi ---
    if "datetime" in df.columns:
        dup_count = int(df.duplicated(subset=["datetime"]).sum())
    else:
        dup_count = int(df.duplicated().sum())
    result["duplicate_count"] = dup_count

    # --- Outliers (IQR method) ---
    outliers = {}
    for col in PARAMETERS:
        if col in df.columns and df[col].notna().sum() > 0:
            q1 = df[col].quantile(0.25)
            q3 = df[col].quantile(0.75)
            iqr = q3 - q1
            lower = q1 - 1.5 * iqr
            upper = q3 + 1.5 * iqr
            n_out = int(((df[col] < lower) | (df[col] > upper)).sum())
            outliers[col] = {
                "count": n_out,
                "percentage": round(n_out / len(df) * 100, 2),
                "lower_bound": round(float(lower), 2),
                "upper_bound": round(float(upper), 2),
            }
    result["outlier_info"] = outliers

    # --- Timestamp Gaps ---
    gaps = []
    total_gaps = 0
    if "datetime" in df.columns and len(df) > 1:
        df_sorted = df.sort_values("datetime")
        time_diffs = df_sorted["datetime"].diff()
        expected_interval = pd.Timedelta(hours=1)

        for i in range(1, len(df_sorted)):
            diff = time_diffs.iloc[i]
            if diff > expected_interval * 1.5:  # Lebih dari 1.5 jam gap
                gaps.append({
                    "start": df_sorted["datetime"].iloc[i - 1].isoformat(),
                    "end": df_sorted["datetime"].iloc[i].isoformat(),
                    "gap_hours": round(diff.total_seconds() / 3600, 1),
                })
                total_gaps += 1
    result["timestamp_gaps"] = gaps[:50]  # Limit 50 gaps
    result["total_gaps"] = total_gaps

    # --- Distribusi PM2.5 ---
    distribution = {}
    if TARGET in df.columns and df[TARGET].notna().sum() > 0:
        pm25 = df[TARGET].dropna()
        distribution["pm25"] = {
            "mean": round(float(pm25.mean()), 2),
            "std": round(float(pm25.std()), 2),
            "min": round(float(pm25.min()), 2),
            "max": round(float(pm25.max()), 2),
            "median": round(float(pm25.median()), 2),
            "q25": round(float(pm25.quantile(0.25)), 2),
            "q75": round(float(pm25.quantile(0.75)), 2),
        }

        # Histogram data (untuk grafik)
        hist, bin_edges = np.histogram(pm25, bins=30)
        distribution["pm25_histogram"] = {
            "counts": hist.tolist(),
            "bin_edges": [round(float(b), 2) for b in bin_edges],
        }

    # Distribusi per waktu
    if "datetime" in df.columns and TARGET in df.columns:
        df_ts = df.dropna(subset=[TARGET]).copy()
        if len(df_ts) > 0:
            # Per jam
            hourly = df_ts.groupby(df_ts["datetime"].dt.hour)[TARGET].mean()
            distribution["by_hour"] = {
                "hours": hourly.index.tolist(),
                "means": [round(float(v), 2) for v in hourly.values],
            }

            # Per hari dalam seminggu
            daily = df_ts.groupby(df_ts["datetime"].dt.dayofweek)[TARGET].mean()
            day_names = ["Sen", "Sel", "Rab", "Kam", "Jum", "Sab", "Min"]
            distribution["by_day"] = {
                "days": [day_names[i] for i in daily.index],
                "means": [round(float(v), 2) for v in daily.values],
            }

            # Per bulan
            monthly = df_ts.groupby(df_ts["datetime"].dt.month)[TARGET].mean()
            month_names = ["Jan", "Feb", "Mar", "Apr", "Mei", "Jun",
                           "Jul", "Ags", "Sep", "Okt", "Nov", "Des"]
            distribution["by_month"] = {
                "months": [month_names[i - 1] for i in monthly.index],
                "means": [round(float(v), 2) for v in monthly.values],
            }

    result["distribution_stats"] = distribution

    return result


def split_chronological(df: pd.DataFrame, feature_cols: List[str], test_ratio: float = TEST_RATIO):
    """
    Split data secara kronologis (TANPA shuffle).
    """
    X = df[feature_cols]
    y = df[TARGET]

    split_idx = int(len(df) * (1 - test_ratio))
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    dt_train = df["datetime"].iloc[:split_idx]
    dt_test = df["datetime"].iloc[split_idx:]

    return X_train, X_test, y_train, y_test, dt_train, dt_test
