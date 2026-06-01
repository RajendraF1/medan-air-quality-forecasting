"""
Prediction Service - Prediksi Rekursif 24 Jam Ke Depan
======================================================
Model memprediksi satu jam ke depan, lalu hasil prediksi dimasukkan kembali
ke histori untuk membangun fitur jam berikutnya.
"""

from datetime import timedelta
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from aplikasi_lab.engine.aqi import calculate_aqi
from aplikasi_lab.engine.fetching import load_local_dataset
from aplikasi_lab.engine.training import load_model
from core.config import PARAMETERS, PREDICTION_HORIZON


def _resolve_feature_columns(model, feature_columns: Optional[List[str]]) -> List[str]:
    """Gunakan urutan fitur dari model jika tersedia."""
    model_features = getattr(model, "feature_names_in_", None)
    if model_features is not None:
        return [str(feature) for feature in model_features]
    return list(feature_columns or [])


def _latest_value(df: pd.DataFrame, column: str, default: float = 0.0) -> float:
    """Ambil nilai numerik terakhir yang valid."""
    if column not in df.columns:
        return default

    values = pd.to_numeric(df[column], errors="coerce").dropna()
    if values.empty:
        return default
    return float(values.iloc[-1])


def _lag_value(df: pd.DataFrame, column: str, lag: int, default: float = 0.0) -> float:
    """Ambil nilai lag dari histori sebelum target prediksi."""
    if column not in df.columns or len(df) < lag:
        return default

    value = pd.to_numeric(df[column], errors="coerce").iloc[-lag]
    if pd.isna(value):
        return _latest_value(df, column, default)
    return float(value)


def _diff_value(df: pd.DataFrame, column: str, periods: int, default: float = 0.0) -> float:
    """Hitung perubahan nilai dari histori terakhir."""
    if column not in df.columns or len(df) <= periods:
        return default

    values = pd.to_numeric(df[column], errors="coerce")
    latest = values.iloc[-1]
    previous = values.iloc[-(periods + 1)]
    if pd.isna(latest) or pd.isna(previous):
        return default
    return float(latest - previous)


def _rolling_stats(df: pd.DataFrame, column: str, window: int) -> Dict[str, float]:
    """Hitung statistik rolling dari histori sebelum target prediksi."""
    if column not in df.columns:
        return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}

    values = pd.to_numeric(df[column], errors="coerce").dropna().tail(window)
    if values.empty:
        return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}

    return {
        "mean": float(values.mean()),
        "std": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
        "min": float(values.min()),
        "max": float(values.max()),
    }


def _build_feature_row(
    history: pd.DataFrame,
    target_time,
    feature_columns: List[str],
) -> pd.DataFrame:
    """Bangun satu baris fitur untuk target_time dari histori yang tersedia."""
    row: Dict[str, Any] = {}
    history = history.sort_values("datetime").reset_index(drop=True)

    hour = target_time.hour
    day_of_week = target_time.weekday()
    latest_pm25 = _latest_value(history, "pm25")
    latest_pm1 = _latest_value(history, "pm1")

    row.update({
        "pm1": latest_pm1,
        "relativehumidity": _latest_value(history, "relativehumidity"),
        "temperature": _latest_value(history, "temperature"),
        "um003": _latest_value(history, "um003"),
        "hour_sin": np.sin(2 * np.pi * hour / 24),
        "hour_cos": np.cos(2 * np.pi * hour / 24),
        "dow_sin": np.sin(2 * np.pi * day_of_week / 7),
        "dow_cos": np.cos(2 * np.pi * day_of_week / 7),
        "is_weekend": int(day_of_week >= 5),
        "rush_hour_pagi": int(6 <= hour <= 9),
        "rush_hour_sore": int(16 <= hour <= 20),
        "pm1_pm25_ratio": latest_pm1 / (latest_pm25 + 1e-6),
        "target_horizon": 1,
    })

    for lag in [1, 2, 3, 6, 12, 24]:
        row[f"pm25_lag_{lag}h"] = _lag_value(history, "pm25", lag, latest_pm25)

    for parameter in ["pm1", "temperature", "relativehumidity", "um003"]:
        latest_parameter = _latest_value(history, parameter)
        for lag in [1, 3]:
            row[f"{parameter}_lag_{lag}h"] = _lag_value(
                history,
                parameter,
                lag,
                latest_parameter,
            )

    for window in [3, 6, 12, 24]:
        stats = _rolling_stats(history, "pm25", window)
        row[f"pm25_rolling_mean_{window}h"] = stats["mean"]
        row[f"pm25_rolling_std_{window}h"] = stats["std"]

    stats_24h = _rolling_stats(history, "pm25", 24)
    row["pm25_rolling_min_24h"] = stats_24h["min"]
    row["pm25_rolling_max_24h"] = stats_24h["max"]
    row["pm25_diff_1h"] = _diff_value(history, "pm25", 1)
    row["pm25_diff_3h"] = _diff_value(history, "pm25", 3)
    row["temperature_diff_1h"] = _diff_value(history, "temperature", 1)
    row["humidity_diff_1h"] = _diff_value(history, "relativehumidity", 1)

    features = pd.DataFrame([{feature: row.get(feature, 0.0) for feature in feature_columns}])
    return features.replace([np.inf, -np.inf], np.nan).fillna(0.0)


def _append_prediction(history: pd.DataFrame, target_time, predicted_pm25: float) -> pd.DataFrame:
    """Tambahkan hasil prediksi sebagai histori untuk langkah rekursif berikutnya."""
    next_row = {"datetime": target_time, "pm25": predicted_pm25}

    for parameter in PARAMETERS:
        if parameter == "pm25":
            continue
        next_row[parameter] = _latest_value(history, parameter)

    return pd.concat([history, pd.DataFrame([next_row])], ignore_index=True)


def predict_next_hours(model=None, feature_columns=None) -> Dict[str, Any]:
    """
    Prediksi PM2.5 untuk 24 jam ke depan dengan recursive forecasting.

    Hasil prediksi jam T+1 dipakai untuk membangun lag dan rolling feature
    pada prediksi T+2, dan proses ini berlanjut sampai horizon 24 jam.
    """
    if model is None or feature_columns is None:
        model, feature_columns = load_model()

    if model is None:
        return {
            "status": "no_model",
            "predictions": [],
            "message": "Model belum tersedia.",
        }

    feature_columns = _resolve_feature_columns(model, feature_columns)
    if not feature_columns:
        return {
            "status": "no_features",
            "predictions": [],
            "message": "Daftar fitur model belum tersedia.",
        }

    history = load_local_dataset()
    if history is None or len(history) < 25:
        total_records = len(history) if history is not None else 0
        return {
            "status": "insufficient_data",
            "predictions": [],
            "message": f"Data historis tidak cukup ({total_records} records, minimal 25).",
        }

    needed_columns = ["datetime"] + [parameter for parameter in PARAMETERS if parameter in history.columns]
    history = history[needed_columns].tail(72).copy()
    history["datetime"] = pd.to_datetime(history["datetime"])
    history = history.sort_values("datetime").reset_index(drop=True)

    base_time = history["datetime"].iloc[-1]
    predictions = []

    for hour_offset in range(1, PREDICTION_HORIZON + 1):
        target_time = base_time + timedelta(hours=hour_offset)
        features = _build_feature_row(history, target_time, feature_columns)

        predicted_pm25 = float(model.predict(features)[0])
        predicted_pm25 = max(0.0, predicted_pm25)
        aqi_result = calculate_aqi(predicted_pm25)

        predictions.append({
            "hour_offset": hour_offset,
            "target_time": target_time.isoformat(),
            "pm25_predicted": round(predicted_pm25, 2),
            "aqi_predicted": aqi_result["aqi"],
            "aqi_category": aqi_result["category"],
            "aqi_color": aqi_result["color"],
        })

        history = _append_prediction(history, target_time, predicted_pm25)

    avg_pm25 = float(np.mean([prediction["pm25_predicted"] for prediction in predictions]))
    avg_aqi = calculate_aqi(avg_pm25)
    summary = {
        "avg_pm25": round(avg_pm25, 2),
        "max_pm25": max(prediction["pm25_predicted"] for prediction in predictions),
        "min_pm25": min(prediction["pm25_predicted"] for prediction in predictions),
        "avg_aqi": avg_aqi["aqi"],
        "avg_category": avg_aqi["category"],
        "avg_color": avg_aqi["color"],
    }

    return {
        "status": "ok",
        "predictions": predictions,
        "summary": summary,
        "base_time": base_time.isoformat(),
        "message": f"Prediksi rekursif {len(predictions)} jam berhasil dibuat.",
    }
