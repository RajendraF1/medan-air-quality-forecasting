"""
API Endpoints — Aplikasi Utama
=================================
Semua endpoint REST API untuk frontend.

Endpoints:
  GET /api/current          — AQI & PM2.5 saat ini
  GET /api/history          — Data historis N jam
  GET /api/predictions      — Prediksi 24 jam ke depan
  GET /api/model-info       — Info model (MAE, RMSE, R²)
  GET /api/system-status    — Status sistem
  GET /api/health-recs      — Rekomendasi kesehatan
  GET /api/stats            — Statistik ringkasan
  GET /api/station-info     — Info lokasi stasiun
  POST /api/sync            — Collect/sync data manual dari OpenAQ
"""

from fastapi import APIRouter, Depends, Query, BackgroundTasks
from sqlalchemy.orm import Session

from core.config import (
    LOCATION_ID, STATION_LAT, STATION_LON,
    STATION_NAME, STATION_CITY, PREDICTION_HORIZON,
)
from core.database.connection import get_db
from core.database.crud import (
    get_latest_air_quality,
    get_air_quality_history,
    get_air_quality_stats,
    get_all_system_status,
)
from aplikasi_lab.engine.aqi import calculate_aqi, get_health_recommendations
from aplikasi_lab.engine.fetching import scan_local_dataset, get_latest_local_timestamp, sync_data
from aplikasi_lab.engine.training import get_training_report
from aplikasi_lab.engine.prediction import predict_next_hours
from core.utils.timezone import now_wib, format_wib, format_wib_iso


api_router = APIRouter(prefix="/api", tags=["api"])


# ==============================================================================
# Dashboard
# ==============================================================================
@api_router.get("/current")
async def get_current(db: Session = Depends(get_db)):
    """Data kualitas udara terbaru beserta AQI."""
    # Coba dari database dulu
    latest = get_latest_air_quality(db)

    # Juga cek dataset lokal untuk data paling baru
    from aplikasi_lab.engine.fetching import load_local_dataset
    import pandas as pd

    # Cari dalam 14 hari terakhir
    df = load_local_dataset(days=14)
    if df is None or len(df) == 0:
        df = load_local_dataset(days=30)
    if df is None or len(df) == 0:
        df = load_local_dataset()

    if df is not None and len(df) > 0:
        last_row = df.iloc[-1]
        pm25_val = float(last_row["pm25"]) if pd.notna(last_row.get("pm25")) else None
        aqi_info = calculate_aqi(pm25_val) if pm25_val is not None else None

        return {
            "status": "ok",
            "data": {
                "recorded_at": str(last_row["datetime"]),
                "pm25": pm25_val,
                "pm1": float(last_row["pm1"]) if pd.notna(last_row.get("pm1")) else None,
                "temperature": float(last_row["temperature"]) if pd.notna(last_row.get("temperature")) else None,
                "relativehumidity": float(last_row["relativehumidity"]) if pd.notna(last_row.get("relativehumidity")) else None,
                "um003": float(last_row["um003"]) if pd.notna(last_row.get("um003")) else None,
            },
            "aqi": aqi_info,
        }

    if latest:
        aqi_info = calculate_aqi(latest.pm25) if latest.pm25 is not None else None
        return {
            "status": "ok",
            "data": latest.to_dict(),
            "aqi": aqi_info,
        }

    return {"status": "no_data", "message": "Belum ada data tersedia"}


@api_router.get("/history")
async def get_history(
    hours: int = Query(default=24, ge=1, le=168),
):
    """Data historis PM2.5 dari dataset lokal."""
    from aplikasi_lab.engine.fetching import load_local_dataset
    import pandas as pd

    df = load_local_dataset(days=max(7, hours // 24 + 1))

    if df is None or len(df) == 0:
        return {"status": "ok", "count": 0, "data": []}

    # Filter by hours
    from datetime import timedelta
    cutoff = now_wib().replace(tzinfo=None) - timedelta(hours=hours)
    df = df[df["datetime"] >= cutoff].copy()

    if len(df) == 0:
        # Fallback: ambil N record terakhir
        df_full = load_local_dataset(days=30)
        if df_full is None or len(df_full) == 0:
            df_full = load_local_dataset()
        if df_full is not None and len(df_full) > 0:
            df = df_full.tail(hours).copy()

    records = []
    for _, row in df.iterrows():
        pm25_val = float(row["pm25"]) if pd.notna(row.get("pm25")) else None
        aqi_info = calculate_aqi(pm25_val) if pm25_val is not None else None

        records.append({
            "recorded_at": str(row["datetime"]),
            "pm25": pm25_val,
            "pm1": float(row["pm1"]) if pd.notna(row.get("pm1")) else None,
            "temperature": float(row["temperature"]) if pd.notna(row.get("temperature")) else None,
            "relativehumidity": float(row["relativehumidity"]) if pd.notna(row.get("relativehumidity")) else None,
            "um003": float(row["um003"]) if pd.notna(row.get("um003")) else None,
            "aqi": aqi_info["aqi"] if aqi_info else None,
            "aqi_category": aqi_info["category"] if aqi_info else None,
        })

    return {"status": "ok", "count": len(records), "data": records}


@api_router.get("/stats")
async def get_stats(db: Session = Depends(get_db)):
    """Statistik ringkasan."""
    stats = get_air_quality_stats(db)
    dataset_info = scan_local_dataset()

    return {
        "status": "ok",
        **stats,
        "dataset": dataset_info,
    }


# ==============================================================================
# Sync Data Endpoint (Manual)
# ==============================================================================
@api_router.post("/sync")
async def sync_now(background_tasks: BackgroundTasks):
    """Trigger manual data synchronization from OpenAQ.

    Returns immediately with acknowledgment, sync runs in background.
    """
    # Add sync task to background tasks
    background_tasks.add_task(sync_data)

    return {
        "status": "accepted",
        "message": "Sinkronisasi data dimulai di latar belakang. Cek status melalui /api/system-status."
    }


# ==============================================================================
# Prediksi AI
# ==============================================================================
@api_router.get("/predictions")
async def get_predictions():
    """Prediksi rekursif 24 jam ke depan menggunakan model yang sedang aktif."""
    result = predict_next_hours()
    
    if result["status"] != "ok":
        return result
        
    # Format predictions for frontend
    preds = []
    for p in result["predictions"]:
        preds.append({
            "hour_offset": p["hour_offset"],
            "target_time": p["target_time"],
            "pm25_predicted": p["pm25_predicted"],
            "aqi_predicted": p["aqi_predicted"],
            "aqi_category": p["aqi_category"],
            "aqi_color": p["aqi_color"],
        })
        
    final_result = {
        "status": "ok",
        "predictions": preds,
        "summary": result.get("summary", {}),
        "message": f"{len(preds)} jam prediksi tersedia."
    }

    # Extend model metrics/results
    report = get_training_report()
    if report:
        test_metrics = report.get("final_metrics", {}).get("test", {})
        final_result["model_info"] = {
            "mae": test_metrics.get("mae"),
            "rmse": test_metrics.get("rmse"),
            "r2": test_metrics.get("r2"),
            "mape": test_metrics.get("mape"),
            "trained_at": report.get("trained_at", "-"),
            "model_type": report.get("model_type", "XGBRegressor"),
            "hyperparameters": report.get("hyperparameters", {}),
        }
    return final_result



# ==============================================================================
# Evaluasi Model (ringkas)
# ==============================================================================
@api_router.get("/model-info")
async def get_model_info(db: Session = Depends(get_db)):
    """Info model: MAE, RMSE, R², tanggal training, jumlah data, versi."""
    report = get_training_report()

    if report:
        test_metrics = report.get("final_metrics", {}).get("test", {})
        return {
            "status": "ok",
            "model": {
                "type": report.get("model_type", "XGBRegressor"),
                "version": report.get("model_version", "-"),
                "trained_at": report.get("trained_at", "-"),
                "n_train": report.get("n_train", 0),
                "n_test": report.get("n_test", 0),
                "n_features": report.get("n_features", 0),
                "duration_seconds": report.get("training_duration_seconds", 0),
                "metrics": {
                    "mae": test_metrics.get("mae"),
                    "rmse": test_metrics.get("rmse"),
                    "r2": test_metrics.get("r2"),
                    "mape": test_metrics.get("mape"),
                },
                "top_features": report.get("top_10_features", []),
                "hyperparameters": report.get("hyperparameters", {}),
            },
        }

    return {"status": "no_model", "message": "Training report tidak ditemukan"}


# ==============================================================================
# Mitigasi Kesehatan
# ==============================================================================
@api_router.get("/health-recs")
async def get_health_recs():
    """Rekomendasi kesehatan berdasarkan AQI/PM2.5 saat ini."""
    from aplikasi_lab.engine.fetching import load_local_dataset
    import pandas as pd

    # Cek dalam 14 hari terakhir
    df = load_local_dataset(days=14)
    if df is None or len(df) == 0:
        df = load_local_dataset(days=30)
    if df is None or len(df) == 0:
        df = load_local_dataset()

    if df is None or len(df) == 0:
        return {
            "status": "no_data",
            "message": "Data belum tersedia untuk menghitung rekomendasi.",
        }

    last_row = df.iloc[-1]
    pm25_val = float(last_row["pm25"]) if pd.notna(last_row.get("pm25")) else 0
    aqi_info = calculate_aqi(pm25_val)
    recommendations = get_health_recommendations(aqi_info["aqi"], pm25_val)

    return {
        "status": "ok",
        "current": {
            "pm25": pm25_val,
            "aqi": aqi_info["aqi"],
            "category": aqi_info["category"],
            "color": aqi_info["color"],
        },
        "recommendations": recommendations,
    }


# ==============================================================================
# System Status
# ==============================================================================
@api_router.get("/system-status")
async def get_system_status_api(db: Session = Depends(get_db)):
    """Status sistem: sync, training, prediksi, dll."""
    all_status = get_all_system_status(db)
    dataset_info = scan_local_dataset()
    last_ts = get_latest_local_timestamp()
    current_time = now_wib()

    # Calculate sync status
    sync_status_info = {}
    if last_ts is None:
        sync_status_info = {
            "status": "no_data",
            "status_display": "Belum ada data",
            "last_sync_display": "Belum ada data lokal",
            "last_sync_iso": None,
            "gap_hours": None,
            "gap_display": "N/A",
        }
    else:
        gap = current_time - last_ts
        gap_hours = gap.total_seconds() / 3600

        if gap_hours <= 2:
            status = "synced"
            status_message = "Dataset Sinkron"
        elif gap_hours <= 6:
            status = "slightly_behind"
            status_message = f"Dataset tertinggal {gap_hours:.0f} jam"
        elif gap_hours <= 24:
            status = "behind"
            status_message = f"Dataset tertinggal {gap_hours:.0f} jam"
        else:
            days_behind = gap_hours / 24
            status = "far_behind"
            status_message = f"Dataset tertinggal {days_behind:.1f} hari"

        sync_status_info = {
            "status": status,
            "status_display": status_message,
            "last_sync_display": format_wib(last_ts),
            "last_sync_iso": last_ts.isoformat(),
            "gap_hours": round(gap_hours, 1),
            "gap_display": (
                f"{gap_hours:.0f} jam" if gap_hours < 48
                else f"{gap_hours / 24:.1f} hari"
            ),
        }

    return {
        "status": "ok",
        "collect_mode": "manual",
        "system": all_status,
        "sync": sync_status_info,
        "dataset": {
            "total_records": dataset_info.get("total_records", 0),
            "total_files": dataset_info.get("total_files", 0),
            "last_timestamp": format_wib(last_ts) if last_ts else "-",
            "last_timestamp_iso": last_ts.isoformat() if last_ts else None,
        },
        "current_time": format_wib(now_wib()),
    }


# ==============================================================================
# Station Info (untuk peta)
# ==============================================================================
@api_router.get("/station-info")
async def get_station_info():
    """Info lokasi stasiun untuk peta."""
    return {
        "location_id": LOCATION_ID,
        "name": STATION_NAME,
        "city": STATION_CITY,
        "lat": STATION_LAT,
        "lon": STATION_LON,
        "prediction_horizon": PREDICTION_HORIZON,
    }
