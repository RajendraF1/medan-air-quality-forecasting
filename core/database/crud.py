"""
Database CRUD Operations
===========================
Fungsi helper untuk operasi database.
Seluruh timestamp menggunakan WIB.
"""

from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any

from sqlalchemy.orm import Session
from sqlalchemy import desc, func

from core.database.models import AirQualityLog, SystemStatus, Prediction
from core.utils.timezone import now_wib, format_wib


# ==============================================================================
# System Status
# ==============================================================================
def get_system_status(db: Session, key: str) -> Optional[str]:
    """Ambil nilai status sistem berdasarkan key."""
    record = db.query(SystemStatus).filter(SystemStatus.key == key).first()
    return record.value if record else None


def set_system_status(db: Session, key: str, value: str):
    """Set nilai status sistem. Insert jika belum ada, update jika sudah."""
    record = db.query(SystemStatus).filter(SystemStatus.key == key).first()
    if record:
        record.value = value
        record.updated_at = now_wib().replace(tzinfo=None)
    else:
        record = SystemStatus(
            key=key,
            value=value,
            updated_at=now_wib().replace(tzinfo=None),
        )
        db.add(record)
    db.commit()


def get_all_system_status(db: Session) -> Dict[str, Any]:
    """Ambil semua status sistem sebagai dictionary."""
    records = db.query(SystemStatus).all()
    result = {}
    for r in records:
        result[r.key] = {
            "value": r.value,
            "updated_at": r.updated_at.isoformat() if r.updated_at else None,
            "updated_at_display": format_wib(r.updated_at) if r.updated_at else "-",
        }
    return result


# ==============================================================================
# Air Quality Log
# ==============================================================================
def get_latest_air_quality(db: Session) -> Optional[AirQualityLog]:
    """Ambil data kualitas udara terbaru."""
    return db.query(AirQualityLog).order_by(
        desc(AirQualityLog.recorded_at)
    ).first()


def get_air_quality_history(
    db: Session,
    hours: int = 24,
) -> List[AirQualityLog]:
    """
    Ambil data historis N jam terakhir.
    Jika tidak ada data dalam rentang, ambil N record terakhir.
    """
    wib_now = now_wib().replace(tzinfo=None)
    cutoff = wib_now - timedelta(hours=hours)

    records = db.query(AirQualityLog).filter(
        AirQualityLog.recorded_at >= cutoff
    ).order_by(AirQualityLog.recorded_at).all()

    if not records:
        records = db.query(AirQualityLog).order_by(
            desc(AirQualityLog.recorded_at)
        ).limit(hours).all()
        records = list(reversed(records))

    return records


def get_air_quality_stats(db: Session) -> Dict[str, Any]:
    """Statistik ringkasan kualitas udara 24 jam terakhir."""
    wib_now = now_wib().replace(tzinfo=None)
    cutoff_24h = wib_now - timedelta(hours=24)

    stats_24h = db.query(
        func.avg(AirQualityLog.pm25).label("avg_pm25"),
        func.min(AirQualityLog.pm25).label("min_pm25"),
        func.max(AirQualityLog.pm25).label("max_pm25"),
        func.avg(AirQualityLog.temperature).label("avg_temp"),
        func.avg(AirQualityLog.relativehumidity).label("avg_humidity"),
        func.count(AirQualityLog.id).label("total_records"),
    ).filter(AirQualityLog.recorded_at >= cutoff_24h).first()

    total = db.query(func.count(AirQualityLog.id)).scalar()

    return {
        "last_24h": {
            "avg_pm25": round(float(stats_24h.avg_pm25 or 0), 2),
            "min_pm25": round(float(stats_24h.min_pm25 or 0), 2),
            "max_pm25": round(float(stats_24h.max_pm25 or 0), 2),
            "avg_temperature": round(float(stats_24h.avg_temp or 0), 1),
            "avg_humidity": round(float(stats_24h.avg_humidity or 0), 1),
            "records": stats_24h.total_records or 0,
        },
        "total_records": total,
    }


# ==============================================================================
# Predictions
# ==============================================================================
def save_predictions(
    db: Session,
    predictions: List[Dict[str, Any]],
    model_version: str,
):
    """
    Simpan batch prediksi 24 jam ke depan.
    Hapus prediksi lama terlebih dahulu.
    """
    # Hapus prediksi sebelumnya
    db.query(Prediction).delete()

    predicted_at = now_wib().replace(tzinfo=None)

    for pred in predictions:
        record = Prediction(
            predicted_at=predicted_at,
            target_time=pred["target_time"],
            hour_offset=pred["hour_offset"],
            pm25_predicted=pred["pm25_predicted"],
            aqi_predicted=pred["aqi_predicted"],
            aqi_category=pred["aqi_category"],
            model_version=model_version,
        )
        db.add(record)

    db.commit()


def get_latest_predictions(db: Session) -> List[Prediction]:
    """Ambil prediksi 24 jam terbaru, urut berdasarkan hour_offset."""
    return db.query(Prediction).order_by(Prediction.hour_offset).all()
