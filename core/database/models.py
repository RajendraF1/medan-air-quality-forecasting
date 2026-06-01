"""
Database Models — ORM Definitions
====================================
Tabel-tabel untuk aplikasi production:
  - AirQualityLog: data sensor per jam
  - SystemStatus: status sistem (key-value)
  - Prediction: prediksi 24 jam ke depan
"""

from datetime import datetime
from sqlalchemy import (
    Column, Integer, Float, DateTime, String, Text, Boolean
)
from sqlalchemy.orm import DeclarativeBase

from core.utils.timezone import now_wib


class Base(DeclarativeBase):
    pass


# ==============================================================================
# Tabel: air_quality_log
# ==============================================================================
class AirQualityLog(Base):
    """Log data kualitas udara per jam dari OpenAQ."""
    __tablename__ = "air_quality_log"

    id = Column(Integer, primary_key=True, autoincrement=True)
    recorded_at = Column(DateTime, unique=True, nullable=False, index=True)

    # Parameter sensor mentah
    pm1 = Column(Float, nullable=True)
    pm25 = Column(Float, nullable=True)
    relativehumidity = Column(Float, nullable=True)
    temperature = Column(Float, nullable=True)
    um003 = Column(Float, nullable=True)

    # Hasil prediksi & AQI
    pm25_predicted = Column(Float, nullable=True)
    aqi = Column(Integer, nullable=True)
    aqi_category = Column(String(50), nullable=True)

    # Metadata
    source = Column(String(20), default="openaq")
    created_at = Column(DateTime, default=lambda: now_wib().replace(tzinfo=None))

    def to_dict(self):
        return {
            "id": self.id,
            "recorded_at": self.recorded_at.isoformat() if self.recorded_at else None,
            "pm1": self.pm1,
            "pm25": self.pm25,
            "relativehumidity": self.relativehumidity,
            "temperature": self.temperature,
            "um003": self.um003,
            "pm25_predicted": self.pm25_predicted,
            "aqi": self.aqi,
            "aqi_category": self.aqi_category,
            "source": self.source,
        }


# ==============================================================================
# Tabel: system_status (BARU)
# ==============================================================================
class SystemStatus(Base):
    """
    Key-value store untuk status sistem.

    Keys:
      - last_sync_time
      - last_successful_sync
      - last_training_time
      - last_prediction_time
      - last_dataset_timestamp
      - total_records
      - sync_status  ("active" / "waiting_for_new_data" / "error" / "syncing")
      - active_model_version
    """
    __tablename__ = "system_status"

    id = Column(Integer, primary_key=True, autoincrement=True)
    key = Column(String(100), unique=True, nullable=False, index=True)
    value = Column(Text, nullable=True)
    updated_at = Column(
        DateTime,
        default=lambda: now_wib().replace(tzinfo=None),
        onupdate=lambda: now_wib().replace(tzinfo=None),
    )

    def to_dict(self):
        return {
            "key": self.key,
            "value": self.value,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
        }


# ==============================================================================
# Tabel: predictions (BARU)
# ==============================================================================
class Prediction(Base):
    """
    Prediksi 24 jam ke depan.
    Setiap batch prediksi menghasilkan 24 rows (hour_offset 1..24).
    """
    __tablename__ = "predictions"

    id = Column(Integer, primary_key=True, autoincrement=True)
    predicted_at = Column(DateTime, nullable=False, index=True)
    target_time = Column(DateTime, nullable=False)
    hour_offset = Column(Integer, nullable=False)

    pm25_predicted = Column(Float, nullable=True)
    aqi_predicted = Column(Integer, nullable=True)
    aqi_category = Column(String(50), nullable=True)

    model_version = Column(String(100), nullable=True)
    created_at = Column(DateTime, default=lambda: now_wib().replace(tzinfo=None))

    def to_dict(self):
        return {
            "id": self.id,
            "predicted_at": self.predicted_at.isoformat() if self.predicted_at else None,
            "target_time": self.target_time.isoformat() if self.target_time else None,
            "hour_offset": self.hour_offset,
            "pm25_predicted": self.pm25_predicted,
            "aqi_predicted": self.aqi_predicted,
            "aqi_category": self.aqi_category,
            "model_version": self.model_version,
        }
