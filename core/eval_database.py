"""
Database Models untuk Dashboard Evaluasi Model
================================================
Tabel-tabel terpisah (prefixed eval_) untuk menyimpan seluruh hasil
evaluasi, eksperimen, dan analisis model.

Menggunakan engine yang sama dari database.py (SQLite/MySQL).

PENTING:
  - Seluruh timestamp menggunakan WIB (UTC+7)
  - Location ID tetap: 5586536
"""

import json
from datetime import datetime, timedelta, timezone
from sqlalchemy import (
    Column, Integer, Float, DateTime, String, Text, Boolean, JSON,
    create_engine
)
from sqlalchemy.orm import sessionmaker
from core.database.connection import engine, Base

# Timezone WIB (UTC+7)
TIMEZONE_WIB = timezone(timedelta(hours=7))


def _now_wib():
    """Dapatkan waktu saat ini dalam WIB."""
    return datetime.now(TIMEZONE_WIB).replace(tzinfo=None)


# ==============================================================================
# Tabel: eval_datasets
# ==============================================================================
class EvalDataset(Base):
    """Metadata dataset yang digunakan untuk training/evaluasi."""
    __tablename__ = "eval_datasets"

    id = Column(Integer, primary_key=True, autoincrement=True)
    created_at = Column(DateTime, default=_now_wib)
    source = Column(String(50), default="openaq")  # openaq, csv, manual
    location_id = Column(Integer, nullable=True)

    # Data range
    date_start = Column(DateTime, nullable=True)
    date_end = Column(DateTime, nullable=True)
    total_records = Column(Integer, default=0)
    total_features = Column(Integer, default=0)

    # Fetch info
    fetch_duration_seconds = Column(Float, nullable=True)
    total_requests = Column(Integer, default=0)
    data_interval = Column(String(20), default="1h")

    # Sync info
    sync_type = Column(String(20), default="full")  # full, incremental, none
    new_records = Column(Integer, default=0)

    # Status
    status = Column(String(20), default="pending")  # pending, fetching, done, synced, no_new_data, error
    error_message = Column(Text, nullable=True)

    def to_dict(self):
        return {
            "id": self.id,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "source": self.source,
            "location_id": self.location_id,
            "date_start": self.date_start.isoformat() if self.date_start else None,
            "date_end": self.date_end.isoformat() if self.date_end else None,
            "total_records": self.total_records,
            "total_features": self.total_features,
            "fetch_duration_seconds": self.fetch_duration_seconds,
            "total_requests": self.total_requests,
            "data_interval": self.data_interval,
            "sync_type": self.sync_type,
            "new_records": self.new_records,
            "status": self.status,
            "error_message": self.error_message,
        }


# ==============================================================================
# Tabel: eval_data_quality
# ==============================================================================
class EvalDataQuality(Base):
    """Hasil analisis kualitas data."""
    __tablename__ = "eval_data_quality"

    id = Column(Integer, primary_key=True, autoincrement=True)
    dataset_id = Column(Integer, nullable=True)
    created_at = Column(DateTime, default=_now_wib)

    # Missing values (JSON: {"column_name": {"count": N, "percentage": P}})
    missing_values = Column(Text, nullable=True)

    # Duplicate info
    duplicate_count = Column(Integer, default=0)

    # Outlier info (JSON)
    outlier_info = Column(Text, nullable=True)

    # Timestamp gaps (JSON: [{"start": ..., "end": ..., "gap_hours": N}])
    timestamp_gaps = Column(Text, nullable=True)
    total_gaps = Column(Integer, default=0)

    # Distribution stats (JSON)
    distribution_stats = Column(Text, nullable=True)

    def to_dict(self):
        return {
            "id": self.id,
            "dataset_id": self.dataset_id,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "missing_values": json.loads(self.missing_values) if self.missing_values else {},
            "duplicate_count": self.duplicate_count,
            "outlier_info": json.loads(self.outlier_info) if self.outlier_info else {},
            "timestamp_gaps": json.loads(self.timestamp_gaps) if self.timestamp_gaps else [],
            "total_gaps": self.total_gaps,
            "distribution_stats": json.loads(self.distribution_stats) if self.distribution_stats else {},
        }


# ==============================================================================
# Tabel: eval_experiments
# ==============================================================================
class EvalExperiment(Base):
    """Riwayat eksperimen training model."""
    __tablename__ = "eval_experiments"

    id = Column(Integer, primary_key=True, autoincrement=True)
    created_at = Column(DateTime, default=_now_wib)
    dataset_id = Column(Integer, nullable=True)

    # Model info
    model_type = Column(String(50), default="XGBRegressor")
    hyperparameters = Column(Text, nullable=True)  # JSON

    # Data split
    n_train = Column(Integer, default=0)
    n_test = Column(Integer, default=0)
    n_features = Column(Integer, default=0)
    feature_columns = Column(Text, nullable=True)  # JSON list

    # Data range (untuk experiment history)
    data_range_start = Column(DateTime, nullable=True)
    data_range_end = Column(DateTime, nullable=True)
    total_data_records = Column(Integer, default=0)

    # Training info
    training_duration_seconds = Column(Float, nullable=True)
    best_iteration = Column(Integer, nullable=True)

    # Horizon prediction
    horizon_prediction = Column(String(50), nullable=True)  # e.g. "1h, 3h, 6h, 12h, 24h, 48h"

    # Status
    status = Column(String(20), default="pending")  # pending, preprocessing, training, evaluating, done, error
    progress_message = Column(String(200), nullable=True)
    error_message = Column(Text, nullable=True)

    # CV scores (JSON)
    cv_scores = Column(Text, nullable=True)

    def to_dict(self):
        return {
            "id": self.id,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "dataset_id": self.dataset_id,
            "model_type": self.model_type,
            "hyperparameters": json.loads(self.hyperparameters) if self.hyperparameters else {},
            "n_train": self.n_train,
            "n_test": self.n_test,
            "n_features": self.n_features,
            "feature_columns": json.loads(self.feature_columns) if self.feature_columns else [],
            "data_range_start": self.data_range_start.isoformat() if self.data_range_start else None,
            "data_range_end": self.data_range_end.isoformat() if self.data_range_end else None,
            "total_data_records": self.total_data_records,
            "training_duration_seconds": self.training_duration_seconds,
            "best_iteration": self.best_iteration,
            "horizon_prediction": self.horizon_prediction,
            "status": self.status,
            "progress_message": self.progress_message,
            "error_message": self.error_message,
            "cv_scores": json.loads(self.cv_scores) if self.cv_scores else {},
        }


# ==============================================================================
# Tabel: eval_metrics
# ==============================================================================
class EvalMetric(Base):
    """Hasil metrik evaluasi per eksperimen."""
    __tablename__ = "eval_metrics"

    id = Column(Integer, primary_key=True, autoincrement=True)
    experiment_id = Column(Integer, nullable=False)
    created_at = Column(DateTime, default=_now_wib)

    # Metrics
    dataset_split = Column(String(10), default="test")  # train / test
    mae = Column(Float, nullable=True)
    rmse = Column(Float, nullable=True)
    r2 = Column(Float, nullable=True)
    mape = Column(Float, nullable=True)
    max_error = Column(Float, nullable=True)
    median_ae = Column(Float, nullable=True)

    def to_dict(self):
        return {
            "id": self.id,
            "experiment_id": self.experiment_id,
            "dataset_split": self.dataset_split,
            "mae": self.mae,
            "rmse": self.rmse,
            "r2": self.r2,
            "mape": self.mape,
            "max_error": self.max_error,
            "median_ae": self.median_ae,
        }


# ==============================================================================
# Tabel: eval_feature_importance
# ==============================================================================
class EvalFeatureImportance(Base):
    """Ranking fitur per eksperimen."""
    __tablename__ = "eval_feature_importance"

    id = Column(Integer, primary_key=True, autoincrement=True)
    experiment_id = Column(Integer, nullable=False)
    feature_name = Column(String(100), nullable=False)
    importance = Column(Float, default=0.0)
    rank = Column(Integer, default=0)

    def to_dict(self):
        return {
            "id": self.id,
            "experiment_id": self.experiment_id,
            "feature_name": self.feature_name,
            "importance": self.importance,
            "rank": self.rank,
        }


# ==============================================================================
# Tabel: eval_horizon_results
# ==============================================================================
class EvalHorizonResult(Base):
    """Hasil evaluasi per horizon prediksi."""
    __tablename__ = "eval_horizon_results"

    id = Column(Integer, primary_key=True, autoincrement=True)
    experiment_id = Column(Integer, nullable=False)
    horizon_hours = Column(Integer, nullable=False)  # 1, 3, 6, 12, 24, 48

    mae = Column(Float, nullable=True)
    rmse = Column(Float, nullable=True)
    r2 = Column(Float, nullable=True)
    mape = Column(Float, nullable=True)

    # Marking
    is_best = Column(Boolean, default=False)
    is_most_stable = Column(Boolean, default=False)
    is_largest_drop = Column(Boolean, default=False)

    def to_dict(self):
        return {
            "id": self.id,
            "experiment_id": self.experiment_id,
            "horizon_hours": self.horizon_hours,
            "mae": self.mae,
            "rmse": self.rmse,
            "r2": self.r2,
            "mape": self.mape,
            "is_best": self.is_best,
            "is_most_stable": self.is_most_stable,
            "is_largest_drop": self.is_largest_drop,
        }


# ==============================================================================
# Tabel: eval_preprocessing_logs
# ==============================================================================
class EvalPreprocessingLog(Base):
    """Log preprocessing yang dilakukan."""
    __tablename__ = "eval_preprocessing_logs"

    id = Column(Integer, primary_key=True, autoincrement=True)
    experiment_id = Column(Integer, nullable=True)
    created_at = Column(DateTime, default=_now_wib)

    # Preprocessing info
    raw_records = Column(Integer, default=0)
    clean_records = Column(Integer, default=0)
    dropped_records = Column(Integer, default=0)

    # Features added (JSON list)
    features_added = Column(Text, nullable=True)
    total_features = Column(Integer, default=0)

    # Preprocessing steps (JSON list of dicts)
    steps_log = Column(Text, nullable=True)

    def to_dict(self):
        return {
            "id": self.id,
            "experiment_id": self.experiment_id,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "raw_records": self.raw_records,
            "clean_records": self.clean_records,
            "dropped_records": self.dropped_records,
            "features_added": json.loads(self.features_added) if self.features_added else [],
            "total_features": self.total_features,
            "steps_log": json.loads(self.steps_log) if self.steps_log else [],
        }


# ==============================================================================
# Tabel: eval_predictions (untuk menyimpan actual vs predicted)
# ==============================================================================
class EvalPrediction(Base):
    """Menyimpan hasil prediksi untuk visualisasi."""
    __tablename__ = "eval_predictions"

    id = Column(Integer, primary_key=True, autoincrement=True)
    experiment_id = Column(Integer, nullable=False)
    datetime_val = Column(DateTime, nullable=True)
    actual = Column(Float, nullable=True)
    predicted = Column(Float, nullable=True)
    residual = Column(Float, nullable=True)
    dataset_split = Column(String(10), default="test")  # train / test

    def to_dict(self):
        return {
            "id": self.id,
            "experiment_id": self.experiment_id,
            "datetime": self.datetime_val.isoformat() if self.datetime_val else None,
            "actual": self.actual,
            "predicted": self.predicted,
            "residual": self.residual,
            "dataset_split": self.dataset_split,
        }


# ==============================================================================
# Init Eval Tables
# ==============================================================================
def init_eval_db():
    """Buat semua tabel evaluasi di database."""
    Base.metadata.create_all(bind=engine)
    
    # Auto-migrate new columns for EvalDataset
    from sqlalchemy import text
    with engine.begin() as conn:
        try:
            conn.execute(text("ALTER TABLE eval_datasets ADD COLUMN sync_type VARCHAR(50) DEFAULT 'incremental'"))
        except Exception:
            pass
        try:
            conn.execute(text("ALTER TABLE eval_datasets ADD COLUMN new_records INTEGER DEFAULT 0"))
        except Exception:
            pass

        # Auto-migrate new columns for EvalExperiment
        try:
            conn.execute(text("ALTER TABLE eval_experiments ADD COLUMN data_range_start DATETIME"))
        except Exception:
            pass
        try:
            conn.execute(text("ALTER TABLE eval_experiments ADD COLUMN data_range_end DATETIME"))
        except Exception:
            pass
        try:
            conn.execute(text("ALTER TABLE eval_experiments ADD COLUMN total_data_records INTEGER DEFAULT 0"))
        except Exception:
            pass
        try:
            conn.execute(text("ALTER TABLE eval_experiments ADD COLUMN horizon_prediction VARCHAR(50)"))
        except Exception:
            pass

    print("[OK] Evaluation database tables created")
