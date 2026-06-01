"""
AirSense — Konfigurasi Terpusat
=================================
Semua konfigurasi aplikasi production di satu tempat.

PENTING:
  - LOCATION_ID tetap 5586536 (single location)
  - Timezone: Asia/Jakarta (WIB, UTC+7)
  - Horizon prediksi: 24 jam (fixed)
  - Collect data berjalan manual lewat tombol/API
"""

import os
from pathlib import Path
from datetime import timezone, timedelta

from dotenv import load_dotenv

# ==============================================================================
# Path
# ==============================================================================
BASE_DIR = Path(__file__).parent
PROJECT_DIR = BASE_DIR.parent

# Default .env ada di root project. core/.env tetap didukung sebagai override lokal.
load_dotenv(PROJECT_DIR / ".env")
load_dotenv(BASE_DIR / ".env", override=True)

MODEL_DIR = BASE_DIR / "models"
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "kualitas_udara"
TEMPLATES_DIR = BASE_DIR.parent / "aplikasi_utama" / "templates"
STATIC_DIR = BASE_DIR.parent / "aplikasi_utama" / "static"

# Pastikan direktori ada
MODEL_DIR.mkdir(exist_ok=True)
DATA_DIR.mkdir(exist_ok=True)
RAW_DATA_DIR.mkdir(exist_ok=True)
TEMPLATES_DIR.mkdir(exist_ok=True)
STATIC_DIR.mkdir(exist_ok=True)

# ==============================================================================
# OpenAQ
# ==============================================================================
OPENAQ_BASE_URL = "https://api.openaq.org"
OPENAQ_API_KEY = os.getenv("OPENAQ_API_KEY", "")
LOCATION_ID = 5586536  # Stasiun USU — single location, jangan multi-location

# ==============================================================================
# Timezone — WIB (UTC+7) untuk SELURUH sistem
# ==============================================================================
TIMEZONE_WIB = timezone(timedelta(hours=7))

# ==============================================================================
# Sensor Parameters
# ==============================================================================
PARAMETERS = ["pm1", "pm25", "relativehumidity", "temperature", "um003"]
TARGET = "pm25"

# ==============================================================================
# Prediction
# ==============================================================================
PREDICTION_HORIZON = 24  # Prediksi 24 jam ke depan (fixed)

# ==============================================================================
# Collect Data
# ==============================================================================
SYNC_INTERVAL_HOURS = 1  # Referensi gap minimal saat collect data manual
COLLECT_CHECK_SECONDS = 300  # Legacy UI interval bila dibutuhkan frontend/admin

# ==============================================================================
# Dataset
# ==============================================================================
DATASET_DAYS = 180  # 6 bulan data untuk training

# ==============================================================================
# Model — Hyperparameters Final (dari Dashboard Evaluasi)
# ==============================================================================
XGBOOST_PARAMS = {
    "objective": "reg:squarederror",
    "n_estimators": 1000,
    "max_depth": 6,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 3,
    "gamma": 0.1,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "random_state": 42,
    "n_jobs": -1,
}

# Training
TEST_RATIO = 0.2
N_CV_SPLITS = 5
EARLY_STOPPING_ROUNDS = 50

# Features to exclude from model to prevent data leakage during recursive prediction
EXCLUDED_CURRENT_FEATURES = ["pm1", "temperature", "relativehumidity", "um003", "pm1_pm25_ratio"]

# ==============================================================================
# Database
# ==============================================================================
DB_USE_MYSQL = os.getenv("DB_USE_MYSQL", "false").lower() == "true"
DB_USER = os.getenv("DB_USER", "root")
DB_PASSWORD = os.getenv("DB_PASSWORD", "")
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = os.getenv("DB_PORT", "3306")
DB_NAME = os.getenv("DB_NAME", "air_quality_db")

# ==============================================================================
# Model Paths
# ==============================================================================
MODEL_PATH = MODEL_DIR / "pm25_xgboost_model.joblib"
MODEL_NEW_PATH = MODEL_DIR / "pm25_xgboost_model_new.joblib"
TRAINING_REPORT_PATH = MODEL_DIR / "training_report.json"
FEATURE_COLS_PATH = DATA_DIR / "feature_columns.json"
FEATURE_IMPORTANCE_PATH = MODEL_DIR / "feature_importance.csv"

# ==============================================================================
# Lokasi Stasiun (untuk peta)
# ==============================================================================
STATION_LAT = 3.5636  # Latitude Stasiun USU Medan
STATION_LON = 98.6559  # Longitude Stasiun USU Medan
STATION_NAME = "Stasiun Pemantau Kualitas Udara USU"
STATION_CITY = "Medan, Sumatera Utara"

# ==============================================================================
# Admin Laboratorium Credentials
# ==============================================================================
ADMIN_USERNAME = os.getenv("ADMIN_USERNAME", "admin")
ADMIN_PASSWORD = os.getenv("ADMIN_PASSWORD", "admin123")
