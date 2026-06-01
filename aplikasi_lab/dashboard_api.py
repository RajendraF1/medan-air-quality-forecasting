"""
Dashboard Evaluasi Model — FastAPI Router
============================================
Router terpisah yang di-mount di /eval/ pada app utama.
Menyediakan API endpoints untuk semua fitur evaluasi model.

Tidak mengubah app utama sama sekali.

PENTING:
  - Seluruh timestamp menggunakan WIB (UTC+7)
  - Location ID tetap: 5586536
  - Dataset disimpan di data/kualitas_udara/ sebagai CSV bulanan
"""

import json
import asyncio
import time
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta, timezone
from pydantic import BaseModel
from fastapi import APIRouter, HTTPException, Query, Depends, status
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from core.config import ADMIN_USERNAME, ADMIN_PASSWORD, RAW_DATA_DIR
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.requests import Request
import secrets

class Hyperparameters(BaseModel):
    n_estimators: int = 1000
    max_depth: int = 6
    learning_rate: float = 0.05
    test_split_pct: float = 20.0

from core.eval_database import (
    EvalDataset, EvalDataQuality, EvalExperiment, EvalMetric,
    EvalFeatureImportance, EvalHorizonResult, EvalPreprocessingLog,
    EvalPrediction, init_eval_db,
)
from core.database.connection import SessionLocal
import aplikasi_lab.engine as engine

# ==============================================================================
# Router Config & Auth
# ==============================================================================
# Auth Setup
def get_current_admin(request: Request):
    auth_cookie = request.cookies.get("admin_auth")
    if not auth_cookie or auth_cookie != "valid_session":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Not authenticated. Silakan login dari dashboard utama.",
        )
    return "admin"

eval_router = APIRouter(
    prefix="/eval",
    tags=["evaluation"],
    dependencies=[Depends(get_current_admin)]
)

auth_router = APIRouter(tags=["auth"])

from fastapi import Form, Response
@auth_router.post("/api/admin-login")
def admin_login(response: Response, username: str = Form(...), password: str = Form(...)):
    correct_username = secrets.compare_digest(username, ADMIN_USERNAME)
    correct_password = secrets.compare_digest(password, ADMIN_PASSWORD)
    if not (correct_username and correct_password):
        raise HTTPException(status_code=401, detail="Username atau password salah")
    
    # Set cookie berlaku 1 hari
    response.set_cookie(key="admin_auth", value="valid_session", httponly=True, max_age=86400)
    return {"status": "ok"}

@auth_router.post("/api/admin-logout")
def admin_logout(response: Response):
    response.delete_cookie("admin_auth")
    return {"status": "ok"}

BASE_DIR = Path(__file__).parent
EVAL_TEMPLATES_DIR = BASE_DIR / "templates"
EVAL_TEMPLATES_DIR.mkdir(exist_ok=True)

# Global state untuk tracking progress async operations
_progress_state = {
    "fetch": {"message": "", "progress": 0, "running": False},
    "train": {"message": "", "progress": 0, "running": False, "step": "idle"},
}


# ==============================================================================
# Init eval DB on import
# ==============================================================================
init_eval_db()


# ==============================================================================
# Serve Dashboard HTML
# ==============================================================================
@eval_router.get("/", response_class=HTMLResponse)
async def serve_eval_dashboard():
    """Serve halaman dashboard evaluasi."""
    html_path = EVAL_TEMPLATES_DIR / "eval_dashboard.html"
    if not html_path.exists():
        raise HTTPException(status_code=404, detail="Dashboard evaluasi belum tersedia")
    return HTMLResponse(content=html_path.read_text(encoding="utf-8"))


# ==============================================================================
# DATA SYNCING (menggantikan fetch-data)
# ==============================================================================
@eval_router.post("/api/sync-data")
async def start_sync_data():
    """
    Sinkronisasi data dari OpenAQ API secara inkremental.
    Cek dataset lokal dulu, fetch hanya data yang belum ada.
    """
    if _progress_state["fetch"]["running"]:
        raise HTTPException(status_code=409, detail="Sinkronisasi sedang berjalan")

    _progress_state["fetch"]["running"] = True
    _progress_state["fetch"]["progress"] = 0
    _progress_state["fetch"]["message"] = "Memulai sinkronisasi..."

    async def run_sync():
        db = SessionLocal()
        try:
            _progress_state["fetch"]["message"] = "Mengambil data dari OpenAQ..."
            _progress_state["fetch"]["progress"] = 30

            # Fetch data from OpenAQ
            result = await engine.sync_data()

            _progress_state["fetch"]["message"] = result.get("message", "Selesai")
            _progress_state["fetch"]["progress"] = 100

            # Simpan ke database - gunakan data yang tersedia dari result
            dataset = EvalDataset(
                source="openaq",
                location_id=engine.LOCATION_ID,
                date_start=None,  # sync_data tidak memberikan tanggal spesifik
                date_end=None,
                total_records=result.get("new_records", 0),  # gunakan new_records sebagai total_records untuk kesederhanaan
                fetch_duration_seconds=0,  # tidak tersedia
                total_requests=0,  # tidak tersedia
                sync_type=result.get("status", "unknown"),
                new_records=result.get("new_records", 0),
                status=result.get("status", "unknown"),
            )
            db.add(dataset)
            db.commit()

        except Exception as e:
            _progress_state["fetch"]["message"] = f"Error: {str(e)}"
        finally:
            _progress_state["fetch"]["running"] = False
            db.close()

    asyncio.create_task(run_sync())
    return {"status": "started", "message": "Sinkronisasi data dimulai"}


# Backward compat: old fetch-data endpoint redirects to sync
@eval_router.post("/api/fetch-data")
async def start_fetch_data(months: int = Query(default=6, ge=1, le=12)):
    """Backward-compatible: redirect ke sync-data."""
    return await start_sync_data()


@eval_router.get("/api/fetch-progress")
async def get_fetch_progress():
    """Stream progress fetch/sync data via SSE."""
    async def event_stream():
        while True:
            state = _progress_state["fetch"]
            data = json.dumps({
                "message": state["message"],
                "progress": state["progress"],
                "running": state["running"],
            })
            yield f"data: {data}\n\n"

            if not state["running"] and state["progress"] >= 100:
                break
            if not state["running"] and "Error" in state.get("message", ""):
                break
            await asyncio.sleep(1)

    return StreamingResponse(event_stream(), media_type="text/event-stream")


# ==============================================================================
# DATASET MONITORING
# ==============================================================================
@eval_router.get("/api/dataset-monitoring")
async def get_dataset_monitoring():
    """
    Dataset Monitoring — info lengkap dataset lokal.

    Returns:
        Dataset Summary, File Status, Synchronization Status
    """
    # Scan dataset lokal
    dataset_info = engine.scan_local_dataset()
    sync_status = engine.get_sync_status()

    # Calculate total_days from date_range if available
    total_days = 0
    date_range = dataset_info.get("date_range", {})
    if date_range and "start" in date_range and "end" in date_range:
        try:
            from datetime import datetime
            start = datetime.fromisoformat(date_range["start"].replace("Z", "+00:00"))
            end = datetime.fromisoformat(date_range["end"].replace("Z", "+00:00"))
            total_days = max(0, (end - start).days)
        except (ValueError, KeyError, AttributeError):
            total_days = 0

    # Get detailed file status for each CSV
    csv_files = sorted(RAW_DATA_DIR.glob("*.csv"))
    file_status = []
    import pandas as pd
    for f in csv_files:
        stat = f.stat()
        # count records
        try:
            df_file = pd.read_csv(f, usecols=[0])
            records = len(df_file)
        except:
            records = 0
            
        # format size
        size_bytes = stat.st_size
        size_display = f"{size_bytes / 1024:.1f} KB"
        if size_bytes > 1024 * 1024:
            size_display = f"{size_bytes / (1024*1024):.1f} MB"
            
        # format date
        from datetime import datetime
        last_modified = datetime.fromtimestamp(stat.st_mtime).strftime("%d %b %Y, %H:%M")
            
        file_status.append({
            "filename": f.name,
            "records": records,
            "size_display": size_display,
            "last_modified_display": last_modified,
        })

    return {
        "location_id": engine.LOCATION_ID,
        "dataset_summary": {
            "location_id": engine.LOCATION_ID,
            "total_records": dataset_info["total_records"],
            "total_days": total_days,
            "total_files": dataset_info["total_files"],
            "date_range": date_range,
        },
        "file_status": file_status,
        "sync_status": sync_status,
    }


# ==============================================================================
# DATASET VALIDATION
# ==============================================================================
@eval_router.get("/api/dataset-validation")
async def get_dataset_validation():
    """
    Data Integrity Validation — cek dataset sebelum training.

    Cek: Duplicate Timestamp, Missing Timestamp, Gap Data,
         Missing Values, Empty Records
    """
    validation = engine.validate_dataset_integrity()
    return validation


# ==============================================================================
# DATASET INFO
# ==============================================================================
@eval_router.get("/api/dataset-info")
async def get_dataset_info():
    """Info tentang dataset yang tersedia (lokal + DB history)."""
    # Scan dataset lokal dari CSV
    local_info = engine.scan_local_dataset()

    # Load data lokal untuk statistik
    local_df = engine.load_local_dataset()

    # Compute date_range and total_days from the loaded dataframe
    if local_df is not None and len(local_df) > 0:
        # Ensure datetime is datetime type
        if not pd.api.types.is_datetime64_any_dtype(local_df["datetime"]):
            local_df["datetime"] = pd.to_datetime(local_df["datetime"])
        date_range = {
            "start": local_df["datetime"].min().isoformat(),
            "end": local_df["datetime"].max().isoformat()
        }
        total_days = (local_df["datetime"].max() - local_df["datetime"].min()).days + 1
    else:
        date_range = {}
        total_days = 0

    info = {
        "local_dataset": {
            "available": local_info["total_records"] > 0,
            "total_records": local_info["total_records"],
            "total_files": local_info["total_files"],
            "total_days": total_days,
            "date_range": date_range,
            "location_id": engine.LOCATION_ID,
        },
        "existing_processed": None,
        "datasets": [],
    }

    # Backward compat: check processed_data.csv juga
    existing = engine.load_existing_data()
    if existing is not None:
        info["existing_processed"] = {
            "records": len(existing),
            "features": len(existing.columns),
            "date_start": existing["datetime"].min().isoformat(),
            "date_end": existing["datetime"].max().isoformat(),
            "columns": existing.columns.tolist(),
            "interval": "1h",
        }

    # Riwayat dari database
    db = SessionLocal()
    try:
        datasets = db.query(EvalDataset).order_by(EvalDataset.created_at.desc()).limit(10).all()
        info["datasets"] = [d.to_dict() for d in datasets]
    finally:
        db.close()

    return info


# ==============================================================================
# DATA QUALITY
# ==============================================================================
@eval_router.get("/api/data-quality")
async def get_data_quality(source: str = Query(default="local")):
    """Analisis kualitas data."""
    if source == "local":
        df = engine.load_local_dataset()
    elif source == "existing":
        df = engine.load_existing_data()
    else:
        path = engine.DATA_DIR / "eval_raw_data.csv"
        if path.exists():
            df = pd.read_csv(path, parse_dates=["datetime"])
        else:
            df = None

    if df is None:
        raise HTTPException(status_code=404, detail="Data tidak ditemukan")

    quality = engine.analyze_data_quality(df)

    # Simpan ke database
    db = SessionLocal()
    try:
        record = EvalDataQuality(
            missing_values=json.dumps(quality["missing_values"]),
            duplicate_count=quality["duplicate_count"],
            outlier_info=json.dumps(quality["outlier_info"]),
            timestamp_gaps=json.dumps(quality["timestamp_gaps"]),
            total_gaps=quality["total_gaps"],
            distribution_stats=json.dumps(quality["distribution_stats"]),
        )
        db.add(record)
        db.commit()
    finally:
        db.close()

    # Tambah summary info
    quality["summary"] = {
        "total_records": len(df),
        "total_columns": len(df.columns),
        "date_range": {
            "start": df["datetime"].min().isoformat() if "datetime" in df.columns else None,
            "end": df["datetime"].max().isoformat() if "datetime" in df.columns else None,
        },
        "interval": "1h",
    }

    return quality


# ==============================================================================
# DISTRIBUTION
# ==============================================================================
@eval_router.get("/api/distribution")
async def get_distribution(source: str = Query(default="local")):
    """Distribusi data PM2.5."""
    if source == "local":
        df = engine.load_local_dataset()
    elif source == "existing":
        df = engine.load_existing_data()
    else:
        path = engine.DATA_DIR / "eval_raw_data.csv"
        if path.exists():
            df = pd.read_csv(path, parse_dates=["datetime"])
        else:
            df = None

    if df is None:
        raise HTTPException(status_code=404, detail="Data tidak ditemukan")

    quality = engine.analyze_data_quality(df)
    return quality.get("distribution_stats", {})


# ==============================================================================
# PREPROCESSING
# ==============================================================================
@eval_router.post("/api/preprocess")
async def run_preprocess(source: str = Query(default="local")):
    """Jalankan preprocessing pada data."""
    if source == "local":
        df = engine.load_local_dataset()
        if df is None:
            raise HTTPException(status_code=404, detail="Dataset lokal tidak ditemukan di data/kualitas_udara/")
        # Ambil hanya kolom raw
        needed_cols = ["datetime"] + engine.PARAMETERS
        available = [c for c in needed_cols if c in df.columns]
        df = df[available].copy()
    elif source == "existing":
        df = engine.load_existing_data()
        if df is None:
            raise HTTPException(status_code=404, detail="processed_data.csv tidak ditemukan")
        needed_cols = ["datetime"] + engine.PARAMETERS
        available = [c for c in needed_cols if c in df.columns]
        df = df[available].copy()
    else:
        path = engine.DATA_DIR / "eval_raw_data.csv"
        if not path.exists():
            raise HTTPException(status_code=404, detail="eval_raw_data.csv tidak ditemukan")
        df = pd.read_csv(path, parse_dates=["datetime"])

    result = engine.preprocess_for_training(df)
    
    prep_log = {
        "raw_records": result["raw_records"],
        "clean_records": result["clean_records"],
        "dropped_records": result["raw_records"] - result["clean_records"],
        "features_added": result["feature_columns"],
        "total_features": len(result["feature_columns"]),
        "steps_log": result["log"],
    }

    # Simpan ke database
    db = SessionLocal()
    try:
        record = EvalPreprocessingLog(
            raw_records=prep_log["raw_records"],
            clean_records=prep_log["clean_records"],
            dropped_records=prep_log["dropped_records"],
            features_added=json.dumps(prep_log["features_added"]),
            total_features=prep_log["total_features"],
            steps_log=json.dumps(prep_log["steps_log"]),
        )
        db.add(record)
        db.commit()
    finally:
        db.close()

    # Simpan preprocessed data
    result["dataframe"].to_csv(engine.DATA_DIR / "processed_data.csv", index=False)

    return {
        "status": "done",
        "preprocessing_log": prep_log,
        "feature_columns": result["feature_columns"],
        "sample_data": result["dataframe"].head(5).to_dict(orient="records"),
    }


# ==============================================================================
# TRAINING
# ==============================================================================
@eval_router.post("/api/train")
async def start_training(
    params: Hyperparameters = None,
    source: str = Query(default="local")
):
    """Mulai training model. Validasi dataset terlebih dahulu."""
    if params is None:
        params = Hyperparameters()
    
    if _progress_state["train"]["running"]:
        raise HTTPException(status_code=409, detail="Training sedang berjalan")

    _progress_state["train"]["running"] = True
    _progress_state["train"]["progress"] = 0
    _progress_state["train"]["step"] = "validation"
    _progress_state["train"]["message"] = "Memvalidasi dataset..."

    def run_training():
        db = SessionLocal()
        try:
            # Step 0: Validasi dataset
            _progress_state["train"]["step"] = "validation"
            _progress_state["train"]["message"] = "Memvalidasi integritas dataset..."
            _progress_state["train"]["progress"] = 2

            validation = engine.validate_dataset_integrity()
            if not validation["valid"]:
                _progress_state["train"]["message"] = f"Dataset tidak valid: {validation['message']}"
                _progress_state["train"]["step"] = "error"
                _progress_state["train"]["running"] = False
                return

            # Step 1: Load data
            _progress_state["train"]["step"] = "loading"
            _progress_state["train"]["message"] = "Memuat data..."
            _progress_state["train"]["progress"] = 5

            # Cek apakah ada data preprocessed
            eval_prep_path = engine.DATA_DIR / "eval_preprocessed.csv"
            if eval_prep_path.exists():
                df = pd.read_csv(eval_prep_path, parse_dates=["datetime"])
                feature_cols = [c for c in df.columns if c not in
                               {"datetime", engine.TARGET, "hour", "day_of_week", "month"}]
            else:
                # Load dari dataset lokal atau existing data
                if source == "local":
                    raw_df = engine.load_local_dataset()
                    if raw_df is None:
                        raw_df = engine.load_existing_data()
                else:
                    raw_df = engine.load_existing_data()

                if raw_df is None:
                    _progress_state["train"]["message"] = "Error: Tidak ada data tersedia"
                    _progress_state["train"]["running"] = False
                    return

                # Subset kolom
                needed_cols = ["datetime"] + engine.PARAMETERS
                available = [c for c in needed_cols if c in raw_df.columns]
                raw_df = raw_df[available].copy()

                _progress_state["train"]["step"] = "preprocessing"
                _progress_state["train"]["message"] = "Preprocessing data..."
                _progress_state["train"]["progress"] = 10

                prep_result = engine.preprocess_for_training(raw_df)
                df = prep_result["dataframe"]
                feature_cols = prep_result["feature_columns"]

            # Step 2: Split data
            _progress_state["train"]["step"] = "splitting"
            _progress_state["train"]["message"] = f"Split data ({len(df)} records)..."
            _progress_state["train"]["progress"] = 15

            # Get test ratio from params
            test_ratio = (params.test_split_pct / 100.0) if hasattr(params, 'test_split_pct') else engine.TEST_RATIO
            
            X_train, X_test, y_train, y_test, dt_train, dt_test = engine.split_chronological(
                df, feature_cols, test_ratio=test_ratio
            )

            # Hitung rentang data untuk experiment history
            data_range_start = df["datetime"].min() if "datetime" in df.columns else None
            data_range_end = df["datetime"].max() if "datetime" in df.columns else None

            # Build hyperparams dictionary
            hp_dict = engine.DEFAULT_XGBOOST_PARAMS.copy()
            hp_dict["n_estimators"] = params.n_estimators
            hp_dict["max_depth"] = params.max_depth
            hp_dict["learning_rate"] = params.learning_rate

            # Create experiment record
            experiment = EvalExperiment(
                model_type="XGBRegressor",
                hyperparameters=json.dumps(hp_dict),
                n_train=len(X_train),
                n_test=len(X_test),
                n_features=len(feature_cols),
                feature_columns=json.dumps(feature_cols),
                data_range_start=data_range_start,
                data_range_end=data_range_end,
                total_data_records=len(df),
                horizon_prediction="24h",
                status="training",
            )
            db.add(experiment)
            db.commit()
            db.refresh(experiment)

            # Step 3: Training
            def train_progress_cb(msg, pct):
                _progress_state["train"]["step"] = "training"
                _progress_state["train"]["message"] = msg
                _progress_state["train"]["progress"] = 15 + int(pct * 0.6)

            train_result = engine.train_model(
                X_train, y_train, X_test, y_test,
                params=hp_dict,
                progress_callback=train_progress_cb,
            )

            # Step 4: Save model
            _progress_state["train"]["step"] = "saving"
            _progress_state["train"]["message"] = "Menyimpan model..."
            _progress_state["train"]["progress"] = 80

            engine.save_model(train_result["model"])
            engine.save_feature_info(feature_cols, len(X_train), len(X_test))
            engine.save_training_report(train_result, feature_cols, len(X_train), len(X_test), params=hp_dict)

            # Save train/test CSVs
            X_train.to_csv(engine.DATA_DIR / "X_train.csv", index=False)
            X_test.to_csv(engine.DATA_DIR / "X_test.csv", index=False)
            y_train.to_csv(engine.DATA_DIR / "y_train.csv", index=False)
            y_test.to_csv(engine.DATA_DIR / "y_test.csv", index=False)
            df.to_csv(engine.DATA_DIR / "processed_data.csv", index=False)

            # Step 5: Save to eval DB
            _progress_state["train"]["step"] = "evaluating"
            _progress_state["train"]["message"] = "Menyimpan hasil evaluasi..."
            _progress_state["train"]["progress"] = 85

            experiment.training_duration_seconds = train_result["training_duration"]
            experiment.best_iteration = train_result["best_iteration"]
            experiment.cv_scores = json.dumps(train_result["cv_scores"])
            experiment.status = "evaluating"
            db.commit()

            # Save metrics
            for split_name, metrics in [("train", train_result["train_metrics"]),
                                         ("test", train_result["test_metrics"])]:
                metric = EvalMetric(
                    experiment_id=experiment.id,
                    dataset_split=split_name,
                    mae=metrics["mae"],
                    rmse=metrics["rmse"],
                    r2=metrics["r2"],
                    mape=metrics["mape"],
                    max_error=metrics["max_error"],
                    median_ae=metrics["median_ae"],
                )
                db.add(metric)

            # Save feature importance
            for fi in train_result["feature_importance"]:
                feat = EvalFeatureImportance(
                    experiment_id=experiment.id,
                    feature_name=fi["feature"],
                    importance=fi["importance"],
                    rank=fi["rank"],
                )
                db.add(feat)

            # Save predictions
            dt_test_list = dt_test.tolist()
            dt_train_list = dt_train.tolist()

            # Save test predictions
            for i, (dt_val, actual, pred) in enumerate(zip(
                dt_test_list,
                y_test.values.tolist(),
                train_result["test_predictions"]
            )):
                p = EvalPrediction(
                    experiment_id=experiment.id,
                    datetime_val=dt_val,
                    actual=actual,
                    predicted=pred,
                    residual=actual - pred,
                    dataset_split="test",
                )
                db.add(p)

            db.commit()

            # Step 6: Finalisasi

            experiment.status = "done"
            db.commit()

            _progress_state["train"]["step"] = "done"
            _progress_state["train"]["message"] = "Training selesai!"
            _progress_state["train"]["progress"] = 100

        except Exception as e:
            _progress_state["train"]["message"] = f"Error: {str(e)}"
            _progress_state["train"]["step"] = "error"
            import traceback
            traceback.print_exc()
        finally:
            _progress_state["train"]["running"] = False
            db.close()

    # Run in thread to avoid blocking
    import threading
    thread = threading.Thread(target=run_training, daemon=True)
    thread.start()

    return {"status": "started", "message": "Training dimulai"}


@eval_router.get("/api/train-progress")
async def get_train_progress():
    """Stream progress training via SSE."""
    async def event_stream():
        while True:
            state = _progress_state["train"]
            data = json.dumps({
                "message": state["message"],
                "progress": state["progress"],
                "step": state["step"],
                "running": state["running"],
            })
            yield f"data: {data}\n\n"

            if not state["running"]:
                break
            await asyncio.sleep(1)

    return StreamingResponse(event_stream(), media_type="text/event-stream")


# ==============================================================================
# EVALUATION RESULTS
# ==============================================================================
@eval_router.get("/api/metrics")
async def get_metrics():
    """Ambil metrik evaluasi terbaru."""
    db = SessionLocal()
    try:
        # Ambil eksperimen terbaru yang done
        experiment = db.query(EvalExperiment).filter(
            EvalExperiment.status == "done"
        ).order_by(EvalExperiment.created_at.desc()).first()

        if not experiment:
            # Fallback ke training_report.json
            report_path = engine.MODEL_DIR / "training_report.json"
            if report_path.exists():
                with open(report_path) as f:
                    report = json.load(f)
                return {
                    "source": "training_report",
                    "experiment_id": None,
                    "train": report.get("final_metrics", {}).get("train", {}),
                    "test": report.get("final_metrics", {}).get("test", {}),
                    "cv_scores": report.get("cv_scores", {}),
                }
            raise HTTPException(status_code=404, detail="Belum ada hasil evaluasi")

        metrics = db.query(EvalMetric).filter(
            EvalMetric.experiment_id == experiment.id
        ).all()

        result = {
            "source": "eval_db",
            "experiment_id": experiment.id,
            "experiment": experiment.to_dict(),
            "train": {},
            "test": {},
            "cv_scores": json.loads(experiment.cv_scores) if experiment.cv_scores else {},
        }

        for m in metrics:
            result[m.dataset_split] = m.to_dict()

        return result
    finally:
        db.close()


@eval_router.get("/api/feature-importance")
async def get_feature_importance():
    """Ambil feature importance terbaru."""
    db = SessionLocal()
    try:
        experiment = db.query(EvalExperiment).filter(
            EvalExperiment.status == "done"
        ).order_by(EvalExperiment.created_at.desc()).first()

        if not experiment:
            # Fallback ke CSV
            csv_path = engine.MODEL_DIR / "feature_importance.csv"
            if csv_path.exists():
                df = pd.read_csv(csv_path)
                return {
                    "source": "csv",
                    "features": df.to_dict(orient="records"),
                }
            raise HTTPException(status_code=404, detail="Feature importance belum tersedia")

        features = db.query(EvalFeatureImportance).filter(
            EvalFeatureImportance.experiment_id == experiment.id
        ).order_by(EvalFeatureImportance.rank).all()

        return {
            "source": "eval_db",
            "experiment_id": experiment.id,
            "features": [f.to_dict() for f in features],
        }
    finally:
        db.close()





@eval_router.get("/api/predictions")
async def get_predictions(split: str = Query(default="test")):
    """Ambil data actual vs predicted."""
    db = SessionLocal()
    try:
        experiment = db.query(EvalExperiment).filter(
            EvalExperiment.status == "done"
        ).order_by(EvalExperiment.created_at.desc()).first()

        if not experiment:
            raise HTTPException(status_code=404, detail="Belum ada prediksi")

        preds = db.query(EvalPrediction).filter(
            EvalPrediction.experiment_id == experiment.id,
            EvalPrediction.dataset_split == split,
        ).order_by(EvalPrediction.datetime_val).all()

        return {
            "experiment_id": experiment.id,
            "split": split,
            "count": len(preds),
            "predictions": [p.to_dict() for p in preds],
        }
    finally:
        db.close()


@eval_router.get("/api/residuals")
async def get_residuals():
    """Ambil data residual analysis."""
    db = SessionLocal()
    try:
        experiment = db.query(EvalExperiment).filter(
            EvalExperiment.status == "done"
        ).order_by(EvalExperiment.created_at.desc()).first()

        if not experiment:
            raise HTTPException(status_code=404, detail="Belum ada data residual")

        preds = db.query(EvalPrediction).filter(
            EvalPrediction.experiment_id == experiment.id,
            EvalPrediction.dataset_split == "test",
        ).order_by(EvalPrediction.datetime_val).all()

        residuals = [p.residual for p in preds if p.residual is not None]

        # Histogram residual
        if residuals:
            hist, bin_edges = np.histogram(residuals, bins=30)
            residual_hist = {
                "counts": hist.tolist(),
                "bin_edges": [round(float(b), 3) for b in bin_edges],
            }
        else:
            residual_hist = {"counts": [], "bin_edges": []}

        return {
            "experiment_id": experiment.id,
            "predictions": [p.to_dict() for p in preds],
            "residual_histogram": residual_hist,
            "stats": {
                "mean": round(float(np.mean(residuals)), 4) if residuals else 0,
                "std": round(float(np.std(residuals)), 4) if residuals else 0,
                "min": round(float(np.min(residuals)), 4) if residuals else 0,
                "max": round(float(np.max(residuals)), 4) if residuals else 0,
            },
        }
    finally:
        db.close()


# ==============================================================================
# EXPERIMENT HISTORY (enhanced)
# ==============================================================================
@eval_router.get("/api/experiment-history")
async def get_experiment_history():
    """
    Riwayat eksperimen lengkap untuk perbandingan.
    Mencakup: tanggal training, jumlah data, rentang data,
    model, horizon, MAE, RMSE, R², MAPE.
    """
    db = SessionLocal()
    try:
        experiments = db.query(EvalExperiment).order_by(
            EvalExperiment.created_at.desc()
        ).limit(50).all()

        result = []
        for exp in experiments:
            exp_dict = exp.to_dict()

            # Ambil metrik test
            test_metric = db.query(EvalMetric).filter(
                EvalMetric.experiment_id == exp.id,
                EvalMetric.dataset_split == "test",
            ).first()
            if test_metric:
                exp_dict["test_metrics"] = test_metric.to_dict()

            # Ambil metrik train
            train_metric = db.query(EvalMetric).filter(
                EvalMetric.experiment_id == exp.id,
                EvalMetric.dataset_split == "train",
            ).first()
            if train_metric:
                exp_dict["train_metrics"] = train_metric.to_dict()



            result.append(exp_dict)

        return {"experiments": result}
    finally:
        db.close()


@eval_router.get("/api/experiments")
async def get_experiments():
    """Riwayat semua eksperimen (backward compat)."""
    return await get_experiment_history()


@eval_router.get("/api/preprocessing-log")
async def get_preprocessing_log():
    """Ambil log preprocessing terbaru."""
    db = SessionLocal()
    try:
        log = db.query(EvalPreprocessingLog).order_by(
            EvalPreprocessingLog.created_at.desc()
        ).first()

        if not log:
            raise HTTPException(status_code=404, detail="Belum ada preprocessing log")

        return log.to_dict()
    finally:
        db.close()
