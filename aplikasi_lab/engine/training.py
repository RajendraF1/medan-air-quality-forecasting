"""
Training Service — Production Training
==========================================
Training model XGBoost untuk production.

TIDAK melakukan:
  - Grid Search
  - Random Search
  - Hyperparameter Tuning
  - Multi Horizon Evaluation

Menggunakan:
  - Hyperparameters final dari Dashboard Evaluasi
  - Pipeline final
  - Model Safety (Last Valid Model)
"""

import json
import time
import shutil
import numpy as np
import pandas as pd
import joblib
from typing import Any, Callable, Dict, List, Optional

from xgboost import XGBRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from core.config import (
    XGBOOST_PARAMS, TEST_RATIO, N_CV_SPLITS, TARGET, EARLY_STOPPING_ROUNDS, PARAMETERS,
    MODEL_PATH, MODEL_NEW_PATH, TRAINING_REPORT_PATH,
    FEATURE_COLS_PATH, FEATURE_IMPORTANCE_PATH,
    MODEL_DIR, DATA_DIR, DATASET_DAYS,
)
from core.utils.timezone import now_wib, format_wib_iso
from aplikasi_lab.engine.fetching import load_local_dataset
from aplikasi_lab.engine.preprocessing import preprocess_for_training


DEFAULT_XGBOOST_PARAMS = {
    "n_estimators": 1000,
    "learning_rate": 0.05,
    "max_depth": 6,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 3,
    "gamma": 0.1,
    "random_state": 42
}


# ==============================================================================
# Training Pipeline
# ==============================================================================
def run_training() -> Dict[str, Any]:
    """
    Jalankan training production.

    Alur:
      1. Load dataset lokal
      2. Preprocess
      3. Split (chronological)
      4. Train XGBoost dengan CV
      5. Evaluasi cepat
      6. Model Safety: validasi sebelum replace
      7. Save model + report

    Returns:
        dict: status, metrics, duration, model_version
    """
    start_time = time.time()
    print("[TRAIN] Memulai training production...")

    # Step 1: Load data
    df_raw = load_local_dataset()
    if df_raw is None or len(df_raw) < 100:
        return {
            "status": "error",
            "message": "Dataset tidak cukup untuk training (minimal 100 records).",
        }

    # Ambil hanya kolom raw
    needed_cols = ["datetime"] + PARAMETERS
    available = [c for c in needed_cols if c in df_raw.columns]
    df_raw = df_raw[available].copy()

    print(f"[TRAIN] Dataset: {len(df_raw)} records")

    # Step 2: Preprocess
    try:
        prep_result = preprocess_for_training(df_raw)
        df = prep_result["dataframe"]
        feature_cols = prep_result["feature_columns"]
        print(f"[TRAIN] Preprocessed: {prep_result['clean_records']} records, {len(feature_cols)} features")
    except Exception as e:
        return {
            "status": "error",
            "message": f"Preprocessing gagal: {str(e)}",
        }

    if len(df) < 50:
        return {
            "status": "error",
            "message": f"Data setelah preprocessing terlalu sedikit ({len(df)} records).",
        }

    # Step 3: Split (chronological)
    X = df[feature_cols]
    y = df[TARGET]
    split_idx = int(len(df) * (1 - TEST_RATIO))

    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    print(f"[TRAIN] Split: train={len(X_train)}, test={len(X_test)}")

    # Step 4: Cross-Validation
    try:
        tscv = TimeSeriesSplit(n_splits=N_CV_SPLITS)
        cv_scores = {"mae": [], "rmse": [], "r2": []}

        for fold, (train_idx, val_idx) in enumerate(tscv.split(X_train), 1):
            X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
            y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]

            fold_model = XGBRegressor(**XGBOOST_PARAMS, early_stopping_rounds=EARLY_STOPPING_ROUNDS)
            fold_model.fit(
                X_tr, y_tr,
                eval_set=[(X_val, y_val)],
                verbose=False
            )

            y_pred = fold_model.predict(X_val)
            cv_scores["mae"].append(float(mean_absolute_error(y_val, y_pred)))
            cv_scores["rmse"].append(float(np.sqrt(mean_squared_error(y_val, y_pred))))
            cv_scores["r2"].append(float(r2_score(y_val, y_pred)))

        print(f"[TRAIN] CV selesai. MAE mean: {np.mean(cv_scores['mae']):.4f}")

        # Step 5: Train final one-step model untuk recursive forecasting
        model = XGBRegressor(**XGBOOST_PARAMS, early_stopping_rounds=EARLY_STOPPING_ROUNDS)
        model.fit(
            X_train, y_train,
            eval_set=[(X_train, y_train), (X_test, y_test)],
            verbose=False
        )

        # Evaluasi
        y_test_pred = model.predict(X_test)
        y_train_pred = model.predict(X_train)

        test_metrics = _calc_metrics(y_test.values, y_test_pred)
        train_metrics = _calc_metrics(y_train.values, y_train_pred)

        print(f"[TRAIN] Test MAE: {test_metrics['mae']:.4f}, R²: {test_metrics['r2']:.4f}")

    except Exception as e:
        return {
            "status": "error",
            "message": f"Training gagal: {str(e)}",
        }

    # Step 6: Model Safety
    model_version = now_wib().strftime("%Y%m%d_%H%M%S")

    # Validasi: model baru harus punya R² > 0.5 dan MAE reasonable
    if test_metrics["r2"] < 0.5:
        print(f"[TRAIN] Model baru gagal validasi (R²={test_metrics['r2']:.4f} < 0.5)")
        return {
            "status": "validation_failed",
            "message": f"Model baru tidak lolos validasi (R²={test_metrics['r2']:.4f}). Model lama tetap digunakan.",
            "metrics": test_metrics,
        }

    # Simpan model baru sebagai temporary dulu
    try:
        joblib.dump(model, MODEL_NEW_PATH)

        # Jika berhasil, replace model aktif
        if MODEL_PATH.exists():
            # Backup model lama
            backup_path = MODEL_DIR / "pm25_xgboost_model_backup.joblib"
            shutil.copy2(MODEL_PATH, backup_path)

        # Replace
        shutil.move(str(MODEL_NEW_PATH), str(MODEL_PATH))
        print(f"[TRAIN] Model baru aktif: {model_version}")

    except Exception as e:
        print(f"[TRAIN] Gagal menyimpan model: {e}")
        return {
            "status": "error",
            "message": f"Gagal menyimpan model: {str(e)}",
        }

    # Step 7: Save feature info & report
    _save_feature_info(feature_cols, len(X_train), len(X_test))
    _save_training_report(
        model_version, train_metrics, test_metrics,
        cv_scores, feature_cols, len(X_train), len(X_test),
        model, time.time() - start_time,
    )

    # Feature importance CSV
    importance = model.feature_importances_
    actual_features = getattr(model, "feature_names_in_", None)
    if actual_features is None:
        actual_features = model.get_booster().feature_names
        
    feat_imp = pd.DataFrame({
        "feature": actual_features,
        "importance": importance,
    }).sort_values("importance", ascending=False).reset_index(drop=True)
    feat_imp["rank"] = range(1, len(feat_imp) + 1)
    feat_imp.to_csv(FEATURE_IMPORTANCE_PATH, index=False)

    duration = time.time() - start_time
    print(f"[TRAIN] Selesai! Durasi: {duration:.1f}s")

    return {
        "status": "success",
        "model_version": model_version,
        "train_metrics": train_metrics,
        "test_metrics": test_metrics,
        "n_train": len(X_train),
        "n_test": len(X_test),
        "n_features": len(feature_cols),
        "duration": round(duration, 2),
        "message": f"Training selesai. MAE={test_metrics['mae']:.4f}, R²={test_metrics['r2']:.4f}",
    }


# ==============================================================================
# Helpers
# ==============================================================================
def _calc_metrics(y_true, y_pred):
    """Hitung metrik evaluasi."""
    return {
        "mae": round(float(mean_absolute_error(y_true, y_pred)), 4),
        "rmse": round(float(np.sqrt(mean_squared_error(y_true, y_pred))), 4),
        "r2": round(float(r2_score(y_true, y_pred)), 4),
        "mape": round(
            float(np.mean(np.abs((y_true - y_pred) / (y_true + 1e-6))) * 100), 2
        ),
    }


def _save_feature_info(feature_cols, n_train, n_test):
    """Simpan feature columns info."""
    info = {
        "feature_columns": feature_cols,
        "target": TARGET,
        "n_features": len(feature_cols),
        "n_train": n_train,
        "n_test": n_test,
        "parameters_used": PARAMETERS,
    }
    with open(FEATURE_COLS_PATH, "w") as f:
        json.dump(info, f, indent=2)


def _save_training_report(
    model_version, train_metrics, test_metrics,
    cv_scores, feature_cols, n_train, n_test,
    model, duration, params=None
):
    """Simpan training report."""
    # Feature importance top 10
    importance = model.feature_importances_
    actual_features = getattr(model, "feature_names_in_", None)
    if actual_features is None:
        actual_features = model.get_booster().feature_names
        
    feat_imp = pd.DataFrame({
        "feature": actual_features,
        "importance": importance,
    }).sort_values("importance", ascending=False)

    report = {
        "model_type": "XGBRegressor",
        "model_version": model_version,
        "hyperparameters": params if params else XGBOOST_PARAMS,
        "cv_scores": {
            "mae_mean": float(np.mean(cv_scores["mae"])),
            "rmse_mean": float(np.mean(cv_scores["rmse"])),
            "r2_mean": float(np.mean(cv_scores["r2"])),
        },
        "final_metrics": {
            "train": train_metrics,
            "test": test_metrics,
        },
        "top_10_features": feat_imp["feature"].head(10).tolist(),
        "n_features": len(feature_cols),
        "n_train": n_train,
        "n_test": n_test,
        "training_duration_seconds": round(duration, 2),
        "trained_at": format_wib_iso(now_wib()),
    }

    with open(TRAINING_REPORT_PATH, "w") as f:
        json.dump(report, f, indent=2)


def save_feature_info(feature_cols, n_train, n_test):
    """Public wrapper for saving feature columns info."""
    _save_feature_info(feature_cols, n_train, n_test)


def save_training_report(train_result, feature_cols, n_train, n_test, params=None):
    """Public wrapper for saving training report from dashboard result."""
    # Resolve duration properly
    duration = train_result.get("training_duration", train_result.get("duration", 0))
    # Resolve cv_scores properly (could be nested under per_fold or direct)
    cv_scores = train_result["cv_scores"].get("per_fold", train_result["cv_scores"]) if isinstance(train_result.get("cv_scores"), dict) else {"mae":[], "rmse":[], "r2":[]}

    _save_training_report(
        model_version=train_result.get("model_version", now_wib().strftime("%Y%m%d_%H%M%S")),
        train_metrics=train_result.get("train_metrics", {}),
        test_metrics=train_result.get("test_metrics", {}),
        cv_scores=cv_scores,
        feature_cols=feature_cols,
        n_train=n_train,
        n_test=n_test,
        model=train_result["model"],
        duration=duration,
        params=params
    )


def load_model():
    """Load model aktif dan feature columns."""
    model = None
    feature_columns = None

    if MODEL_PATH.exists():
        model = joblib.load(MODEL_PATH)
        print(f"[MODEL] Loaded: {MODEL_PATH.name}")

    if FEATURE_COLS_PATH.exists():
        with open(FEATURE_COLS_PATH, "r") as f:
            info = json.load(f)
            feature_columns = info.get("feature_columns", [])
        print(f"[MODEL] Features: {len(feature_columns)}")

    model_features = getattr(model, "feature_names_in_", None) if model is not None else None
    if model_features is not None:
        model_features = [str(feature) for feature in model_features]
        if feature_columns != model_features:
            feature_columns = model_features
            print(f"[MODEL] Using model feature order: {len(feature_columns)}")

    return model, feature_columns


def get_training_report() -> Optional[Dict]:
    """Load training report."""
    if TRAINING_REPORT_PATH.exists():
        with open(TRAINING_REPORT_PATH) as f:
            return json.load(f)
    return None


def train_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    params: Optional[Dict] = None,
    progress_callback: Optional[Callable] = None,
) -> Dict[str, Any]:
    """
    Train XGBoost model dengan TimeSeriesSplit CV.

    Returns:
        dict: model, metrics, cv_scores, feature_importance, training_duration
    """
    if params is None:
        params = DEFAULT_XGBOOST_PARAMS.copy()

    start_time = time.time()

    # --- Cross-Validation ---
    if progress_callback:
        progress_callback("Cross-validation dimulai...", 20)

    tscv = TimeSeriesSplit(n_splits=N_CV_SPLITS)
    cv_scores = {"mae": [], "rmse": [], "r2": []}

    for fold, (train_idx, val_idx) in enumerate(tscv.split(X_train), 1):
        X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
        y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]

        fold_model = XGBRegressor(**params, early_stopping_rounds=50)
        fold_model.fit(
            X_tr, y_tr,
            eval_set=[(X_val, y_val)],
            verbose=False
        )

        y_pred = fold_model.predict(X_val)
        cv_scores["mae"].append(float(mean_absolute_error(y_val, y_pred)))
        cv_scores["rmse"].append(float(np.sqrt(mean_squared_error(y_val, y_pred))))
        cv_scores["r2"].append(float(r2_score(y_val, y_pred)))

        if progress_callback:
            progress_callback(f"CV Fold {fold}/{N_CV_SPLITS} selesai", 20 + fold * 10)

    # --- Train Final Model ---
    if progress_callback:
        progress_callback("Training model final...", 75)

    model = XGBRegressor(**params, early_stopping_rounds=50)
    model.fit(
        X_train, y_train,
        eval_set=[(X_train, y_train), (X_test, y_test)],
        verbose=False
    )

    training_duration = time.time() - start_time

    # --- Evaluate ---
    if progress_callback:
        progress_callback("Menghitung metrik evaluasi...", 90)

    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    def calc_metrics(y_true, y_pred):
        return {
            "mae": float(mean_absolute_error(y_true, y_pred)),
            "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
            "r2": float(r2_score(y_true, y_pred)),
            "mape": float(np.mean(np.abs((y_true - y_pred) / (y_true + 1e-6))) * 100),
            "max_error": float(np.max(np.abs(y_true - y_pred))),
            "median_ae": float(np.median(np.abs(y_true - y_pred))),
        }

    train_metrics = calc_metrics(y_train.values, y_train_pred)
    test_metrics = calc_metrics(y_test.values, y_test_pred)

    # --- Feature Importance ---
    importance = model.feature_importances_
    feat_imp = pd.DataFrame({
        "feature": X_train.columns.tolist(),
        "importance": importance,
    }).sort_values("importance", ascending=False).reset_index(drop=True)
    feat_imp["rank"] = range(1, len(feat_imp) + 1)

    if progress_callback:
        progress_callback("Training selesai!", 100)

    return {
        "model": model,
        "train_predictions": y_train_pred.tolist(),
        "test_predictions": y_test_pred.tolist(),
        "train_metrics": train_metrics,
        "test_metrics": test_metrics,
        "cv_scores": {
            "mae_mean": float(np.mean(cv_scores["mae"])),
            "mae_std": float(np.std(cv_scores["mae"])),
            "rmse_mean": float(np.mean(cv_scores["rmse"])),
            "rmse_std": float(np.std(cv_scores["rmse"])),
            "r2_mean": float(np.mean(cv_scores["r2"])),
            "r2_std": float(np.std(cv_scores["r2"])),
            "per_fold": cv_scores,
        },
        "feature_importance": feat_imp.to_dict(orient="records"),
        "training_duration": round(training_duration, 2),
        "best_iteration": getattr(model, "best_iteration", None),
    }




def save_model(model, filename: str = "pm25_xgboost_model.joblib"):
    """Simpan model ke file."""
    path = MODEL_DIR / filename
    joblib.dump(model, path)
    return str(path)


def load_existing_data() -> Optional[pd.DataFrame]:
    """Load data yang sudah ada dari processed_data.csv (backward compat)."""
    path = DATA_DIR / "processed_data.csv"
    if path.exists():
        df = pd.read_csv(path, parse_dates=["datetime"])
        return df
    return None


def load_existing_model():
    """Load model yang sudah ada."""
    path = MODEL_DIR / "pm25_xgboost_model.joblib"
    if path.exists():
        return joblib.load(path)
    return None


def load_feature_columns() -> Optional[List[str]]:
    """Load feature columns dari file."""
    path = DATA_DIR / "feature_columns.json"
    if path.exists():
        with open(path) as f:
            info = json.load(f)
        return info.get("feature_columns", [])
    return None
