"""
Data Fetching Service — Sinkronisasi OpenAQ
=============================================
Inkremental fetch data dari OpenAQ API.

Logika:
  1. Cek dataset lokal → ambil timestamp terakhir
  2. Cek data terbaru di OpenAQ
  3. Jika ada data baru → update CSV bulanan
  4. Jika tidak ada → status "Waiting For New Data" (bukan error)

PENTING:
  - Seluruh timestamp WIB (UTC+7)
  - Hanya LOCATION_ID = 5586536
  - Tidak membuat processed_data.csv
"""

import asyncio
import time
import numpy as np
import pandas as pd
import httpx
from pathlib import Path
from datetime import datetime, timedelta, timezone
from typing import Optional, Dict, Any

from core.config import (
    OPENAQ_BASE_URL, OPENAQ_API_KEY, LOCATION_ID,
    PARAMETERS, RAW_DATA_DIR, DATASET_DAYS, TIMEZONE_WIB,
)
from core.utils.timezone import now_wib, format_wib, to_wib


# ==============================================================================
# Helpers — CSV Lokal
# ==============================================================================
def _get_csv_month_path(year: int, month: int) -> Path:
    """Path file CSV untuk bulan tertentu."""
    return RAW_DATA_DIR / f"{year:04d}-{month:02d}.csv"


def get_latest_local_timestamp() -> Optional[datetime]:
    """
    Ambil timestamp terakhir dari dataset lokal.
    Returns datetime (WIB) atau None.
    """
    csv_files = sorted(RAW_DATA_DIR.glob("*.csv"), reverse=True)

    for csv_path in csv_files:
        try:
            df = pd.read_csv(csv_path, parse_dates=["datetime"])
            if len(df) > 0:
                max_dt = df["datetime"].max()
                if max_dt.tzinfo is None:
                    max_dt = max_dt.replace(tzinfo=TIMEZONE_WIB)
                return max_dt
        except Exception:
            continue

    return None


def load_local_dataset(days: Optional[int] = None) -> Optional[pd.DataFrame]:
    """
    Load semua CSV lokal yang masuk dalam rentang hari.

    Returns:
        DataFrame gabungan (datetime naive, WIB) atau None.
    """
    csv_files = sorted(RAW_DATA_DIR.glob("*.csv"))
    if not csv_files:
        return None

    if days is not None:
        cutoff = now_wib() - timedelta(days=days)
    else:
        cutoff = None
        
    all_frames = []

    for csv_path in csv_files:
        try:
            df = pd.read_csv(csv_path, parse_dates=["datetime"])
            if len(df) > 0:
                if df["datetime"].dt.tz is None:
                    df["datetime"] = df["datetime"].dt.tz_localize(TIMEZONE_WIB)
                
                if cutoff is not None:
                    mask = df["datetime"] >= cutoff
                    filtered = df[mask]
                else:
                    filtered = df
                    
                if len(filtered) > 0:
                    all_frames.append(filtered)
        except Exception:
            continue

    if not all_frames:
        return None

    combined = pd.concat(all_frames, ignore_index=True)
    combined = combined.sort_values("datetime").reset_index(drop=True)
    combined = combined.drop_duplicates(subset=["datetime"], keep="last")
    combined = combined.reset_index(drop=True)

    # Strip timezone untuk kompatibilitas downstream
    combined["datetime"] = combined["datetime"].dt.tz_localize(None)

    return combined


def scan_local_dataset() -> Dict[str, Any]:
    """Scan semua file CSV di data/kualitas_udara/."""
    csv_files = sorted(RAW_DATA_DIR.glob("*.csv"))

    if not csv_files:
        return {
            "total_records": 0,
            "total_files": 0,
            "last_timestamp": None,
        }

    df = load_local_dataset()
    if df is not None and len(df) > 0:
        return {
            "total_records": len(df),
            "total_files": len(csv_files),
            "last_timestamp": df["datetime"].max(),
        }

    return {
        "total_records": 0,
        "total_files": len(csv_files),
        "last_timestamp": None,
    }


def _save_to_monthly_csv(df: pd.DataFrame):
    """Simpan DataFrame ke file CSV bulanan (append jika ada)."""
    if len(df) == 0:
        return

    df = df.copy()
    df["datetime"] = pd.to_datetime(df["datetime"])
    df["_year"] = df["datetime"].dt.year
    df["_month"] = df["datetime"].dt.month

    for (year, month), group in df.groupby(["_year", "_month"]):
        csv_path = _get_csv_month_path(int(year), int(month))
        save_df = group.drop(columns=["_year", "_month"])

        raw_cols = ["datetime"] + PARAMETERS
        available_cols = [c for c in raw_cols if c in save_df.columns]
        save_df = save_df[available_cols]

        if csv_path.exists():
            existing = pd.read_csv(csv_path, parse_dates=["datetime"])
            combined = pd.concat([existing, save_df], ignore_index=True)
            combined = combined.drop_duplicates(subset=["datetime"], keep="last")
            combined = combined.sort_values("datetime").reset_index(drop=True)
            combined.to_csv(csv_path, index=False)
        else:
            save_df = save_df.sort_values("datetime").reset_index(drop=True)
            save_df.to_csv(csv_path, index=False)


# ==============================================================================
# Sinkronisasi Utama
# ==============================================================================
async def sync_data() -> Dict[str, Any]:
    """
    Sinkronisasi data dari OpenAQ API secara inkremental.

    Returns:
        dict: status, new_records, message
    """
    headers = {}
    if OPENAQ_API_KEY:
        headers["X-API-Key"] = OPENAQ_API_KEY

    current_time = now_wib()
    fetch_start = time.time()

    # Step 1: Cek dataset lokal
    last_local_ts = get_latest_local_timestamp()

    if last_local_ts is None:
        start_date = current_time - timedelta(days=DATASET_DAYS)
        print(f"[FETCH] Dataset kosong. Fetch {DATASET_DAYS} hari terakhir...")
    else:
        gap = current_time - last_local_ts
        gap_hours = gap.total_seconds() / 3600

        if gap_hours <= 1.5:
            return {
                "status": "synced",
                "new_records": 0,
                "message": f"Dataset sudah sinkron. Data hingga {format_wib(last_local_ts)}",
            }

        start_date = last_local_ts
        print(f"[FETCH] Gap {gap_hours:.1f} jam. Fetch mulai {format_wib(last_local_ts)}...")

    end_date = current_time

    # Step 2: Fetch dari OpenAQ
    all_data = []
    total_requests = 0

    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            # Fetch sensors
            sensors = []
            loc_url = f"{OPENAQ_BASE_URL}/v3/locations/{LOCATION_ID}"
            loc_resp = await client.get(loc_url, headers=headers)
            total_requests += 1

            if loc_resp.status_code == 200:
                loc_data = loc_resp.json()
                results = loc_data.get("results", [])
                if results:
                    sensors = results[0].get("sensors", [])

            # Fetch measurements
            current_start = start_date

            while current_start < end_date:
                current_end = min(current_start + timedelta(days=30), end_date)

                date_from = current_start.astimezone(timezone.utc).strftime(
                    "%Y-%m-%dT%H:%M:%SZ"
                )
                date_to = current_end.astimezone(timezone.utc).strftime(
                    "%Y-%m-%dT%H:%M:%SZ"
                )

                for sensor in sensors:
                    sensor_id = sensor.get("id")
                    if not sensor_id:
                        continue

                    page = 1
                    has_more = True

                    while has_more:
                        try:
                            url = f"{OPENAQ_BASE_URL}/v3/sensors/{sensor_id}/measurements"
                            params = {
                                "datetime_from": date_from,
                                "datetime_to": date_to,
                                "page": page,
                                "limit": 200,
                            }

                            response = await client.get(
                                url, headers=headers, params=params
                            )
                            total_requests += 1

                            if response.status_code == 200:
                                data = response.json()
                                api_results = data.get("results", [])

                                if api_results:
                                    for r in api_results:
                                        dt_info = r.get("period", {}).get(
                                            "datetimeFrom", {}
                                        )
                                        record = {
                                            "datetime_local": dt_info.get(
                                                "local", dt_info.get("utc", "")
                                            ),
                                            "parameter_name": r.get(
                                                "parameter", {}
                                            ).get("name", ""),
                                            "value": r.get("value", None),
                                        }
                                        if (
                                            record["parameter_name"] in PARAMETERS
                                            and record["value"] is not None
                                        ):
                                            all_data.append(record)

                                    if len(api_results) < 200:
                                        has_more = False
                                    else:
                                        page += 1
                                else:
                                    has_more = False
                            elif response.status_code == 429:
                                print("[FETCH] Rate limited, waiting 5s...")
                                await asyncio.sleep(5)
                                continue
                            else:
                                has_more = False

                            await asyncio.sleep(0.3)

                        except Exception:
                            has_more = False

                current_start = current_end

    except Exception as e:
        print(f"[FETCH] Error: {e}")
        return {
            "status": "error",
            "new_records": 0,
            "message": f"Gagal koneksi ke OpenAQ: {str(e)}",
        }

    fetch_duration = time.time() - fetch_start

    # Step 3: Proses dan simpan
    if not all_data:
        # Tidak ada data baru — OpenAQ belum punya data terbaru
        last_ts = get_latest_local_timestamp()
        return {
            "status": "waiting_for_new_data",
            "new_records": 0,
            "message": (
                f"Menunggu data baru dari OpenAQ. "
                f"Data tersedia hingga {format_wib(last_ts) if last_ts else '-'}"
            ),
        }

    # Konversi ke DataFrame
    df = pd.DataFrame(all_data)
    df["datetime_local"] = pd.to_datetime(df["datetime_local"])

    # Pivot long → wide
    df_wide = df.pivot_table(
        index="datetime_local",
        columns="parameter_name",
        values="value",
        aggfunc="mean",
    ).reset_index()
    df_wide = df_wide.rename(columns={"datetime_local": "datetime"})
    df_wide = df_wide.sort_values("datetime").reset_index(drop=True)

    for param in PARAMETERS:
        if param not in df_wide.columns:
            df_wide[param] = np.nan

    # Simpan ke CSV bulanan
    _save_to_monthly_csv(df_wide)

    new_records = len(df_wide)
    print(
        f"[FETCH] Selesai! {new_records} records baru. "
        f"Durasi: {fetch_duration:.1f}s, Requests: {total_requests}"
    )

    return {
        "status": "done",
        "new_records": new_records,
        "message": f"Sinkronisasi selesai. {new_records} data baru ditambahkan.",
    }


def validate_dataset_integrity(df: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
    """
    Validasi integritas dataset sebelum training.

    Cek:
    - Duplicate Timestamp
    - Missing Timestamp (gap dalam interval 1 jam)
    - Gap Data (lebih dari 1.5 jam antar record)
    - Missing Values per kolom
    - Empty Records (baris tanpa data sensor)

    Args:
        df: DataFrame untuk divalidasi. Jika None, load dari dataset lokal.

    Returns:
        dict: detail masalah yang ditemukan, status, warnings
    """
    if df is None:
        df = load_local_dataset()

    if df is None or len(df) == 0:
        return {
            "valid": False,
            "status": "no_data",
            "message": "Tidak ada dataset untuk divalidasi.",
            "issues": [],
            "summary": {},
        }

    issues = []
    df = df.copy()
    df["datetime"] = pd.to_datetime(df["datetime"])
    df = df.sort_values("datetime").reset_index(drop=True)

    # 1. Duplicate Timestamp
    dup_mask = df.duplicated(subset=["datetime"], keep=False)
    dup_count = int(dup_mask.sum())
    if dup_count > 0:
        dup_times = df[dup_mask]["datetime"].dt.strftime("%Y-%m-%d %H:%M").unique().tolist()
        issues.append({
            "type": "duplicate_timestamp",
            "severity": "warning",
            "count": dup_count,
            "message": f"Ditemukan {dup_count} baris dengan timestamp duplikat.",
            "samples": dup_times[:10],
        })

    # 2. Missing Timestamp / Gap Data
    if len(df) > 1:
        time_diffs = df["datetime"].diff()
        expected_interval = pd.Timedelta(hours=1)

        # Gap > 1.5 jam
        gap_mask = time_diffs > expected_interval * 1.5
        gap_count = int(gap_mask.sum())

        if gap_count > 0:
            gap_details = []
            for i in range(1, len(df)):
                if gap_mask.iloc[i]:
                    gap_start = df["datetime"].iloc[i - 1]
                    gap_end = df["datetime"].iloc[i]
                    gap_hours = time_diffs.iloc[i].total_seconds() / 3600
                    gap_details.append({
                        "start": format_wib(to_wib(gap_start)),
                        "end": format_wib(to_wib(gap_end)),
                        "gap_hours": round(gap_hours, 1),
                    })

            issues.append({
                "type": "gap_data",
                "severity": "warning",
                "count": gap_count,
                "message": f"Ditemukan {gap_count} gap data (lebih dari 1.5 jam).",
                "details": gap_details[:20],
            })

        # Hitung total missing timestamps (hourly yang seharusnya ada)
        total_expected = int((df["datetime"].max() - df["datetime"].min()).total_seconds() / 3600) + 1
        actual_unique = df["datetime"].nunique()
        missing_timestamps = total_expected - actual_unique
        if missing_timestamps > 0:
            issues.append({
                "type": "missing_timestamp",
                "severity": "info",
                "count": missing_timestamps,
                "message": f"Terdapat {missing_timestamps} timestamp yang hilang dari total {total_expected} yang diharapkan.",
            })

    # 3. Missing Values per kolom
    missing_per_col = {}
    total_missing = 0
    for col in PARAMETERS:
        if col in df.columns:
            n_miss = int(df[col].isna().sum())
            pct = round(n_miss / len(df) * 100, 2) if len(df) > 0 else 0
            missing_per_col[col] = {"count": n_miss, "percentage": pct}
            total_missing += n_miss

    if total_missing > 0:
        issues.append({
            "type": "missing_values",
            "severity": "warning",
            "count": total_missing,
            "message": f"Ditemukan {total_missing} missing values di {sum(1 for v in missing_per_col.values() if v['count'] > 0)} kolom.",
            "details": missing_per_col,
        })

    # 4. Empty Records (baris tanpa data sensor sama sekali)
    sensor_cols = [c for c in PARAMETERS if c in df.columns]
    if sensor_cols:
        empty_mask = df[sensor_cols].isna().all(axis=1)
        empty_count = int(empty_mask.sum())
        if empty_count > 0:
            issues.append({
                "type": "empty_record",
                "severity": "warning",
                "count": empty_count,
                "message": f"Ditemukan {empty_count} baris tanpa data sensor sama sekali.",
            })

    # Summary
    has_warnings = any(i["severity"] == "warning" for i in issues)
    summary = {
        "total_records": len(df),
        "total_issues": len(issues),
        "has_warnings": has_warnings,
        "duplicate_timestamps": dup_count,
        "gap_count": gap_count if len(df) > 1 else 0,
        "missing_values_total": total_missing,
        "empty_records": empty_count if sensor_cols else 0,
        "date_range": {
            "start": format_wib(to_wib(df["datetime"].min())),
            "end": format_wib(to_wib(df["datetime"].max())),
        },
    }

    return {
        "valid": not has_warnings or total_missing < len(df) * len(sensor_cols) * 0.5,
        "status": "warning" if has_warnings else "ok",
        "message": (
            f"Dataset memiliki {len(issues)} masalah yang perlu diperhatikan."
            if has_warnings
            else "Dataset dalam kondisi baik."
        ),
        "issues": issues,
        "summary": summary,
        "missing_per_column": missing_per_col,
    }


def get_sync_status() -> Dict[str, Any]:
    """
    Dapatkan status sinkronisasi antara dataset lokal dan OpenAQ.

    Returns:
        dict: last_local, current_time, gap, status_message
    """
    last_ts = get_latest_local_timestamp()
    current = now_wib()

    if last_ts is None:
        return {
            "last_local": None,
            "last_local_display": "Belum ada data",
            "current_time": current.isoformat(),
            "current_time_display": format_wib(current),
            "gap_hours": None,
            "gap_display": "N/A",
            "status": "no_data",
            "status_message": "Belum ada dataset lokal. Jalankan sinkronisasi terlebih dahulu.",
            "status_color": "red",
        }

    gap = current - last_ts
    gap_hours = gap.total_seconds() / 3600

    if gap_hours <= 2:
        status = "synced"
        status_message = "Dataset Sinkron"
        status_color = "green"
    elif gap_hours <= 6:
        status = "slightly_behind"
        status_message = f"Dataset tertinggal {gap_hours:.0f} jam"
        status_color = "yellow"
    elif gap_hours <= 24:
        status = "behind"
        status_message = f"Dataset tertinggal {gap_hours:.0f} jam"
        status_color = "orange"
    else:
        days_behind = gap_hours / 24
        status = "far_behind"
        status_message = f"Dataset tertinggal {days_behind:.1f} hari"
        status_color = "red"

    return {
        "last_local": last_ts.isoformat(),
        "last_local_display": format_wib(last_ts),
        "current_time": current.isoformat(),
        "current_time_display": format_wib(current),
        "gap_hours": round(gap_hours, 1),
        "gap_display": (
            f"{gap_hours:.0f} jam" if gap_hours < 48
            else f"{gap_hours / 24:.1f} hari"
        ),
        "status": status,
        "status_message": status_message,
        "status_color": status_color,
    }
