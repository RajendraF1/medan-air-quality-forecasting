# AirSense - Prediksi Kualitas Udara Real-time

AirSense adalah sistem pemantauan dan prediksi kualitas udara PM2.5 berbasis FastAPI dan XGBoost. Nilai PM2.5 dikonversi menjadi AQI menggunakan standar US EPA 2024.

Data utama disimpan sebagai CSV bulanan di `core/data/kualitas_udara/`. Pengambilan data dari OpenAQ tidak berjalan otomatis. Admin menjalankan tombol collect/sync data, lalu sistem mengecek timestamp terakhir di CSV lokal dan hanya mengambil data baru yang tersedia dari OpenAQ. Jika OpenAQ belum menyediakan data baru, aplikasi tetap memakai dataset lokal yang sudah ada.

## Struktur Project

- `app.py`: Entry point FastAPI.
- `core/config.py`: Konfigurasi path, OpenAQ, timezone WIB, database, model, dan lokasi stasiun.
- `core/database/`: Koneksi SQLAlchemy dan model ORM.
- `aplikasi_utama/`: API dan frontend SPA untuk pengguna utama.
- `aplikasi_lab/engine/`: Logika fetch data, preprocessing, training, prediksi, dan AQI.
- `aplikasi_lab/dashboard_api.py`: API dashboard evaluasi model untuk admin/lab.
- `core/data/`: Dataset lokal, database SQLite, dan file fitur training.
- `core/models/`: Model aktif dan training report.

## Fitur Utama

1. **Dashboard Utama**: Status data terbaru, PM2.5, dan indikator AQI.
2. **Area Paparan**: Peta Leaflet untuk lokasi stasiun pemantau.
3. **Prediksi AI**: Recursive forecasting PM2.5 untuk 24 jam ke depan.
4. **Mitigasi Kesehatan**: Rekomendasi berdasarkan kategori AQI.
5. **Info Model**: Metrik performa model dan fitur penting.
6. **Dashboard Evaluasi**: Sinkronisasi data manual, validasi dataset, preprocessing, training, dan evaluasi eksperimen.

## Prasyarat

- Python 3.10+
- Dataset historis di `core/data/kualitas_udara/*.csv`
- Model terlatih di `core/models/pm25_xgboost_model.joblib` atau training ulang melalui dashboard evaluasi

## Setup & Instalasi

1. Buat virtual environment:

   ```bash
   python -m venv venv
   venv\Scripts\activate
   ```

2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Konfigurasi environment variables di file `.env` pada root project:

   ```env
   OPENAQ_API_KEY=your_api_key_here
   DB_USE_MYSQL=false
   ADMIN_USERNAME=admin
   ADMIN_PASSWORD=admin123
   ```

## Menjalankan Aplikasi

```bash
uvicorn app:app --reload
```

Atau:

```bash
python -m uvicorn app:app
```

- Aplikasi utama: `http://localhost:8000/`
- Dashboard evaluasi: `http://localhost:8000/eval/`
- Swagger UI: `http://localhost:8000/docs`

## Catatan Penting

- Collect/sync data bersifat manual melalui tombol atau endpoint `/api/sync` dan `/eval/api/sync-data`.
- Jika tidak ada data baru dari OpenAQ, sistem tetap menampilkan data dan prediksi dari dataset lokal terakhir.
- Prediksi berjalan 24 jam ke depan dengan pendekatan recursive forecasting.
- Jangan menghapus file di `aplikasi_lab/templates/`, `core/data/`, dan `core/models/` jika masih dibutuhkan untuk dashboard dan model aktif.
