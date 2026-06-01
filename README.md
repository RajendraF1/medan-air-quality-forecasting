# ARISQA - Prediksi Kualitas Udara Real-time

![ARISQA](https://img.shields.io/badge/ARISQA-Production-050505?style=for-the-badge&logo=wind&logoColor=white) ![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=for-the-badge&logo=FastAPI&logoColor=white) ![XGBoost](https://img.shields.io/badge/XGBoost-191A1B?style=for-the-badge) ![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)

ARISQA adalah sistem pemantauan dan prediksi kualitas udara (PM2.5) berbasis **FastAPI** dan **XGBoost**. Nilai PM2.5 yang diproses dan diprediksi oleh sistem akan dikonversi menjadi Air Quality Index (AQI) menggunakan standar US EPA 2024. Aplikasi ini berfokus pada area Stasiun USU Medan.

Data utama disimpan sebagai CSV secara lokal (`core/data/kualitas_udara/`). Pengambilan data dari OpenAQ bersifat **manual/on-demand**. Sistem akan memeriksa *timestamp* data lokal terakhir dan hanya mengambil data terbaru dari OpenAQ. Jika OpenAQ belum menyediakan data baru, aplikasi tetap beroperasi secara normal menggunakan dataset lokal terakhir.

---

## 🏗️ Struktur Project

Arsitektur aplikasi dibagi menjadi beberapa komponen utama untuk memisahkan tanggung jawab (Separation of Concerns):

- `app.py` : Entry point untuk menjalankan server FastAPI (melayani Aplikasi Utama dan Lab).
- `core/` : Pusat konfigurasi dan sumber data.
  - `config.py` : Konfigurasi path, kredensial OpenAQ, timezone WIB, konfigurasi database, hyperparameter model, dan detail lokasi stasiun.
  - `database/` : Koneksi SQLAlchemy dan model ORM.
  - `data/` : Dataset lokal (CSV historis bulanan) dan database SQLite.
  - `models/` : Tempat penyimpanan model aktif (`.joblib`) dan laporan hasil *training*.
- `aplikasi_utama/` : API publik dan antarmuka web (SPA) untuk pengguna umum (dashboard utama).
- `aplikasi_lab/` : Area tersembunyi khusus admin (Laboratorium).
  - `engine/` : Logika *back-end* berat (fetch data, preprocessing, iterasi *training*, *recursive forecasting*, kalkulasi AQI).
  - `dashboard_api.py`: Endpoint API khusus untuk dashboard evaluasi model admin.

---

## ✨ Fitur Utama

Aplikasi utama (*frontend*) memiliki berbagai fitur interaktif berbasis *glassmorphism* modern:

1. **Dashboard Kualitas Udara**: Status pemantauan *real-time*, parameter PM2.5, Suhu, Kelembaban, dan indikator AQI visual.
2. **Area Paparan (Peta Interaktif)**: Pemetaan visual jangkauan paparan polusi di radius stasiun pemantau menggunakan Leaflet.js.
3. **Prediksi AI (24 Jam)**: Tren *recursive forecasting* polusi PM2.5 untuk 24 jam ke depan, ditampilkan dalam grafik interaktif Chart.js.
4. **Sistem Peringatan Dini Global**: Notifikasi *banner* otomatis yang terintegrasi di seluruh halaman aplikasi. Sistem akan mendeteksi jika prediksi 24 jam ke depan menghasilkan AQI > 100 ("Tidak Sehat") dan memperingatkan pengguna secara proaktif. Tersedia juga fitur simulasi demo.
5. **Mitigasi Kesehatan**: Rekomendasi aktivitas dan tindakan perlindungan dinamis berdasarkan tingkat keparahan kategori AQI.
6. **Info Model**: Rangkuman metrik performa model (MAE, R² Score) dan fitur-fitur yang digunakan oleh AI.
7. **Dashboard Evaluasi (Lab Admin)**: Ruang kerja admin untuk melakukan sinkronisasi data manual ke OpenAQ, evaluasi akurasi, dan *re-training* model.

---

## ⚙️ Prasyarat & Instalasi

### Prasyarat
- Python 3.10 atau versi lebih baru.
- Dataset historis telah tersedia di `core/data/kualitas_udara/`.
- Model terlatih telah tersedia di `core/models/pm25_xgboost_model.joblib` (atau lakukan *training* awal melalui dashboard evaluasi).

### Langkah Instalasi

1. **Buat Virtual Environment**
   ```bash
   python -m venv venv
   # Di Windows
   venv\Scripts\activate
   # Di Linux/Mac
   source venv/bin/activate
   ```

2. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Konfigurasi Environment Variables**
   Buat file `.env` di *root directory* (sejajar dengan `app.py`) dan isi dengan kredensial Anda:
   ```env
   OPENAQ_API_KEY=your_api_key_here
   DB_USE_MYSQL=false
   ADMIN_USERNAME=admin
   ADMIN_PASSWORD=admin123
   ```

---

## 🚀 Menjalankan Aplikasi

Jalankan server menggunakan uvicorn:

```bash
uvicorn app:app --reload
```
*Atau menggunakan Python secara langsung:*
```bash
python -m uvicorn app:app
```

**Akses URL:**
- **Aplikasi Utama (Publik)**: [http://localhost:8000/](http://localhost:8000/)
- **Dashboard Evaluasi (Admin)**: [http://localhost:8000/eval/](http://localhost:8000/eval/)
- **Dokumentasi API (Swagger)**: [http://localhost:8000/docs](http://localhost:8000/docs)

---

## ⚠️ Catatan Penting

- **Mode Offline/Lokal**: Sinkronisasi data OpenAQ murni **bersifat manual**. Anda bisa menekan tombol sinkronisasi di Dashboard Lab atau memanggil endpoint `/api/sync`. Jika tidak ada data baru (atau API OpenAQ mati), sistem 100% masih berjalan menggunakan model dan dataset lokal terakhir.
- **Data Prediksi**: Prediksi dijalankan hingga 24 jam ke depan. Karena PM2.5 sangat fluktuatif, *recursive forecasting* digunakan untuk mendapatkan aproksimasi tren per jamnya.
- **Keamanan Direktori**: Jangan pernah menghapus folder `core/data/` dan `core/models/`. Kehilangan dataset atau file *joblib* akan mengharuskan sistem untuk mengunduh ulang data dan melatih model dari awal.
- **Pengujian Peringatan Dini**: Terdapat tombol tersembunyi "Toggle Peringatan Demo" pada bagian footer halaman utama bagi Anda yang ingin menguji tampilan UI peringatan dini.
