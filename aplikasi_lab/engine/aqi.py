"""
AQI Calculator — US EPA 2024
================================
Konversi PM2.5 ke Air Quality Index menggunakan standar US EPA
yang berlaku sejak 6 Mei 2024.

Juga menyediakan rekomendasi kesehatan profesional.
"""

from typing import Dict, Any, List


# ==============================================================================
# US EPA PM2.5 Breakpoints (Updated May 6, 2024)
# ==============================================================================
# Format: (C_low, C_high, I_low, I_high, category, color, health_message)
PM25_BREAKPOINTS = [
    (0.0, 9.0, 0, 50, "Baik", "#00e400",
     "Kualitas udara baik. Tidak ada risiko kesehatan."),
    (9.1, 35.4, 51, 100, "Sedang", "#ffff00",
     "Kualitas udara dapat diterima. Kelompok sensitif mungkin mengalami gejala ringan."),
    (35.5, 55.4, 101, 150, "Tidak Sehat bagi Kelompok Sensitif", "#ff7e00",
     "Kelompok sensitif (asma, lansia, anak-anak) sebaiknya kurangi aktivitas luar."),
    (55.5, 125.4, 151, 200, "Tidak Sehat", "#ff0000",
     "Semua orang mulai merasakan dampak kesehatan. Kurangi aktivitas luar ruangan."),
    (125.5, 225.4, 201, 300, "Sangat Tidak Sehat", "#8f3f97",
     "Peringatan kesehatan: semua orang berisiko. Hindari aktivitas luar ruangan."),
    (225.5, 325.4, 301, 500, "Berbahaya", "#7e0023",
     "BAHAYA! Kondisi darurat kesehatan. Semua orang terdampak serius."),
]


def calculate_aqi(pm25_value: float) -> Dict[str, Any]:
    """
    Hitung AQI dari konsentrasi PM2.5 menggunakan standar US EPA 2024.

    Formula:
      AQI = ((I_high - I_low) / (C_high - C_low)) * (Cp - C_low) + I_low
    """
    if pm25_value is None or pm25_value < 0:
        return {
            "aqi": 0,
            "pm25_truncated": 0.0,
            "category": "Invalid",
            "color": "#808080",
            "health_message": "Nilai PM2.5 tidak valid.",
            "description": "Data error"
        }

    # Truncate ke 1 desimal
    cp = float(int(pm25_value * 10)) / 10.0

    for c_low, c_high, i_low, i_high, category, color, health_msg in PM25_BREAKPOINTS:
        if c_low <= cp <= c_high:
            aqi = ((i_high - i_low) / (c_high - c_low)) * (cp - c_low) + i_low
            aqi = round(aqi)

            return {
                "aqi": aqi,
                "pm25_truncated": cp,
                "category": category,
                "color": color,
                "health_message": health_msg,
                "description": f"PM2.5: {cp} µg/m³ → AQI: {aqi} ({category})"
            }

    # Beyond scale
    if cp > 325.4:
        return {
            "aqi": 500,
            "pm25_truncated": cp,
            "category": "Berbahaya",
            "color": "#7e0023",
            "health_message": "BAHAYA MAKSIMAL! Kondisi sangat berbahaya.",
            "description": f"PM2.5: {cp} µg/m³ → AQI: 500+ (Beyond Scale)"
        }

    return {
        "aqi": 0,
        "pm25_truncated": cp,
        "category": "Unknown",
        "color": "#808080",
        "health_message": "Tidak dapat menghitung AQI.",
        "description": "Error"
    }


def get_aqi_category_info(aqi_value: int) -> Dict[str, Any]:
    """Dapatkan info kategori berdasarkan nilai AQI."""
    categories = [
        (0, 50, "Baik", "#00e400"),
        (51, 100, "Sedang", "#ffff00"),
        (101, 150, "Tidak Sehat bagi Kelompok Sensitif", "#ff7e00"),
        (151, 200, "Tidak Sehat", "#ff0000"),
        (201, 300, "Sangat Tidak Sehat", "#8f3f97"),
        (301, 500, "Berbahaya", "#7e0023"),
    ]

    for low, high, name, color in categories:
        if low <= aqi_value <= high:
            return {"category": name, "color": color, "range": f"{low}-{high}"}

    return {"category": "Beyond AQI", "color": "#7e0023", "range": "500+"}


# ==============================================================================
# Rekomendasi Kesehatan Profesional
# ==============================================================================
def get_health_recommendations(aqi: int, pm25: float) -> Dict[str, Any]:
    """
    Rekomendasi kesehatan berdasarkan AQI dan PM2.5.

    Returns:
        dict: risk_level, general, sensitive_groups, outdoor_activity,
              indoor_tips, health_effects, protective_measures
    """
    if aqi <= 50:
        return {
            "risk_level": "Rendah",
            "risk_color": "#00e400",
            "risk_icon": "✅",
            "general": "Kualitas udara sangat baik. Kondisi ideal untuk semua aktivitas.",
            "sensitive_groups": "Tidak ada peringatan khusus untuk kelompok sensitif.",
            "outdoor_activity": "Semua aktivitas luar ruangan aman dilakukan.",
            "indoor_tips": "Ventilasi alami dapat digunakan dengan bebas.",
            "health_effects": "Tidak ada risiko kesehatan yang teridentifikasi pada tingkat ini.",
            "protective_measures": [
                "Tidak diperlukan tindakan perlindungan khusus",
                "Nikmati aktivitas luar ruangan dengan bebas",
                "Waktu yang tepat untuk berolahraga di luar ruangan",
            ],
            "activity_guide": {
                "olahraga_outdoor": {"status": "Aman", "color": "#00e400"},
                "aktivitas_anak": {"status": "Aman", "color": "#00e400"},
                "lansia": {"status": "Aman", "color": "#00e400"},
                "ibu_hamil": {"status": "Aman", "color": "#00e400"},
            }
        }
    elif aqi <= 100:
        return {
            "risk_level": "Sedang",
            "risk_color": "#ffff00",
            "risk_icon": "⚠️",
            "general": "Kualitas udara cukup baik untuk sebagian besar orang. Kelompok sensitif perlu waspada.",
            "sensitive_groups": "Penderita asma dan gangguan pernapasan kronis mungkin mengalami gejala ringan. Pertimbangkan untuk mengurangi aktivitas berat di luar ruangan.",
            "outdoor_activity": "Aktivitas luar ruangan normal masih diperbolehkan. Kurangi aktivitas berat berkepanjangan jika mengalami gejala.",
            "indoor_tips": "Pertimbangkan untuk menutup jendela jika berada di dekat sumber polusi.",
            "health_effects": "Kemungkinan iritasi ringan pada saluran pernapasan bagi kelompok sensitif.",
            "protective_measures": [
                "Kelompok sensitif sebaiknya membatasi aktivitas berat di luar",
                "Perhatikan gejala seperti batuk atau sesak napas",
                "Sediakan obat pernapasan jika memiliki riwayat asma",
                "Tetap terhidrasi dengan baik",
            ],
            "activity_guide": {
                "olahraga_outdoor": {"status": "Hati-hati", "color": "#ffff00"},
                "aktivitas_anak": {"status": "Batasi durasi", "color": "#ffff00"},
                "lansia": {"status": "Waspada", "color": "#ffff00"},
                "ibu_hamil": {"status": "Batasi paparan", "color": "#ffff00"},
            }
        }
    elif aqi <= 150:
        return {
            "risk_level": "Tinggi (Sensitif)",
            "risk_color": "#ff7e00",
            "risk_icon": "🟠",
            "general": "Kelompok sensitif berisiko mengalami dampak kesehatan. Masyarakat umum kemungkinan tidak terpengaruh.",
            "sensitive_groups": "Anak-anak, lansia, penderita asma, dan penyakit jantung/paru HARUS mengurangi aktivitas luar ruangan berkepanjangan.",
            "outdoor_activity": "Kurangi aktivitas fisik berat di luar ruangan, terutama jika merasakan gejala.",
            "indoor_tips": "Tutup jendela dan gunakan air purifier jika tersedia. Hindari menyalakan lilin atau memasak dengan banyak asap.",
            "health_effects": "Peningkatan risiko gejala pernapasan pada kelompok sensitif. Kemungkinan iritasi mata dan tenggorokan.",
            "protective_measures": [
                "Kelompok sensitif harus mengurangi waktu di luar ruangan",
                "Gunakan masker N95/KN95 saat keluar",
                "Tutup jendela dan nyalakan air purifier",
                "Hindari olahraga intensif di luar ruangan",
                "Segera konsultasi dokter jika gejala memburuk",
            ],
            "activity_guide": {
                "olahraga_outdoor": {"status": "Kurangi", "color": "#ff7e00"},
                "aktivitas_anak": {"status": "Dalam ruangan", "color": "#ff7e00"},
                "lansia": {"status": "Hindari keluar", "color": "#ff7e00"},
                "ibu_hamil": {"status": "Dalam ruangan", "color": "#ff7e00"},
            }
        }
    elif aqi <= 200:
        return {
            "risk_level": "Tinggi",
            "risk_color": "#ff0000",
            "risk_icon": "🔴",
            "general": "Semua orang mulai merasakan dampak kesehatan. Kelompok sensitif mengalami efek yang lebih serius.",
            "sensitive_groups": "Kelompok sensitif HARUS menghindari semua aktivitas luar ruangan. Pertimbangkan untuk tetap di dalam ruangan.",
            "outdoor_activity": "Kurangi aktivitas luar ruangan secara signifikan. Hindari olahraga di luar.",
            "indoor_tips": "Pastikan ruangan tertutup rapat. Gunakan air purifier. Hindari membuka jendela.",
            "health_effects": "Peningkatan risiko gangguan pernapasan, kardiovaskular, dan iritasi pada semua kelompok usia.",
            "protective_measures": [
                "Batasi waktu di luar ruangan seminimal mungkin",
                "WAJIB gunakan masker N95/KN95 saat keluar",
                "Semua aktivitas fisik sebaiknya di dalam ruangan",
                "Kelompok sensitif tidak boleh keluar rumah",
                "Sediakan obat-obatan darurat",
                "Hubungi layanan kesehatan jika mengalami gejala berat",
            ],
            "activity_guide": {
                "olahraga_outdoor": {"status": "Hindari", "color": "#ff0000"},
                "aktivitas_anak": {"status": "Dilarang keluar", "color": "#ff0000"},
                "lansia": {"status": "Tetap di dalam", "color": "#ff0000"},
                "ibu_hamil": {"status": "Tetap di dalam", "color": "#ff0000"},
            }
        }
    elif aqi <= 300:
        return {
            "risk_level": "Sangat Tinggi",
            "risk_color": "#8f3f97",
            "risk_icon": "🟣",
            "general": "PERINGATAN KESEHATAN: Semua orang berisiko tinggi. Kondisi udara sangat berbahaya.",
            "sensitive_groups": "Kelompok sensitif dalam kondisi DARURAT. Jangan keluar rumah dalam kondisi apapun.",
            "outdoor_activity": "HINDARI semua aktivitas luar ruangan. Semua kegiatan harus dilakukan di dalam ruangan.",
            "indoor_tips": "Segel ruangan dari udara luar. Gunakan air purifier pada pengaturan maksimal. Pertimbangkan evakuasi jika memungkinkan.",
            "health_effects": "Risiko serius terhadap sistem pernapasan dan kardiovaskular. Potensi efek kesehatan jangka panjang.",
            "protective_measures": [
                "JANGAN keluar rumah kecuali keadaan darurat",
                "Masker N95/KN95 WAJIB jika harus keluar",
                "Segel pintu dan jendela dari udara luar",
                "Air purifier HARUS digunakan secara terus-menerus",
                "Siapkan rencana evakuasi",
                "Segera ke fasilitas kesehatan jika mengalami sesak napas",
            ],
            "activity_guide": {
                "olahraga_outdoor": {"status": "DILARANG", "color": "#8f3f97"},
                "aktivitas_anak": {"status": "DILARANG keluar", "color": "#8f3f97"},
                "lansia": {"status": "DARURAT", "color": "#8f3f97"},
                "ibu_hamil": {"status": "DARURAT", "color": "#8f3f97"},
            }
        }
    else:
        return {
            "risk_level": "Berbahaya",
            "risk_color": "#7e0023",
            "risk_icon": "☠️",
            "general": "KONDISI DARURAT KESEHATAN. Seluruh populasi terdampak serius. Kualitas udara sangat berbahaya.",
            "sensitive_groups": "SEMUA orang dalam kondisi DARURAT. Evakuasi mungkin diperlukan.",
            "outdoor_activity": "DILARANG keras berada di luar ruangan.",
            "indoor_tips": "Segel seluruh ruangan. Gunakan air purifier. Pertimbangkan evakuasi ke area dengan kualitas udara lebih baik.",
            "health_effects": "Dampak kesehatan serius dan segera pada seluruh populasi. Risiko kematian meningkat signifikan.",
            "protective_measures": [
                "KONDISI DARURAT — pertimbangkan evakuasi",
                "DILARANG keras keluar rumah",
                "Masker N95 tidak cukup melindungi pada level ini",
                "Hubungi layanan darurat jika mengalami gejala",
                "Segel seluruh bukaan udara di rumah",
                "Siapkan pertolongan pertama dan obat-obatan darurat",
            ],
            "activity_guide": {
                "olahraga_outdoor": {"status": "DARURAT", "color": "#7e0023"},
                "aktivitas_anak": {"status": "DARURAT", "color": "#7e0023"},
                "lansia": {"status": "DARURAT", "color": "#7e0023"},
                "ibu_hamil": {"status": "DARURAT", "color": "#7e0023"},
            }
        }
