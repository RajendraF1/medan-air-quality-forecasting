"""
Timezone Utilities — WIB (UTC+7)
==================================
Helper functions untuk memastikan seluruh sistem
menggunakan waktu WIB secara konsisten.

Digunakan di: fetching, preprocessing, training,
database, logging, prediksi, dashboard.
"""

from datetime import datetime, timedelta, timezone

# WIB = UTC+7
TIMEZONE_WIB = timezone(timedelta(hours=7))


def now_wib() -> datetime:
    """Dapatkan waktu saat ini dalam WIB (UTC+7)."""
    return datetime.now(TIMEZONE_WIB)


def to_wib(dt: datetime) -> datetime:
    """Konversi datetime ke WIB."""
    if dt.tzinfo is None:
        # Asumsikan sudah WIB jika naive
        return dt.replace(tzinfo=TIMEZONE_WIB)
    return dt.astimezone(TIMEZONE_WIB)


def format_wib(dt: datetime) -> str:
    """
    Format datetime ke string WIB yang mudah dibaca.

    Contoh: "31 Mei 2026 13:00 WIB"
    """
    if dt is None:
        return "-"

    wib_dt = to_wib(dt)

    bulan = [
        "", "Januari", "Februari", "Maret", "April", "Mei", "Juni",
        "Juli", "Agustus", "September", "Oktober", "November", "Desember"
    ]

    return (
        f"{wib_dt.day} {bulan[wib_dt.month]} {wib_dt.year} "
        f"{wib_dt.strftime('%H:%M')} WIB"
    )


def format_wib_short(dt: datetime) -> str:
    """
    Format datetime ke string WIB ringkas.

    Contoh: "31 Mei 2026 13:00"
    """
    if dt is None:
        return "-"

    wib_dt = to_wib(dt)

    bulan = [
        "", "Jan", "Feb", "Mar", "Apr", "Mei", "Jun",
        "Jul", "Ags", "Sep", "Okt", "Nov", "Des"
    ]

    return (
        f"{wib_dt.day} {bulan[wib_dt.month]} {wib_dt.year} "
        f"{wib_dt.strftime('%H:%M')}"
    )


def format_wib_iso(dt: datetime) -> str:
    """Format datetime ke ISO 8601 string dengan offset WIB."""
    if dt is None:
        return None
    wib_dt = to_wib(dt)
    return wib_dt.isoformat()
