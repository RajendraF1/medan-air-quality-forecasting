"""
ARISQA — Aplikasi Utama (Production)
=========================================
Main FastAPI application.
- Mount static files & templates
- Include API router (production)
- Include Eval router (legacy dashboard)
- Manual collect/sync data dari OpenAQ
"""

import contextlib
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.middleware.cors import CORSMiddleware

from core.config import STATIC_DIR, TEMPLATES_DIR
from core.database.connection import init_db
from aplikasi_lab.engine.training import load_model

# Routers
from aplikasi_utama.api import api_router
from aplikasi_lab.dashboard_api import eval_router, auth_router

# ==============================================================================
# Lifecycle & App Init
# ==============================================================================
@contextlib.asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    print("[APP] Starting up ARISQA Production...")
    
    # Init Database
    init_db()
    
    # Coba load model awal (hanya warning jika gagal, jangan crash)
    try:
        model, features = load_model()
        if model is None:
            print("[APP] Warning: Model belum tersedia. Akan ditraining di background.")
    except Exception as e:
        print(f"[APP] Error loading initial model: {e}")
    
    yield
    
    # Shutdown
    print("[APP] Shutting down...")

app = FastAPI(
    title="ARISQA API",
    description="Prediksi Kualitas Udara Real-time",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==============================================================================
# Mount Static & Routers
# ==============================================================================
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

app.include_router(api_router)
app.include_router(auth_router)
app.include_router(eval_router)

# ==============================================================================
# Frontend Routes
# ==============================================================================
@app.get("/", response_class=HTMLResponse)
async def serve_index():
    """Serve SPA Frontend."""
    index_path = TEMPLATES_DIR / "index.html"
    if not index_path.exists():
        return "<h1>Frontend sedang dibangun...</h1>"
    with open(index_path, "r", encoding="utf-8") as f:
        return HTMLResponse(content=f.read())
