"""Main FastAPI application."""
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .config import get_settings
from .routers import data, detect, incidents, metrics
from .scheduler import init_scheduler, shutdown_scheduler

settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan events."""
    # Startup
    if settings.scheduler_enabled:
        init_scheduler()
    yield
    # Shutdown
    shutdown_scheduler()


app = FastAPI(
    title="Inventory Anomaly Detection API",
    description="Detect and manage inventory and sales anomalies across stores and SKUs",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(data.router, prefix="/data", tags=["Data"])
app.include_router(detect.router, prefix="/detect", tags=["Detection"])
app.include_router(incidents.router, prefix="/incidents", tags=["Incidents"])
app.include_router(metrics.router, prefix="/metrics", tags=["Metrics"])


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "service": "inventory-anomaly-api"}


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "Inventory Anomaly Detection API",
        "docs": "/docs",
        "health": "/health",
    }
