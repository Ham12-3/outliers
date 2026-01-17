"""Detection router for running anomaly detection."""
from datetime import datetime
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.orm import Session

from ..database import get_db
from ..services.detection_pipeline import run_detection_pipeline, run_flexible_detection

router = APIRouter()


class DetectionRunResponse(BaseModel):
    """Response model for detection run."""

    message: str
    status: str
    features_created: int = 0
    tukey_outliers: int = 0
    isolation_forest_outliers: int = 0
    incidents_created: int = 0
    incidents_updated: int = 0


# Global state for tracking detection runs
_detection_status = {
    "running": False,
    "last_run": None,
    "last_result": None,
}


def get_detection_status():
    """Get current detection status."""
    return _detection_status


@router.post("/run", response_model=DetectionRunResponse)
async def run_detection(
    background: bool = False,
    background_tasks: BackgroundTasks = None,
    db: Session = Depends(get_db),
):
    """
    Run the full detection pipeline.

    This includes:
    1. Computing daily features from raw metrics
    2. Running Tukey IQR outlier detection
    3. Running Isolation Forest anomaly detection
    4. Creating/updating incidents from detection results

    Parameters:
    - background: Run in background (returns immediately)
    """
    global _detection_status

    if _detection_status["running"]:
        raise HTTPException(
            status_code=409,
            detail="Detection is already running. Please wait for it to complete.",
        )

    if background and background_tasks:
        _detection_status["running"] = True
        background_tasks.add_task(_run_detection_background, db)
        return DetectionRunResponse(
            message="Detection started in background",
            status="running",
        )

    # Run synchronously
    _detection_status["running"] = True
    try:
        result = run_detection_pipeline(db)
        _detection_status["last_run"] = datetime.utcnow().isoformat()
        _detection_status["last_result"] = result
        return DetectionRunResponse(
            message="Detection completed successfully",
            status="completed",
            **result,
        )
    finally:
        _detection_status["running"] = False


def _run_detection_background(db: Session):
    """Run detection in background."""
    global _detection_status
    try:
        result = run_detection_pipeline(db)
        _detection_status["last_run"] = datetime.utcnow().isoformat()
        _detection_status["last_result"] = result
    except Exception as e:
        _detection_status["last_result"] = {"error": str(e)}
    finally:
        _detection_status["running"] = False


@router.get("/status")
async def detection_status():
    """Get the current detection status."""
    return {
        "running": _detection_status["running"],
        "last_run": _detection_status["last_run"],
        "last_result": _detection_status["last_result"],
    }


class FlexibleDetectionResponse(BaseModel):
    """Response model for flexible dataset detection."""
    dataset_id: int
    dataset_name: Optional[str] = None
    rows_analyzed: int = 0
    metrics_analyzed: List[str] = []
    total_outliers: int = 0
    incidents_created: int = 0
    error: Optional[str] = None
    message: Optional[str] = None


@router.post("/run/{dataset_id}", response_model=FlexibleDetectionResponse)
async def run_dataset_detection(
    dataset_id: int,
    db: Session = Depends(get_db),
):
    """
    Run anomaly detection on a specific flexible dataset.

    This will:
    1. Load the dataset and its configured schema
    2. Run Tukey IQR detection on each metric column
    3. Run Isolation Forest detection if multiple metrics are configured
    4. Create incidents for detected anomalies

    Parameters:
    - dataset_id: ID of the dataset to analyze
    """
    result = run_flexible_detection(db, dataset_id)

    if "error" in result:
        return FlexibleDetectionResponse(
            dataset_id=dataset_id,
            error=result.get("error"),
            message=result.get("message"),
        )

    return FlexibleDetectionResponse(
        dataset_id=dataset_id,
        dataset_name=result.get("dataset_name"),
        rows_analyzed=result.get("rows_analyzed", 0),
        metrics_analyzed=result.get("metrics_analyzed", []),
        total_outliers=result.get("total_outliers", 0),
        incidents_created=result.get("incidents_created", 0),
    )
