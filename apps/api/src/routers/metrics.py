"""Metrics router for time series data."""
from datetime import date, timedelta
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel
from sqlalchemy import and_, case, func
from sqlalchemy.orm import Session

from ..database import get_db, RawDailyMetric, FeatureDaily, DetectionResult, Incident

router = APIRouter()


class TimeSeriesPoint(BaseModel):
    """A single point in a time series."""

    date: date
    value: float
    is_outlier: bool = False


class TimeSeriesResponse(BaseModel):
    """Response model for time series data."""

    store_id: str
    sku_id: str
    metric: str
    data: List[TimeSeriesPoint]


class BoxplotData(BaseModel):
    """Boxplot statistics for a metric."""

    metric: str
    min: float
    q1: float
    median: float
    q3: float
    max: float
    outliers: List[float]
    iqr: float
    lower_fence: float
    upper_fence: float


@router.get("/timeseries", response_model=List[TimeSeriesResponse])
async def get_timeseries(
    store_id: str = Query(..., description="Store ID"),
    sku_id: str = Query(..., description="SKU ID"),
    date_from: Optional[date] = Query(None, description="Start date"),
    date_to: Optional[date] = Query(None, description="End date"),
    db: Session = Depends(get_db),
):
    """
    Get time series data for a specific store and SKU.

    Returns sold, on_hand, and delta_on_hand series with outlier flags.
    """
    # Default date range: last 90 days
    if not date_to:
        date_to = date.today()
    if not date_from:
        date_from = date_to - timedelta(days=90)

    # Get raw metrics
    metrics = (
        db.query(RawDailyMetric)
        .filter(
            RawDailyMetric.store_id == store_id,
            RawDailyMetric.sku_id == sku_id,
            RawDailyMetric.date >= date_from,
            RawDailyMetric.date <= date_to,
        )
        .order_by(RawDailyMetric.date)
        .all()
    )

    if not metrics:
        raise HTTPException(status_code=404, detail="No data found for the specified store/SKU combination")

    # Get detection results for outlier flags
    outlier_dates = set()
    detection_results = (
        db.query(DetectionResult.date, DetectionResult.metric_name)
        .filter(
            DetectionResult.store_id == store_id,
            DetectionResult.sku_id == sku_id,
            DetectionResult.date >= date_from,
            DetectionResult.date <= date_to,
            DetectionResult.is_outlier == True,
        )
        .all()
    )

    outlier_map = {}
    for dr in detection_results:
        if dr.date not in outlier_map:
            outlier_map[dr.date] = set()
        if dr.metric_name:
            outlier_map[dr.date].add(dr.metric_name)

    # Build time series for each metric
    sold_series = []
    on_hand_series = []
    delta_on_hand_series = []

    prev_on_hand = None
    for m in metrics:
        sold_series.append(
            TimeSeriesPoint(
                date=m.date,
                value=m.sold,
                is_outlier="sold" in outlier_map.get(m.date, set()),
            )
        )
        on_hand_series.append(
            TimeSeriesPoint(
                date=m.date,
                value=m.on_hand,
                is_outlier="delta_on_hand" in outlier_map.get(m.date, set()),
            )
        )

        if prev_on_hand is not None:
            delta = m.on_hand - prev_on_hand
            delta_on_hand_series.append(
                TimeSeriesPoint(
                    date=m.date,
                    value=delta,
                    is_outlier="delta_on_hand" in outlier_map.get(m.date, set()),
                )
            )
        prev_on_hand = m.on_hand

    return [
        TimeSeriesResponse(store_id=store_id, sku_id=sku_id, metric="sold", data=sold_series),
        TimeSeriesResponse(store_id=store_id, sku_id=sku_id, metric="on_hand", data=on_hand_series),
        TimeSeriesResponse(store_id=store_id, sku_id=sku_id, metric="delta_on_hand", data=delta_on_hand_series),
    ]


@router.get("/boxplot", response_model=List[BoxplotData])
async def get_boxplot_data(
    store_id: str = Query(..., description="Store ID"),
    sku_id: str = Query(..., description="SKU ID"),
    date_from: Optional[date] = Query(None, description="Start date for baseline"),
    date_to: Optional[date] = Query(None, description="End date for baseline"),
    db: Session = Depends(get_db),
):
    """
    Get boxplot statistics for sold and delta_on_hand metrics.

    Uses the Tukey method with 1.5 * IQR for fence calculation.
    """
    if not date_to:
        date_to = date.today()
    if not date_from:
        date_from = date_to - timedelta(days=28)

    # Get raw metrics
    metrics = (
        db.query(RawDailyMetric)
        .filter(
            RawDailyMetric.store_id == store_id,
            RawDailyMetric.sku_id == sku_id,
            RawDailyMetric.date >= date_from,
            RawDailyMetric.date <= date_to,
        )
        .order_by(RawDailyMetric.date)
        .all()
    )

    if len(metrics) < 5:
        raise HTTPException(status_code=400, detail="Insufficient data for boxplot (need at least 5 data points)")

    import numpy as np

    # Calculate for sold
    sold_values = [m.sold for m in metrics]
    sold_boxplot = _calculate_boxplot(sold_values, "sold")

    # Calculate delta_on_hand
    delta_values = []
    for i in range(1, len(metrics)):
        delta_values.append(metrics[i].on_hand - metrics[i - 1].on_hand)

    delta_boxplot = _calculate_boxplot(delta_values, "delta_on_hand") if delta_values else None

    result = [sold_boxplot]
    if delta_boxplot:
        result.append(delta_boxplot)

    return result


def _calculate_boxplot(values: List[float], metric_name: str) -> BoxplotData:
    """Calculate boxplot statistics using Tukey's method."""
    import numpy as np

    arr = np.array(values)
    q1 = float(np.percentile(arr, 25))
    q3 = float(np.percentile(arr, 75))
    median = float(np.median(arr))
    iqr = q3 - q1
    lower_fence = q1 - 1.5 * iqr
    upper_fence = q3 + 1.5 * iqr

    # Identify outliers
    outliers = [float(v) for v in arr if v < lower_fence or v > upper_fence]

    return BoxplotData(
        metric=metric_name,
        min=float(np.min(arr)),
        q1=q1,
        median=median,
        q3=q3,
        max=float(np.max(arr)),
        outliers=outliers,
        iqr=iqr,
        lower_fence=lower_fence,
        upper_fence=upper_fence,
    )


@router.get("/stores")
async def list_stores(db: Session = Depends(get_db)):
    """List all unique store IDs."""
    stores = db.query(RawDailyMetric.store_id).distinct().order_by(RawDailyMetric.store_id).all()
    return {"stores": [s[0] for s in stores]}


@router.get("/skus")
async def list_skus(
    store_id: Optional[str] = Query(None, description="Filter SKUs by store"),
    db: Session = Depends(get_db),
):
    """List all unique SKU IDs, optionally filtered by store."""
    query = db.query(RawDailyMetric.sku_id).distinct()
    if store_id:
        query = query.filter(RawDailyMetric.store_id == store_id)
    skus = query.order_by(RawDailyMetric.sku_id).all()
    return {"skus": [s[0] for s in skus]}


@router.get("/incidents-over-time")
async def get_incidents_over_time(
    days: int = Query(30, ge=7, le=365, description="Number of days to look back"),
    db: Session = Depends(get_db),
):
    """Get incident counts grouped by date for charting."""
    from_date = date.today() - timedelta(days=days)

    results = (
        db.query(Incident.date, func.count(Incident.id))
        .filter(Incident.date >= from_date)
        .group_by(Incident.date)
        .order_by(Incident.date)
        .all()
    )

    return {
        "data": [{"date": r[0].isoformat(), "count": r[1]} for r in results]
    }


@router.get("/severity-distribution")
async def get_severity_distribution(db: Session = Depends(get_db)):
    """Get distribution of incident severities for charting."""
    # Group into buckets: Low (0-30), Medium (30-60), High (60-80), Critical (80-100)
    severity_bucket = case(
        (Incident.severity_score < 30, "Low"),
        (Incident.severity_score < 60, "Medium"),
        (Incident.severity_score < 80, "High"),
        else_="Critical",
    ).label("bucket")

    results = (
        db.query(
            severity_bucket,
            func.count(Incident.id),
        )
        .group_by(severity_bucket)
        .all()
    )

    # Ensure all buckets are present
    distribution = {"Low": 0, "Medium": 0, "High": 0, "Critical": 0}
    for bucket, count in results:
        distribution[bucket] = count

    return {"distribution": distribution}


@router.get("/top-stores")
async def get_top_stores_by_incidents(
    limit: int = Query(10, ge=1, le=50, description="Number of stores to return"),
    db: Session = Depends(get_db),
):
    """Get stores ranked by number of open incidents."""
    from ..database import IncidentStatus

    results = (
        db.query(Incident.store_id, func.count(Incident.id).label("count"))
        .filter(Incident.status != IncidentStatus.RESOLVED)
        .group_by(Incident.store_id)
        .order_by(func.count(Incident.id).desc())
        .limit(limit)
        .all()
    )

    return {"stores": [{"store_id": r[0], "incident_count": r[1]} for r in results]}
