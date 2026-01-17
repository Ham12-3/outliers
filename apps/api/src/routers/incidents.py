"""Incidents router for managing anomaly incidents."""
from datetime import date, datetime
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel
from sqlalchemy import and_, desc, func
from sqlalchemy.orm import Session, joinedload

from ..database import (
    get_db,
    Incident,
    IncidentItem,
    IncidentNote,
    IncidentStatus,
    DetectionResult,
    FeatureDaily,
)

router = APIRouter()


class IncidentItemResponse(BaseModel):
    """Response model for incident item."""

    id: int
    sku_id: str
    detection_result_ids: list
    contribution_score: Optional[float]

    class Config:
        from_attributes = True


class IncidentNoteResponse(BaseModel):
    """Response model for incident note."""

    id: int
    author: str
    content: str
    note_type: str
    created_at: datetime

    class Config:
        from_attributes = True


class IncidentListItem(BaseModel):
    """Response model for incident list item."""

    id: int
    date: date
    store_id: str
    status: str
    severity_score: float
    headline: str
    sku_count: int
    estimated_impact: Optional[float]
    detectors_triggered: list
    assignee: Optional[str]
    created_at: datetime

    class Config:
        from_attributes = True


class IncidentDetail(BaseModel):
    """Detailed incident response model."""

    id: int
    date: date
    store_id: str
    status: str
    severity_score: float
    headline: str
    description: Optional[str]
    sku_count: int
    estimated_impact: Optional[float]
    detectors_triggered: list
    assignee: Optional[str]
    resolution_reason: Optional[str]
    resolved_at: Optional[datetime]
    created_at: datetime
    updated_at: datetime
    items: List[IncidentItemResponse]
    notes: List[IncidentNoteResponse]

    class Config:
        from_attributes = True


class IncidentUpdate(BaseModel):
    """Model for updating an incident."""

    status: Optional[str] = None
    assignee: Optional[str] = None
    resolution_reason: Optional[str] = None


class AddNoteRequest(BaseModel):
    """Model for adding a note to an incident."""

    content: str
    author: str = "user"


class IncidentsListResponse(BaseModel):
    """Response model for incidents list."""

    incidents: List[IncidentListItem]
    total: int
    page: int
    page_size: int


@router.get("", response_model=IncidentsListResponse)
async def list_incidents(
    status: Optional[str] = Query(None, description="Filter by status: open, investigating, resolved"),
    min_severity: Optional[float] = Query(None, ge=0, le=100, description="Minimum severity score"),
    store_id: Optional[str] = Query(None, description="Filter by store ID"),
    date_from: Optional[date] = Query(None, description="Filter incidents from this date"),
    date_to: Optional[date] = Query(None, description="Filter incidents until this date"),
    page: int = Query(1, ge=1, description="Page number"),
    page_size: int = Query(20, ge=1, le=100, description="Items per page"),
    db: Session = Depends(get_db),
):
    """
    List incidents with optional filters.

    Returns paginated list of incidents sorted by severity (descending) and date (descending).
    """
    query = db.query(Incident)

    # Apply filters
    if status:
        try:
            status_enum = IncidentStatus(status.lower())
            query = query.filter(Incident.status == status_enum)
        except ValueError:
            raise HTTPException(status_code=400, detail=f"Invalid status: {status}")

    if min_severity is not None:
        query = query.filter(Incident.severity_score >= min_severity)

    if store_id:
        query = query.filter(Incident.store_id == store_id)

    if date_from:
        query = query.filter(Incident.date >= date_from)

    if date_to:
        query = query.filter(Incident.date <= date_to)

    # Get total count
    total = query.count()

    # Apply pagination and sorting
    incidents = (
        query.order_by(desc(Incident.severity_score), desc(Incident.date))
        .offset((page - 1) * page_size)
        .limit(page_size)
        .all()
    )

    return IncidentsListResponse(
        incidents=[
            IncidentListItem(
                id=inc.id,
                date=inc.date,
                store_id=inc.store_id,
                status=inc.status.value,
                severity_score=inc.severity_score,
                headline=inc.headline,
                sku_count=inc.sku_count,
                estimated_impact=inc.estimated_impact,
                detectors_triggered=inc.detectors_triggered,
                assignee=inc.assignee,
                created_at=inc.created_at,
            )
            for inc in incidents
        ],
        total=total,
        page=page,
        page_size=page_size,
    )


@router.get("/summary")
async def get_incidents_summary(db: Session = Depends(get_db)):
    """Get summary statistics for incidents."""
    # Count by status
    status_counts = (
        db.query(Incident.status, func.count(Incident.id))
        .group_by(Incident.status)
        .all()
    )

    # High severity count (>=70)
    high_severity = db.query(func.count(Incident.id)).filter(
        Incident.severity_score >= 70,
        Incident.status != IncidentStatus.RESOLVED,
    ).scalar()

    # Affected stores (with open incidents)
    affected_stores = db.query(func.count(func.distinct(Incident.store_id))).filter(
        Incident.status != IncidentStatus.RESOLVED
    ).scalar()

    # Total affected SKUs (from open incidents)
    affected_skus = db.query(func.sum(Incident.sku_count)).filter(
        Incident.status != IncidentStatus.RESOLVED
    ).scalar() or 0

    # Total estimated impact (open incidents)
    total_impact = db.query(func.sum(Incident.estimated_impact)).filter(
        Incident.status != IncidentStatus.RESOLVED
    ).scalar() or 0

    return {
        "status_counts": {status.value: count for status, count in status_counts},
        "high_severity_count": high_severity or 0,
        "affected_stores": affected_stores or 0,
        "affected_skus": affected_skus,
        "total_estimated_impact": round(total_impact, 2),
    }


@router.get("/{incident_id}", response_model=IncidentDetail)
async def get_incident(
    incident_id: int,
    db: Session = Depends(get_db),
):
    """Get detailed information about a specific incident."""
    incident = (
        db.query(Incident)
        .options(joinedload(Incident.items), joinedload(Incident.notes))
        .filter(Incident.id == incident_id)
        .first()
    )

    if not incident:
        raise HTTPException(status_code=404, detail="Incident not found")

    return IncidentDetail(
        id=incident.id,
        date=incident.date,
        store_id=incident.store_id,
        status=incident.status.value,
        severity_score=incident.severity_score,
        headline=incident.headline,
        description=incident.description,
        sku_count=incident.sku_count,
        estimated_impact=incident.estimated_impact,
        detectors_triggered=incident.detectors_triggered,
        assignee=incident.assignee,
        resolution_reason=incident.resolution_reason,
        resolved_at=incident.resolved_at,
        created_at=incident.created_at,
        updated_at=incident.updated_at,
        items=[
            IncidentItemResponse(
                id=item.id,
                sku_id=item.sku_id,
                detection_result_ids=item.detection_result_ids,
                contribution_score=item.contribution_score,
            )
            for item in incident.items
        ],
        notes=sorted(
            [
                IncidentNoteResponse(
                    id=note.id,
                    author=note.author,
                    content=note.content,
                    note_type=note.note_type,
                    created_at=note.created_at,
                )
                for note in incident.notes
            ],
            key=lambda n: n.created_at,
            reverse=True,
        ),
    )


@router.patch("/{incident_id}", response_model=IncidentDetail)
async def update_incident(
    incident_id: int,
    update: IncidentUpdate,
    db: Session = Depends(get_db),
):
    """
    Update an incident's status, assignee, or resolution reason.

    Creates audit notes for status and assignment changes.
    """
    incident = db.query(Incident).filter(Incident.id == incident_id).first()

    if not incident:
        raise HTTPException(status_code=404, detail="Incident not found")

    # Track changes for audit notes
    changes = []

    if update.status is not None:
        try:
            new_status = IncidentStatus(update.status.lower())
            if incident.status != new_status:
                changes.append(f"Status changed from {incident.status.value} to {new_status.value}")
                incident.status = new_status

                if new_status == IncidentStatus.RESOLVED:
                    incident.resolved_at = datetime.utcnow()
                else:
                    incident.resolved_at = None
        except ValueError:
            raise HTTPException(status_code=400, detail=f"Invalid status: {update.status}")

    if update.assignee is not None:
        if incident.assignee != update.assignee:
            old_assignee = incident.assignee or "unassigned"
            changes.append(f"Assignment changed from {old_assignee} to {update.assignee or 'unassigned'}")
            incident.assignee = update.assignee if update.assignee else None

    if update.resolution_reason is not None:
        incident.resolution_reason = update.resolution_reason
        changes.append("Resolution reason updated")

    # Create audit notes for changes
    for change in changes:
        note = IncidentNote(
            incident_id=incident.id,
            author="system",
            content=change,
            note_type="status_change" if "Status" in change else "assignment",
        )
        db.add(note)

    incident.updated_at = datetime.utcnow()
    db.commit()
    db.refresh(incident)

    return await get_incident(incident_id, db)


@router.post("/{incident_id}/notes", response_model=IncidentNoteResponse)
async def add_note(
    incident_id: int,
    note_request: AddNoteRequest,
    db: Session = Depends(get_db),
):
    """Add a note to an incident."""
    incident = db.query(Incident).filter(Incident.id == incident_id).first()

    if not incident:
        raise HTTPException(status_code=404, detail="Incident not found")

    note = IncidentNote(
        incident_id=incident_id,
        author=note_request.author,
        content=note_request.content,
        note_type="comment",
    )
    db.add(note)
    db.commit()
    db.refresh(note)

    return IncidentNoteResponse(
        id=note.id,
        author=note.author,
        content=note.content,
        note_type=note.note_type,
        created_at=note.created_at,
    )


@router.get("/{incident_id}/detection-details")
async def get_incident_detection_details(
    incident_id: int,
    db: Session = Depends(get_db),
):
    """
    Get detailed detection results for an incident.

    Returns Tukey and Isolation Forest details for each affected SKU.
    """
    incident = (
        db.query(Incident)
        .options(joinedload(Incident.items))
        .filter(Incident.id == incident_id)
        .first()
    )

    if not incident:
        raise HTTPException(status_code=404, detail="Incident not found")

    # Get all detection result IDs from items
    detection_ids = []
    for item in incident.items:
        detection_ids.extend(item.detection_result_ids)

    # Fetch detection results
    detection_results = (
        db.query(DetectionResult)
        .filter(DetectionResult.id.in_(detection_ids))
        .all()
    )

    # Group by SKU and detector type
    by_sku = {}
    for result in detection_results:
        if result.sku_id not in by_sku:
            by_sku[result.sku_id] = {"tukey": [], "isolation_forest": None}

        if result.detector_type.value == "tukey":
            by_sku[result.sku_id]["tukey"].append({
                "metric": result.metric_name,
                "actual_value": result.actual_value,
                "q1": result.q1,
                "q3": result.q3,
                "iqr": result.iqr,
                "lower_fence": result.lower_fence,
                "upper_fence": result.upper_fence,
                "outlier_distance": result.outlier_distance,
                "sample_size": result.sample_size,
                "fallback_used": result.fallback_used,
            })
        else:
            by_sku[result.sku_id]["isolation_forest"] = {
                "anomaly_score": result.anomaly_score,
                "threshold": result.threshold_used,
                "reasons": result.reasons,
                "fallback_used": result.fallback_used,
            }

    return {
        "incident_id": incident_id,
        "date": incident.date.isoformat(),
        "store_id": incident.store_id,
        "detections_by_sku": by_sku,
    }
