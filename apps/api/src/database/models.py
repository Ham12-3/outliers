"""Database models for the inventory anomaly detection system."""
from datetime import datetime, date
from sqlalchemy import (
    Column,
    Integer,
    String,
    Float,
    Boolean,
    Date,
    DateTime,
    Text,
    JSON,
    ForeignKey,
    Index,
    UniqueConstraint,
    Enum as SQLEnum,
)
from sqlalchemy.orm import relationship, declarative_base
import enum

Base = declarative_base()


class IncidentStatus(str, enum.Enum):
    """Incident status values."""
    OPEN = "open"
    INVESTIGATING = "investigating"
    RESOLVED = "resolved"


class DetectorType(str, enum.Enum):
    """Detector type values."""
    TUKEY = "tukey"
    ISOLATION_FOREST = "isolation_forest"


class RawDailyMetric(Base):
    """Raw daily inventory metrics from CSV uploads or demo data."""

    __tablename__ = "raw_daily_metrics"

    id = Column(Integer, primary_key=True, autoincrement=True)
    date = Column(Date, nullable=False)
    store_id = Column(String(50), nullable=False)
    sku_id = Column(String(50), nullable=False)
    on_hand = Column(Integer, nullable=False)
    sold = Column(Integer, nullable=False, default=0)
    delivered = Column(Integer, nullable=False, default=0)
    returned = Column(Integer, nullable=False, default=0)
    price = Column(Float, nullable=False)
    promo_flag = Column(Boolean, nullable=False, default=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    __table_args__ = (
        UniqueConstraint("date", "store_id", "sku_id", name="uq_daily_metric"),
        Index("ix_raw_daily_metrics_date_store_sku", "date", "store_id", "sku_id"),
        Index("ix_raw_daily_metrics_store", "store_id"),
        Index("ix_raw_daily_metrics_sku", "sku_id"),
        Index("ix_raw_daily_metrics_date", "date"),
    )


class FeatureDaily(Base):
    """Computed daily features for ML models."""

    __tablename__ = "features_daily"

    id = Column(Integer, primary_key=True, autoincrement=True)
    date = Column(Date, nullable=False)
    store_id = Column(String(50), nullable=False)
    sku_id = Column(String(50), nullable=False)

    # Raw metrics (copied for convenience)
    on_hand = Column(Integer, nullable=False)
    sold = Column(Integer, nullable=False)
    delivered = Column(Integer, nullable=False)
    returned = Column(Integer, nullable=False)
    price = Column(Float, nullable=False)
    promo_flag = Column(Boolean, nullable=False)

    # Computed features
    delta_on_hand = Column(Integer, nullable=True)  # on_hand[t] - on_hand[t-1]
    day_of_week = Column(Integer, nullable=False)  # 0-6, Monday=0

    # Rolling statistics (28-day window)
    sold_rolling_mean = Column(Float, nullable=True)
    sold_rolling_std = Column(Float, nullable=True)
    on_hand_rolling_mean = Column(Float, nullable=True)
    on_hand_rolling_std = Column(Float, nullable=True)

    created_at = Column(DateTime, default=datetime.utcnow)

    __table_args__ = (
        UniqueConstraint("date", "store_id", "sku_id", name="uq_feature_daily"),
        Index("ix_features_daily_date_store_sku", "date", "store_id", "sku_id"),
        Index("ix_features_daily_date", "date"),
    )


class DetectionResult(Base):
    """Detection results from outlier detection algorithms."""

    __tablename__ = "detection_results"

    id = Column(Integer, primary_key=True, autoincrement=True)
    date = Column(Date, nullable=False)
    store_id = Column(String(50), nullable=False)
    sku_id = Column(String(50), nullable=False)
    detector_type = Column(
        SQLEnum(DetectorType, values_callable=lambda obj: [e.value for e in obj]),
        nullable=False
    )

    # Tukey-specific fields
    metric_name = Column(String(50), nullable=True)  # 'sold' or 'delta_on_hand'
    q1 = Column(Float, nullable=True)
    q3 = Column(Float, nullable=True)
    iqr = Column(Float, nullable=True)
    lower_fence = Column(Float, nullable=True)
    upper_fence = Column(Float, nullable=True)
    actual_value = Column(Float, nullable=True)
    outlier_distance = Column(Float, nullable=True)  # How far outside fence

    # Isolation Forest specific fields
    anomaly_score = Column(Float, nullable=True)
    threshold_used = Column(Float, nullable=True)

    # Common fields
    is_outlier = Column(Boolean, nullable=False, default=False)
    reasons = Column(JSON, nullable=True)  # Explainability data
    sample_size = Column(Integer, nullable=True)  # Window sample size used
    fallback_used = Column(String(50), nullable=True)  # 'store', 'sku', 'global', or null

    created_at = Column(DateTime, default=datetime.utcnow)

    __table_args__ = (
        Index("ix_detection_results_date_store_sku", "date", "store_id", "sku_id"),
        Index("ix_detection_results_is_outlier", "is_outlier"),
        Index("ix_detection_results_detector", "detector_type"),
    )


class Incident(Base):
    """Aggregated incidents grouping related anomalies."""

    __tablename__ = "incidents"

    id = Column(Integer, primary_key=True, autoincrement=True)
    date = Column(Date, nullable=False)
    store_id = Column(String(50), nullable=False)
    dataset_id = Column(Integer, ForeignKey("datasets.id", ondelete="CASCADE"), nullable=True)
    status = Column(
        SQLEnum(IncidentStatus, values_callable=lambda obj: [e.value for e in obj]),
        nullable=False,
        default=IncidentStatus.OPEN
    )
    severity_score = Column(Float, nullable=False)  # 0-100
    headline = Column(String(500), nullable=False)
    description = Column(Text, nullable=True)

    # Aggregated info
    sku_count = Column(Integer, nullable=False, default=1)
    estimated_impact = Column(Float, nullable=True)  # Financial impact estimate
    detectors_triggered = Column(JSON, nullable=False)  # List of detector types

    # Workflow
    assignee = Column(String(100), nullable=True)
    resolution_reason = Column(Text, nullable=True)
    resolved_at = Column(DateTime, nullable=True)

    # Deduplication key
    dedup_key = Column(String(255), nullable=False, unique=True)

    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    items = relationship("IncidentItem", back_populates="incident", cascade="all, delete-orphan")
    notes = relationship("IncidentNote", back_populates="incident", cascade="all, delete-orphan")

    __table_args__ = (
        Index("ix_incidents_date", "date"),
        Index("ix_incidents_store", "store_id"),
        Index("ix_incidents_status", "status"),
        Index("ix_incidents_severity", "severity_score"),
        Index("ix_incidents_date_store", "date", "store_id"),
    )


class IncidentItem(Base):
    """Individual SKU items within an incident."""

    __tablename__ = "incident_items"

    id = Column(Integer, primary_key=True, autoincrement=True)
    incident_id = Column(Integer, ForeignKey("incidents.id", ondelete="CASCADE"), nullable=False)
    sku_id = Column(String(50), nullable=False)
    detection_result_ids = Column(JSON, nullable=False)  # List of detection_result IDs
    contribution_score = Column(Float, nullable=True)  # How much this SKU contributes to severity

    incident = relationship("Incident", back_populates="items")

    __table_args__ = (
        Index("ix_incident_items_incident", "incident_id"),
        Index("ix_incident_items_sku", "sku_id"),
    )


class IncidentNote(Base):
    """Notes and activity log for incidents."""

    __tablename__ = "incident_notes"

    id = Column(Integer, primary_key=True, autoincrement=True)
    incident_id = Column(Integer, ForeignKey("incidents.id", ondelete="CASCADE"), nullable=False)
    author = Column(String(100), nullable=False, default="system")
    content = Column(Text, nullable=False)
    note_type = Column(String(50), nullable=False, default="comment")  # 'comment', 'status_change', 'assignment'
    created_at = Column(DateTime, default=datetime.utcnow)

    incident = relationship("Incident", back_populates="notes")

    __table_args__ = (Index("ix_incident_notes_incident", "incident_id"),)


class Dataset(Base):
    """Uploaded datasets with flexible schema configuration.

    Each dataset represents a CSV upload with user-defined column roles:
    - date_column: optional column to use as time dimension
    - identifier_columns: columns used for grouping (e.g., store, product)
    - metric_columns: numeric columns to run anomaly detection on
    - attribute_columns: additional data to store but not analyze
    """

    __tablename__ = "datasets"

    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String(255), nullable=False)
    description = Column(Text, nullable=True)

    # Schema configuration (user-defined)
    date_column = Column(String(100), nullable=True)
    identifier_columns = Column(JSON, nullable=False, default=list)
    metric_columns = Column(JSON, nullable=False, default=list)
    attribute_columns = Column(JSON, nullable=False, default=list)

    # Column analysis (auto-detected during upload)
    column_analysis = Column(JSON, nullable=True)

    # Stats
    row_count = Column(Integer, nullable=False, default=0)
    date_range_start = Column(Date, nullable=True)
    date_range_end = Column(Date, nullable=True)

    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    rows = relationship("DataRow", back_populates="dataset", cascade="all, delete-orphan")

    __table_args__ = (
        Index("ix_datasets_name", "name"),
    )


class DataRow(Base):
    """Flexible data storage for any CSV structure.

    Stores data in a normalized format:
    - identifiers: JSON dict of identifier column values (e.g., {"store_id": "LONDON-001", "product": "SKU-123"})
    - metrics: JSON dict of metric column values (e.g., {"sales": 100, "inventory": 500})
    - attributes: JSON dict of attribute column values (e.g., {"promo": true, "category": "Food"})
    - identifier_key: hash of identifier values for efficient grouping/querying
    """

    __tablename__ = "data_rows"

    id = Column(Integer, primary_key=True, autoincrement=True)
    dataset_id = Column(Integer, ForeignKey("datasets.id", ondelete="CASCADE"), nullable=False)

    # Date (parsed from user-specified column)
    date = Column(Date, nullable=True)

    # Flexible data storage as JSON
    identifiers = Column(JSON, nullable=False, default=dict)
    metrics = Column(JSON, nullable=False, default=dict)
    attributes = Column(JSON, nullable=False, default=dict)

    # Computed identifier key for grouping (hash of identifier values)
    identifier_key = Column(String(64), nullable=True)

    created_at = Column(DateTime, default=datetime.utcnow)

    # Relationships
    dataset = relationship("Dataset", back_populates="rows")

    __table_args__ = (
        Index("ix_data_rows_dataset", "dataset_id"),
        Index("ix_data_rows_date", "date"),
        Index("ix_data_rows_identifier_key", "identifier_key"),
        Index("ix_data_rows_dataset_date", "dataset_id", "date"),
        Index("ix_data_rows_dataset_identifier", "dataset_id", "identifier_key"),
    )
