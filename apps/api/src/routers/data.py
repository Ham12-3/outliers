"""Data ingestion router for flexible CSV upload and dataset management."""
import hashlib
import io
import json
from datetime import date, datetime
from typing import Any, Dict, List, Optional

import pandas as pd
from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from pydantic import BaseModel, Field
from sqlalchemy import text
from sqlalchemy.orm import Session

from ..database import get_db, RawDailyMetric, Dataset, DataRow
from ..services.demo_data import generate_demo_data
from ..services.schema_mapper import (
    analyze_dataframe,
    transform_dataframe,
    parse_date_value,
    parse_numeric_value,
    ColumnAnalysis,
    SchemaAnalysis,
    UserSchema,
    ColumnType,
    ColumnRole,
)

router = APIRouter()


# Response Models

class ColumnAnalysisResponse(BaseModel):
    """Analysis result for a single column."""
    name: str
    detected_type: str
    suggested_role: str
    sample_values: List[Any]
    unique_count: int
    null_count: int
    total_count: int
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    mean_value: Optional[float] = None
    std_value: Optional[float] = None
    cardinality_ratio: float = 0.0
    role_confidence: float = 0.5


class AnalyzeCSVResponse(BaseModel):
    """Response model for CSV analysis."""
    columns: List[ColumnAnalysisResponse]
    row_count: int
    suggested_date_column: Optional[str] = None
    suggested_identifiers: List[str] = []
    suggested_metrics: List[str] = []
    warnings: List[str] = []
    preview_rows: List[Dict[str, Any]] = []


class DatasetSchemaRequest(BaseModel):
    """Request model for creating a dataset with schema configuration."""
    name: str
    description: Optional[str] = None
    date_column: Optional[str] = None
    identifier_columns: List[str] = Field(default_factory=list)
    metric_columns: List[str] = Field(default_factory=list)
    attribute_columns: List[str] = Field(default_factory=list)


class DatasetResponse(BaseModel):
    """Response model for dataset information."""
    id: int
    name: str
    description: Optional[str]
    date_column: Optional[str]
    identifier_columns: List[str]
    metric_columns: List[str]
    attribute_columns: List[str]
    row_count: int
    date_range_start: Optional[str]
    date_range_end: Optional[str]
    created_at: str
    updated_at: str

    class Config:
        from_attributes = True


class DatasetListResponse(BaseModel):
    """Response model for listing datasets."""
    datasets: List[DatasetResponse]
    total: int


class UploadToDatasetResponse(BaseModel):
    """Response model for uploading data to a dataset."""
    message: str
    dataset_id: int
    rows_inserted: int
    rows_skipped: int
    errors: List[Dict[str, Any]]


class DemoDataResponse(BaseModel):
    """Response model for demo data generation."""
    message: str
    dataset_id: int
    dataset_name: str
    row_count: int
    identifiers: List[str]
    metrics: List[str]


# Helper functions

def column_analysis_to_response(analysis: ColumnAnalysis) -> ColumnAnalysisResponse:
    """Convert ColumnAnalysis to response model."""
    return ColumnAnalysisResponse(
        name=analysis.name,
        detected_type=analysis.detected_type.value,
        suggested_role=analysis.suggested_role.value,
        sample_values=analysis.sample_values,
        unique_count=analysis.unique_count,
        null_count=analysis.null_count,
        total_count=analysis.total_count,
        min_value=analysis.min_value,
        max_value=analysis.max_value,
        mean_value=analysis.mean_value,
        std_value=analysis.std_value,
        cardinality_ratio=analysis.cardinality_ratio,
        role_confidence=analysis.role_confidence,
    )


def compute_identifier_key(identifiers: Dict[str, Any]) -> str:
    """Compute a hash key from identifier values for grouping."""
    sorted_items = sorted(identifiers.items())
    key_string = "|".join(f"{k}={v}" for k, v in sorted_items)
    return hashlib.sha256(key_string.encode()).hexdigest()[:32]


def dataset_to_response(dataset: Dataset) -> DatasetResponse:
    """Convert Dataset model to response."""
    return DatasetResponse(
        id=dataset.id,
        name=dataset.name,
        description=dataset.description,
        date_column=dataset.date_column,
        identifier_columns=dataset.identifier_columns or [],
        metric_columns=dataset.metric_columns or [],
        attribute_columns=dataset.attribute_columns or [],
        row_count=dataset.row_count,
        date_range_start=dataset.date_range_start.isoformat() if dataset.date_range_start else None,
        date_range_end=dataset.date_range_end.isoformat() if dataset.date_range_end else None,
        created_at=dataset.created_at.isoformat() if dataset.created_at else "",
        updated_at=dataset.updated_at.isoformat() if dataset.updated_at else "",
    )


# Endpoints

@router.post("/analyze-csv", response_model=AnalyzeCSVResponse)
async def analyze_csv(
    file: UploadFile = File(...),
):
    """
    Analyze a CSV file and detect column types and suggest roles.

    This endpoint analyzes the CSV structure and returns:
    - Column type detection (date, integer, float, boolean, categorical, text)
    - Suggested roles (date, identifier, metric, attribute)
    - Sample values and statistics for each column
    - Suggestions for which columns to use for analysis

    Use this to preview the data and configure the schema before creating a dataset.
    """
    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="File must be a CSV")

    try:
        contents = await file.read()
        try:
            df = pd.read_csv(io.BytesIO(contents), encoding='utf-8')
        except UnicodeDecodeError:
            df = pd.read_csv(io.BytesIO(contents), encoding='latin-1')
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to parse CSV: {str(e)}")

    # Analyze the dataframe
    analysis = analyze_dataframe(df)

    # Get preview rows (first 20)
    preview_df = df.head(20)
    preview_rows = preview_df.to_dict(orient='records')

    return AnalyzeCSVResponse(
        columns=[column_analysis_to_response(c) for c in analysis.columns],
        row_count=analysis.row_count,
        suggested_date_column=analysis.suggested_date_column,
        suggested_identifiers=analysis.suggested_identifiers,
        suggested_metrics=analysis.suggested_metrics,
        warnings=analysis.warnings,
        preview_rows=preview_rows,
    )


@router.get("/datasets", response_model=DatasetListResponse)
async def list_datasets(
    db: Session = Depends(get_db),
):
    """List all datasets."""
    datasets = db.query(Dataset).order_by(Dataset.created_at.desc()).all()
    return DatasetListResponse(
        datasets=[dataset_to_response(d) for d in datasets],
        total=len(datasets),
    )


@router.post("/datasets", response_model=DatasetResponse)
async def create_dataset(
    schema: DatasetSchemaRequest,
    db: Session = Depends(get_db),
):
    """
    Create a new dataset with the specified schema configuration.

    The schema defines:
    - date_column: Optional column to use as the time dimension
    - identifier_columns: Columns used for grouping (e.g., store, product, region)
    - metric_columns: Numeric columns to run anomaly detection on
    - attribute_columns: Additional data to store but not analyze
    """
    dataset = Dataset(
        name=schema.name,
        description=schema.description,
        date_column=schema.date_column,
        identifier_columns=schema.identifier_columns,
        metric_columns=schema.metric_columns,
        attribute_columns=schema.attribute_columns,
        row_count=0,
    )
    db.add(dataset)
    db.commit()
    db.refresh(dataset)

    return dataset_to_response(dataset)


@router.get("/datasets/{dataset_id}", response_model=DatasetResponse)
async def get_dataset(
    dataset_id: int,
    db: Session = Depends(get_db),
):
    """Get details of a specific dataset."""
    dataset = db.query(Dataset).filter(Dataset.id == dataset_id).first()
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")
    return dataset_to_response(dataset)


@router.delete("/datasets/{dataset_id}")
async def delete_dataset(
    dataset_id: int,
    db: Session = Depends(get_db),
):
    """Delete a dataset and all its data."""
    dataset = db.query(Dataset).filter(Dataset.id == dataset_id).first()
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")

    db.delete(dataset)
    db.commit()

    return {"message": f"Dataset '{dataset.name}' deleted successfully"}


@router.post("/datasets/{dataset_id}/upload", response_model=UploadToDatasetResponse)
async def upload_to_dataset(
    dataset_id: int,
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
):
    """
    Upload CSV data to an existing dataset.

    The CSV must contain the columns defined in the dataset schema.
    Data is transformed according to the schema and stored in flexible JSON format.
    """
    dataset = db.query(Dataset).filter(Dataset.id == dataset_id).first()
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")

    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="File must be a CSV")

    try:
        contents = await file.read()
        try:
            df = pd.read_csv(io.BytesIO(contents), encoding='utf-8')
        except UnicodeDecodeError:
            df = pd.read_csv(io.BytesIO(contents), encoding='latin-1')
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to parse CSV: {str(e)}")

    # Build UserSchema from dataset
    user_schema = UserSchema(
        date_column=dataset.date_column,
        identifier_columns=dataset.identifier_columns or [],
        metric_columns=dataset.metric_columns or [],
        attribute_columns=dataset.attribute_columns or [],
    )

    # Validate that required columns exist in CSV
    csv_columns = set(df.columns)
    all_schema_columns = (
        ([user_schema.date_column] if user_schema.date_column else []) +
        user_schema.identifier_columns +
        user_schema.metric_columns +
        user_schema.attribute_columns
    )
    missing_columns = [c for c in all_schema_columns if c not in csv_columns]
    if missing_columns:
        raise HTTPException(
            status_code=422,
            detail={
                "message": "CSV is missing required columns",
                "missing_columns": missing_columns,
                "available_columns": list(csv_columns),
            }
        )

    # Transform data
    transformed_df, metadata = transform_dataframe(df, user_schema)

    if transformed_df.empty:
        raise HTTPException(
            status_code=422,
            detail={
                "message": "No valid rows after transformation",
                "errors": metadata.get("errors", [])[:20],
            }
        )

    # Insert rows
    rows_inserted = 0
    rows_skipped = 0
    insert_errors = []
    dates_seen = []

    for idx, row in transformed_df.iterrows():
        try:
            # Extract date
            row_date = row.get('_date')
            if row_date:
                dates_seen.append(row_date)

            # Build identifiers dict
            identifiers = {}
            for col in user_schema.identifier_columns:
                val = row.get(col)
                if val is not None:
                    identifiers[col] = str(val)

            # Build metrics dict
            metrics = {}
            for col in user_schema.metric_columns:
                val = row.get(col)
                if val is not None:
                    metrics[col] = float(val)

            # Build attributes dict
            attributes = {}
            for col in user_schema.attribute_columns:
                val = row.get(col)
                if val is not None:
                    attributes[col] = val

            # Compute identifier key
            identifier_key = compute_identifier_key(identifiers) if identifiers else None

            # Create DataRow
            data_row = DataRow(
                dataset_id=dataset_id,
                date=row_date,
                identifiers=identifiers,
                metrics=metrics,
                attributes=attributes,
                identifier_key=identifier_key,
            )
            db.add(data_row)
            rows_inserted += 1

        except Exception as e:
            insert_errors.append({
                "row": idx + 2,  # 1-indexed + header
                "error": str(e),
            })
            rows_skipped += 1

    # Update dataset stats
    dataset.row_count = dataset.row_count + rows_inserted
    if dates_seen:
        valid_dates = [d for d in dates_seen if d is not None]
        if valid_dates:
            min_date = min(valid_dates)
            max_date = max(valid_dates)
            if dataset.date_range_start is None or min_date < dataset.date_range_start:
                dataset.date_range_start = min_date
            if dataset.date_range_end is None or max_date > dataset.date_range_end:
                dataset.date_range_end = max_date

    # Store column analysis (convert numpy types to Python native types for JSON)
    analysis = analyze_dataframe(df)
    dataset.column_analysis = [
        {
            "name": c.name,
            "detected_type": c.detected_type.value,
            "suggested_role": c.suggested_role.value,
            "unique_count": int(c.unique_count),
            "null_count": int(c.null_count),
        }
        for c in analysis.columns
    ]

    db.commit()

    return UploadToDatasetResponse(
        message="Data uploaded successfully",
        dataset_id=dataset_id,
        rows_inserted=rows_inserted,
        rows_skipped=rows_skipped,
        errors=insert_errors[:20],
    )


@router.post("/datasets/{dataset_id}/upload-direct", response_model=UploadToDatasetResponse)
async def upload_with_schema(
    dataset_id: int,
    file: UploadFile = File(...),
    schema: str = Form(...),
    db: Session = Depends(get_db),
):
    """
    Upload CSV and create/update dataset schema in one step.

    This is a convenience endpoint that combines creating/updating the dataset
    schema and uploading data in a single request.

    Parameters:
    - file: CSV file to upload
    - schema: JSON string with schema configuration:
      {
        "name": "My Dataset",
        "description": "Optional description",
        "date_column": "date",
        "identifier_columns": ["store_id", "product_id"],
        "metric_columns": ["sales", "inventory"],
        "attribute_columns": ["category", "promo"]
      }
    """
    # Parse schema
    try:
        schema_data = json.loads(schema)
    except json.JSONDecodeError as e:
        raise HTTPException(status_code=400, detail=f"Invalid schema JSON: {str(e)}")

    # Get or create dataset
    dataset = db.query(Dataset).filter(Dataset.id == dataset_id).first()
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")

    # Update dataset schema
    if "name" in schema_data:
        dataset.name = schema_data["name"]
    if "description" in schema_data:
        dataset.description = schema_data["description"]
    if "date_column" in schema_data:
        dataset.date_column = schema_data["date_column"]
    if "identifier_columns" in schema_data:
        dataset.identifier_columns = schema_data["identifier_columns"]
    if "metric_columns" in schema_data:
        dataset.metric_columns = schema_data["metric_columns"]
    if "attribute_columns" in schema_data:
        dataset.attribute_columns = schema_data["attribute_columns"]

    db.commit()
    db.refresh(dataset)

    # Now upload the data (reuse the upload endpoint logic)
    return await upload_to_dataset(dataset_id, file, db)


@router.post("/generate-demo", response_model=DemoDataResponse)
async def generate_demo(
    dataset_name: str = "Demo Sales Data",
    stores: int = 5,
    products: int = 50,
    days: int = 120,
    clear_existing: bool = True,
    db: Session = Depends(get_db),
):
    """
    Generate synthetic demo data with realistic patterns and anomalies.

    Creates a new dataset with the following schema:
    - Identifiers: store_id, product_id
    - Metrics: units_sold, revenue, inventory_level
    - Attributes: category, is_promo, day_of_week

    Parameters:
    - dataset_name: Name for the demo dataset
    - stores: Number of stores to generate (default: 5)
    - products: Number of products (default: 50)
    - days: Number of days of history (default: 120)
    - clear_existing: Whether to clear existing demo datasets (default: true)
    """
    import random
    import numpy as np
    from datetime import timedelta

    if clear_existing:
        # Delete existing demo datasets
        existing_demos = db.query(Dataset).filter(Dataset.name.like("Demo%")).all()
        for d in existing_demos:
            db.delete(d)
        db.commit()

    # Create dataset
    dataset = Dataset(
        name=dataset_name,
        description="Auto-generated demo data with sales anomalies",
        date_column="date",
        identifier_columns=["store_id", "product_id"],
        metric_columns=["units_sold", "revenue", "inventory_level"],
        attribute_columns=["category", "is_promo", "day_of_week"],
        row_count=0,
    )
    db.add(dataset)
    db.commit()
    db.refresh(dataset)

    # Generate store and product IDs
    store_ids = [f"STORE-{i:03d}" for i in range(1, stores + 1)]
    categories = ["Electronics", "Food", "Clothing", "Home", "Sports"]
    product_ids = []
    product_categories = {}
    for i in range(1, products + 1):
        cat = random.choice(categories)
        pid = f"{cat[:3].upper()}-{i:04d}"
        product_ids.append(pid)
        product_categories[pid] = cat

    # Generate data
    end_date = date.today()
    start_date = end_date - timedelta(days=days)
    current_date = start_date
    rows_inserted = 0
    dates_list = []

    # Base patterns per store-product
    base_sales = {
        (s, p): random.randint(5, 50)
        for s in store_ids
        for p in product_ids
    }

    while current_date <= end_date:
        dates_list.append(current_date)
        day_of_week = current_date.weekday()
        is_weekend = day_of_week >= 5

        for store_id in store_ids:
            for product_id in product_ids:
                base = base_sales[(store_id, product_id)]

                # Apply patterns
                weekend_mult = 1.3 if is_weekend else 1.0
                seasonal = 1 + 0.2 * np.sin(2 * np.pi * (current_date.timetuple().tm_yday / 365))
                noise = random.gauss(1, 0.15)
                is_promo = random.random() < 0.1
                promo_mult = 1.5 if is_promo else 1.0

                units_sold = max(0, int(base * weekend_mult * seasonal * noise * promo_mult))

                # Inject anomalies (5% chance)
                if random.random() < 0.05:
                    anomaly_type = random.choice(["spike", "drop", "zero"])
                    if anomaly_type == "spike":
                        units_sold = int(units_sold * random.uniform(3, 5))
                    elif anomaly_type == "drop":
                        units_sold = int(units_sold * random.uniform(0.1, 0.3))
                    else:
                        units_sold = 0

                price = random.uniform(5, 100)
                revenue = units_sold * price
                inventory_level = max(0, random.randint(50, 200) - units_sold)

                # Create data row
                data_row = DataRow(
                    dataset_id=dataset.id,
                    date=current_date,
                    identifiers={
                        "store_id": store_id,
                        "product_id": product_id,
                    },
                    metrics={
                        "units_sold": units_sold,
                        "revenue": round(revenue, 2),
                        "inventory_level": inventory_level,
                    },
                    attributes={
                        "category": product_categories[product_id],
                        "is_promo": is_promo,
                        "day_of_week": day_of_week,
                    },
                    identifier_key=compute_identifier_key({
                        "store_id": store_id,
                        "product_id": product_id,
                    }),
                )
                db.add(data_row)
                rows_inserted += 1

        current_date += timedelta(days=1)

    # Update dataset stats
    dataset.row_count = rows_inserted
    dataset.date_range_start = start_date
    dataset.date_range_end = end_date
    db.commit()

    return DemoDataResponse(
        message="Demo data generated successfully",
        dataset_id=dataset.id,
        dataset_name=dataset.name,
        row_count=rows_inserted,
        identifiers=["store_id", "product_id"],
        metrics=["units_sold", "revenue", "inventory_level"],
    )


@router.get("/stats")
async def get_data_stats(db: Session = Depends(get_db)):
    """Get overall statistics about all data."""
    # Dataset stats
    dataset_count = db.query(Dataset).count()
    total_rows = db.query(DataRow).count()

    # Get date range across all datasets
    result = db.execute(
        text("""
            SELECT MIN(date_range_start), MAX(date_range_end)
            FROM datasets
            WHERE date_range_start IS NOT NULL
        """)
    ).fetchone()

    min_date = result[0] if result else None
    max_date = result[1] if result else None

    # Also get legacy data stats if any
    legacy_result = db.execute(
        text("""
            SELECT
                COUNT(*) as total_rows,
                COUNT(DISTINCT store_id) as store_count,
                COUNT(DISTINCT sku_id) as sku_count,
                MIN(date) as min_date,
                MAX(date) as max_date
            FROM raw_daily_metrics
        """)
    ).fetchone()

    return {
        "datasets": {
            "count": dataset_count,
            "total_rows": total_rows,
            "date_range": {
                "from": min_date.isoformat() if min_date else None,
                "to": max_date.isoformat() if max_date else None,
            },
        },
        "legacy": {
            "total_rows": legacy_result[0] or 0,
            "store_count": legacy_result[1] or 0,
            "sku_count": legacy_result[2] or 0,
            "date_range": {
                "from": legacy_result[3].isoformat() if legacy_result[3] else None,
                "to": legacy_result[4].isoformat() if legacy_result[4] else None,
            },
        },
    }


@router.get("/datasets/{dataset_id}/data")
async def get_dataset_data(
    dataset_id: int,
    limit: int = 100,
    offset: int = 0,
    identifier_filter: Optional[str] = None,
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
    db: Session = Depends(get_db),
):
    """
    Get data rows from a dataset with optional filtering.

    Parameters:
    - limit: Maximum rows to return (default: 100)
    - offset: Number of rows to skip (default: 0)
    - identifier_filter: Filter by identifier key
    - date_from: Filter by start date (YYYY-MM-DD)
    - date_to: Filter by end date (YYYY-MM-DD)
    """
    dataset = db.query(Dataset).filter(Dataset.id == dataset_id).first()
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")

    query = db.query(DataRow).filter(DataRow.dataset_id == dataset_id)

    if identifier_filter:
        query = query.filter(DataRow.identifier_key == identifier_filter)
    if date_from:
        query = query.filter(DataRow.date >= date_from)
    if date_to:
        query = query.filter(DataRow.date <= date_to)

    total = query.count()
    rows = query.order_by(DataRow.date.desc(), DataRow.id).offset(offset).limit(limit).all()

    return {
        "dataset_id": dataset_id,
        "total": total,
        "limit": limit,
        "offset": offset,
        "rows": [
            {
                "id": row.id,
                "date": row.date.isoformat() if row.date else None,
                "identifiers": row.identifiers,
                "metrics": row.metrics,
                "attributes": row.attributes,
            }
            for row in rows
        ],
    }


@router.get("/datasets/{dataset_id}/unique-identifiers")
async def get_unique_identifiers(
    dataset_id: int,
    db: Session = Depends(get_db),
):
    """Get unique identifier combinations for a dataset."""
    dataset = db.query(Dataset).filter(Dataset.id == dataset_id).first()
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")

    # Get distinct identifier keys with their values
    rows = (
        db.query(DataRow.identifier_key, DataRow.identifiers)
        .filter(DataRow.dataset_id == dataset_id)
        .distinct(DataRow.identifier_key)
        .all()
    )

    return {
        "dataset_id": dataset_id,
        "identifier_columns": dataset.identifier_columns,
        "unique_combinations": [
            {
                "key": row.identifier_key,
                "values": row.identifiers,
            }
            for row in rows
        ],
        "count": len(rows),
    }
