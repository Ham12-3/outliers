"""
Flexible schema mapper for any CSV structure.

Automatically detects column types and suggests roles (identifier vs metric),
then allows user to configure which columns to analyze.
"""
from dataclasses import dataclass, field
from datetime import datetime, date
from typing import Any, Dict, List, Optional, Tuple, Set
from enum import Enum
import re

from dateutil import parser as date_parser
from dateutil.parser import ParserError
import numpy as np
import pandas as pd


class ColumnType(str, Enum):
    """Detected column data type."""
    DATE = "date"
    DATETIME = "datetime"
    INTEGER = "integer"
    FLOAT = "float"
    BOOLEAN = "boolean"
    CATEGORICAL = "categorical"  # String with low cardinality
    TEXT = "text"  # String with high cardinality
    UNKNOWN = "unknown"


class ColumnRole(str, Enum):
    """Suggested role for a column."""
    DATE = "date"  # Time dimension
    IDENTIFIER = "identifier"  # Grouping column (store, product, etc.)
    METRIC = "metric"  # Numeric column to analyze
    ATTRIBUTE = "attribute"  # Additional info (not used for detection)
    IGNORE = "ignore"  # Don't use


@dataclass
class ColumnAnalysis:
    """Analysis result for a single column."""
    name: str
    detected_type: ColumnType
    suggested_role: ColumnRole
    sample_values: List[Any]
    unique_count: int
    null_count: int
    total_count: int

    # Numeric stats (if applicable)
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    mean_value: Optional[float] = None
    std_value: Optional[float] = None

    # Cardinality info
    cardinality_ratio: float = 0.0  # unique_count / total_count

    # Confidence in the suggested role (0-1)
    role_confidence: float = 0.5


@dataclass
class SchemaAnalysis:
    """Complete analysis of a CSV schema."""
    columns: List[ColumnAnalysis]
    row_count: int
    suggested_date_column: Optional[str] = None
    suggested_identifiers: List[str] = field(default_factory=list)
    suggested_metrics: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)


@dataclass
class UserSchema:
    """User-configured schema for data upload."""
    date_column: Optional[str] = None  # Column to use as date (optional)
    identifier_columns: List[str] = field(default_factory=list)  # Grouping columns
    metric_columns: List[str] = field(default_factory=list)  # Columns to analyze
    attribute_columns: List[str] = field(default_factory=list)  # Extra columns to store
    ignore_columns: List[str] = field(default_factory=list)  # Columns to skip


def analyze_column(series: pd.Series, name: str) -> ColumnAnalysis:
    """Analyze a single column to determine its type and suggested role."""
    total_count = len(series)
    null_count = series.isna().sum()
    non_null = series.dropna()
    unique_count = non_null.nunique()

    # Sample values (up to 5 unique non-null)
    sample_values = non_null.unique()[:5].tolist()

    # Cardinality ratio
    cardinality_ratio = unique_count / total_count if total_count > 0 else 0

    # Detect type
    detected_type = _detect_column_type(non_null, name)

    # Calculate numeric stats if applicable
    min_val, max_val, mean_val, std_val = None, None, None, None
    if detected_type in (ColumnType.INTEGER, ColumnType.FLOAT):
        try:
            numeric_series = pd.to_numeric(non_null, errors='coerce').dropna()
            if len(numeric_series) > 0:
                min_val = float(numeric_series.min())
                max_val = float(numeric_series.max())
                mean_val = float(numeric_series.mean())
                std_val = float(numeric_series.std()) if len(numeric_series) > 1 else 0.0
        except:
            pass

    # Suggest role based on type and cardinality
    suggested_role, role_confidence = _suggest_column_role(
        name, detected_type, cardinality_ratio, unique_count, total_count
    )

    return ColumnAnalysis(
        name=name,
        detected_type=detected_type,
        suggested_role=suggested_role,
        sample_values=sample_values,
        unique_count=unique_count,
        null_count=null_count,
        total_count=total_count,
        min_value=min_val,
        max_value=max_val,
        mean_value=mean_val,
        std_value=std_val,
        cardinality_ratio=cardinality_ratio,
        role_confidence=role_confidence,
    )


def _detect_column_type(series: pd.Series, name: str) -> ColumnType:
    """Detect the data type of a column."""
    if len(series) == 0:
        return ColumnType.UNKNOWN

    # Check if already numeric
    if pd.api.types.is_integer_dtype(series):
        return ColumnType.INTEGER
    if pd.api.types.is_float_dtype(series):
        return ColumnType.FLOAT
    if pd.api.types.is_bool_dtype(series):
        return ColumnType.BOOLEAN
    if pd.api.types.is_datetime64_any_dtype(series):
        return ColumnType.DATETIME

    # Sample values for type detection
    sample = series.head(100).astype(str)

    # Try to detect dates
    if _looks_like_date_column(sample, name):
        return ColumnType.DATE

    # Try to detect booleans
    if _looks_like_boolean(sample):
        return ColumnType.BOOLEAN

    # Try to detect integers
    if _looks_like_integer(sample):
        return ColumnType.INTEGER

    # Try to detect floats
    if _looks_like_float(sample):
        return ColumnType.FLOAT

    # String types - check cardinality
    unique_ratio = series.nunique() / len(series) if len(series) > 0 else 0
    if unique_ratio < 0.5:  # Less than 50% unique = categorical
        return ColumnType.CATEGORICAL

    return ColumnType.TEXT


def _looks_like_date_column(sample: pd.Series, name: str) -> bool:
    """Check if a column looks like it contains dates."""
    # Check column name
    date_keywords = ['date', 'time', 'day', 'month', 'year', 'created', 'updated', 'timestamp', 'dt']
    name_lower = name.lower()
    if any(kw in name_lower for kw in date_keywords):
        # Verify by trying to parse some values
        success = 0
        for val in sample.head(10):
            try:
                date_parser.parse(str(val))
                success += 1
            except:
                pass
        return success >= 5  # At least 50% parse as dates

    # Try parsing anyway
    success = 0
    for val in sample.head(20):
        try:
            parsed = date_parser.parse(str(val))
            # Make sure it's a reasonable date (not just a number being interpreted as date)
            if 1900 <= parsed.year <= 2100:
                success += 1
        except:
            pass
    return success >= 10  # At least 50% parse as dates


def _looks_like_boolean(sample: pd.Series) -> bool:
    """Check if a column contains boolean values."""
    bool_values = {'true', 'false', 'yes', 'no', 'y', 'n', '1', '0', 't', 'f', 'on', 'off'}
    unique_lower = set(str(v).lower().strip() for v in sample.unique())
    return unique_lower.issubset(bool_values) and len(unique_lower) <= 2


def _looks_like_integer(sample: pd.Series) -> bool:
    """Check if a column contains integer values."""
    try:
        for val in sample:
            s = str(val).strip().replace(',', '').replace(' ', '')
            if s and s not in ('', 'nan', 'null', 'none', 'na'):
                # Check if it's a clean integer
                float_val = float(s)
                if float_val != int(float_val):
                    return False
        return True
    except:
        return False


def _looks_like_float(sample: pd.Series) -> bool:
    """Check if a column contains float values."""
    try:
        success = 0
        for val in sample:
            s = str(val).strip().replace(',', '').replace(' ', '').replace('£', '').replace('$', '').replace('€', '')
            if s and s not in ('', 'nan', 'null', 'none', 'na'):
                float(s)
                success += 1
        return success >= len(sample) * 0.8  # 80% success rate
    except:
        return False


def _suggest_column_role(
    name: str,
    col_type: ColumnType,
    cardinality_ratio: float,
    unique_count: int,
    total_count: int,
) -> Tuple[ColumnRole, float]:
    """Suggest a role for a column based on its characteristics."""
    name_lower = name.lower()

    # Date columns
    if col_type in (ColumnType.DATE, ColumnType.DATETIME):
        return ColumnRole.DATE, 0.95

    # Check for ID-like column names
    id_keywords = ['id', 'code', 'key', 'name', 'store', 'product', 'sku', 'item',
                   'customer', 'user', 'location', 'branch', 'region', 'category',
                   'department', 'supplier', 'vendor']
    if any(kw in name_lower for kw in id_keywords):
        if col_type in (ColumnType.CATEGORICAL, ColumnType.TEXT, ColumnType.INTEGER):
            return ColumnRole.IDENTIFIER, 0.8

    # Numeric columns with metric-like names
    metric_keywords = ['amount', 'total', 'count', 'qty', 'quantity', 'price', 'cost',
                       'value', 'sales', 'revenue', 'profit', 'stock', 'inventory',
                       'sold', 'returned', 'delivered', 'units', 'volume', 'weight',
                       'score', 'rating', 'percent', 'rate', 'avg', 'sum', 'balance']
    if col_type in (ColumnType.INTEGER, ColumnType.FLOAT):
        if any(kw in name_lower for kw in metric_keywords):
            return ColumnRole.METRIC, 0.9

    # Low cardinality = likely identifier
    if col_type in (ColumnType.CATEGORICAL, ColumnType.TEXT):
        if cardinality_ratio < 0.1 and unique_count < 1000:
            return ColumnRole.IDENTIFIER, 0.7
        elif cardinality_ratio < 0.3:
            return ColumnRole.IDENTIFIER, 0.5

    # Numeric with reasonable variance = likely metric
    if col_type in (ColumnType.INTEGER, ColumnType.FLOAT):
        # High cardinality numeric = probably a metric
        if cardinality_ratio > 0.1:
            return ColumnRole.METRIC, 0.7
        # Low cardinality numeric could be an ID (like year, category code)
        elif unique_count < 50:
            return ColumnRole.IDENTIFIER, 0.5
        else:
            return ColumnRole.METRIC, 0.6

    # Boolean = attribute
    if col_type == ColumnType.BOOLEAN:
        return ColumnRole.ATTRIBUTE, 0.7

    # Default to attribute
    return ColumnRole.ATTRIBUTE, 0.3


def analyze_dataframe(df: pd.DataFrame) -> SchemaAnalysis:
    """Analyze a DataFrame and suggest schema configuration."""
    columns = []

    for col_name in df.columns:
        analysis = analyze_column(df[col_name], col_name)
        columns.append(analysis)

    # Find suggested date column (highest confidence date)
    date_columns = [c for c in columns if c.suggested_role == ColumnRole.DATE]
    suggested_date = date_columns[0].name if date_columns else None

    # Find suggested identifiers (sorted by confidence)
    identifier_columns = sorted(
        [c for c in columns if c.suggested_role == ColumnRole.IDENTIFIER],
        key=lambda x: x.role_confidence,
        reverse=True
    )
    suggested_identifiers = [c.name for c in identifier_columns]

    # Find suggested metrics (sorted by confidence)
    metric_columns = sorted(
        [c for c in columns if c.suggested_role == ColumnRole.METRIC],
        key=lambda x: x.role_confidence,
        reverse=True
    )
    suggested_metrics = [c.name for c in metric_columns]

    # Warnings
    warnings = []
    if not suggested_date:
        warnings.append("No date column detected. Time-series analysis will be limited.")
    if not suggested_identifiers:
        warnings.append("No identifier columns detected. Data will be analyzed at row level.")
    if not suggested_metrics:
        warnings.append("No metric columns detected. Please select columns to analyze.")

    return SchemaAnalysis(
        columns=columns,
        row_count=len(df),
        suggested_date_column=suggested_date,
        suggested_identifiers=suggested_identifiers,
        suggested_metrics=suggested_metrics,
        warnings=warnings,
    )


def parse_date_value(value: Any, dayfirst: bool = True) -> Optional[date]:
    """Parse a date value with robust handling."""
    if value is None or (isinstance(value, str) and not value.strip()):
        return None
    if isinstance(value, date):
        return value
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, pd.Timestamp):
        return value.date()
    if isinstance(value, (int, float)):
        try:
            if value > 50000:  # Likely Unix timestamp
                return datetime.fromtimestamp(value).date()
            else:  # Likely Excel serial date
                return (pd.Timestamp('1899-12-30') + pd.Timedelta(days=int(value))).date()
        except (ValueError, OSError):
            return None
    try:
        parsed = date_parser.parse(str(value), dayfirst=dayfirst)
        return parsed.date()
    except (ParserError, ValueError):
        return None


def parse_numeric_value(value: Any) -> Optional[float]:
    """Parse a numeric value."""
    if value is None or (isinstance(value, str) and not value.strip()):
        return None
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float)):
        return float(value)
    try:
        cleaned = re.sub(r'[£$€,\s]', '', str(value))
        return float(cleaned)
    except (ValueError, TypeError):
        return None


def transform_dataframe(
    df: pd.DataFrame,
    schema: UserSchema,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Transform a DataFrame according to user-defined schema.

    Returns (transformed_df, metadata).
    """
    result_data = []
    errors = []

    for idx, row in df.iterrows():
        row_dict = row.to_dict()
        transformed = {}
        row_valid = True

        # Parse date column if specified
        if schema.date_column:
            date_val = parse_date_value(row_dict.get(schema.date_column))
            if date_val:
                transformed['_date'] = date_val
            else:
                # Date is optional, but warn if can't parse
                transformed['_date'] = None

        # Copy identifier columns
        for col in schema.identifier_columns:
            val = row_dict.get(col)
            transformed[col] = str(val).strip() if val is not None else None

        # Parse and copy metric columns
        for col in schema.metric_columns:
            val = parse_numeric_value(row_dict.get(col))
            transformed[col] = val

        # Copy attribute columns as-is
        for col in schema.attribute_columns:
            transformed[col] = row_dict.get(col)

        # Check that we have at least one identifier and one metric
        has_identifier = any(
            transformed.get(col) is not None
            for col in schema.identifier_columns
        )
        has_metric = any(
            transformed.get(col) is not None
            for col in schema.metric_columns
        )

        if schema.identifier_columns and not has_identifier:
            errors.append({"row": idx + 2, "error": "All identifier columns are empty"})
            row_valid = False

        if schema.metric_columns and not has_metric:
            errors.append({"row": idx + 2, "error": "All metric columns are empty"})
            row_valid = False

        if row_valid:
            result_data.append(transformed)

    result_df = pd.DataFrame(result_data) if result_data else pd.DataFrame()

    metadata = {
        "total_rows": len(df),
        "valid_rows": len(result_data),
        "error_rows": len(df) - len(result_data),
        "errors": errors[:50],  # Limit errors
        "schema": {
            "date_column": schema.date_column,
            "identifier_columns": schema.identifier_columns,
            "metric_columns": schema.metric_columns,
            "attribute_columns": schema.attribute_columns,
        }
    }

    return result_df, metadata
