"""Tests for the flexible schema mapper module."""
import pytest
from datetime import date
import pandas as pd
import numpy as np

from src.services.schema_mapper import (
    ColumnType,
    ColumnRole,
    ColumnAnalysis,
    SchemaAnalysis,
    UserSchema,
    analyze_column,
    analyze_dataframe,
    parse_date_value,
    parse_numeric_value,
    transform_dataframe,
)


class TestColumnTypeDetection:
    """Tests for column type detection."""

    def test_detect_integer_column(self):
        series = pd.Series([1, 2, 3, 4, 5])
        analysis = analyze_column(series, "count")
        assert analysis.detected_type == ColumnType.INTEGER

    def test_detect_float_column(self):
        series = pd.Series([1.5, 2.5, 3.5, 4.5, 5.5])
        analysis = analyze_column(series, "price")
        assert analysis.detected_type == ColumnType.FLOAT

    def test_detect_string_integer(self):
        series = pd.Series(["1", "2", "3", "4", "5"])
        analysis = analyze_column(series, "quantity")
        assert analysis.detected_type == ColumnType.INTEGER

    def test_detect_string_float(self):
        series = pd.Series(["1.5", "2.5", "3.5", "4.5", "5.5"])
        analysis = analyze_column(series, "amount")
        assert analysis.detected_type == ColumnType.FLOAT

    def test_detect_date_by_name(self):
        series = pd.Series(["2024-01-15", "2024-01-16", "2024-01-17"])
        analysis = analyze_column(series, "date")
        assert analysis.detected_type == ColumnType.DATE

    def test_detect_date_by_content(self):
        series = pd.Series([
            "2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04",
            "2024-01-05", "2024-01-06", "2024-01-07", "2024-01-08",
            "2024-01-09", "2024-01-10", "2024-01-11", "2024-01-12",
            "2024-01-13", "2024-01-14", "2024-01-15", "2024-01-16",
            "2024-01-17", "2024-01-18", "2024-01-19", "2024-01-20",
        ])
        analysis = analyze_column(series, "event_time")
        assert analysis.detected_type == ColumnType.DATE

    def test_detect_boolean_column(self):
        series = pd.Series(["true", "false", "true", "true", "false"])
        analysis = analyze_column(series, "active")
        assert analysis.detected_type == ColumnType.BOOLEAN

    def test_detect_boolean_yes_no(self):
        series = pd.Series(["yes", "no", "yes", "yes", "no"])
        analysis = analyze_column(series, "promo")
        assert analysis.detected_type == ColumnType.BOOLEAN

    def test_detect_categorical(self):
        series = pd.Series(["A", "B", "A", "C", "B", "A", "C", "A", "B", "A"])
        analysis = analyze_column(series, "category")
        assert analysis.detected_type == ColumnType.CATEGORICAL

    def test_detect_text_high_cardinality(self):
        series = pd.Series([f"unique_value_{i}" for i in range(100)])
        analysis = analyze_column(series, "description")
        assert analysis.detected_type == ColumnType.TEXT


class TestColumnRoleSuggestion:
    """Tests for column role suggestion."""

    def test_date_column_role(self):
        series = pd.Series(["2024-01-15", "2024-01-16", "2024-01-17"])
        analysis = analyze_column(series, "date")
        assert analysis.suggested_role == ColumnRole.DATE

    def test_identifier_by_name(self):
        series = pd.Series(["STORE-001", "STORE-002", "STORE-003"])
        analysis = analyze_column(series, "store_id")
        assert analysis.suggested_role == ColumnRole.IDENTIFIER

    def test_identifier_by_cardinality(self):
        # Low cardinality string column
        series = pd.Series(["A", "B", "A", "C", "B", "A", "C", "A", "B", "A"] * 10)
        analysis = analyze_column(series, "region")
        assert analysis.suggested_role == ColumnRole.IDENTIFIER

    def test_metric_by_name(self):
        series = pd.Series([100, 200, 150, 300, 250])
        analysis = analyze_column(series, "sales")
        assert analysis.suggested_role == ColumnRole.METRIC

    def test_metric_by_cardinality(self):
        # High cardinality numeric column
        series = pd.Series([float(i) for i in range(100)])
        analysis = analyze_column(series, "value")
        assert analysis.suggested_role == ColumnRole.METRIC

    def test_boolean_attribute(self):
        series = pd.Series([True, False, True, True, False])
        analysis = analyze_column(series, "is_promo")
        assert analysis.suggested_role == ColumnRole.ATTRIBUTE


class TestAnalyzeDataframe:
    """Tests for full dataframe analysis."""

    def test_analyze_sales_data(self):
        df = pd.DataFrame({
            "date": ["2024-01-15", "2024-01-16", "2024-01-17"],
            "store_id": ["STORE-001", "STORE-001", "STORE-002"],
            "product_id": ["SKU-001", "SKU-002", "SKU-001"],
            "quantity_sold": [10, 20, 15],
            "revenue": [100.50, 200.75, 150.25],
            "is_promo": [True, False, True],
        })

        analysis = analyze_dataframe(df)

        assert analysis.row_count == 3
        assert len(analysis.columns) == 6
        assert analysis.suggested_date_column == "date"
        assert "store_id" in analysis.suggested_identifiers
        assert "product_id" in analysis.suggested_identifiers
        assert "quantity_sold" in analysis.suggested_metrics or "revenue" in analysis.suggested_metrics

    def test_no_date_warning(self):
        df = pd.DataFrame({
            "store": ["A", "B", "C"],
            "sales": [100, 200, 300],
        })

        analysis = analyze_dataframe(df)

        assert any("date" in w.lower() for w in analysis.warnings)

    def test_no_metrics_warning(self):
        df = pd.DataFrame({
            "category": ["A", "B", "C"],
            "name": ["Item 1", "Item 2", "Item 3"],
        })

        analysis = analyze_dataframe(df)

        assert any("metric" in w.lower() for w in analysis.warnings)


class TestDateParsing:
    """Tests for date parsing."""

    def test_iso_format(self):
        result = parse_date_value("2024-01-15")
        assert result == date(2024, 1, 15)

    def test_uk_format(self):
        result = parse_date_value("15/01/2024")
        assert result == date(2024, 1, 15)

    def test_text_format(self):
        result = parse_date_value("January 15, 2024")
        assert result == date(2024, 1, 15)

    def test_date_object(self):
        d = date(2024, 1, 15)
        result = parse_date_value(d)
        assert result == date(2024, 1, 15)

    def test_unix_timestamp(self):
        # Unix timestamp for 2024-01-15 00:00:00 UTC
        result = parse_date_value(1705276800)
        assert result is not None
        assert result.year == 2024

    def test_empty_string(self):
        result = parse_date_value("")
        assert result is None

    def test_none(self):
        result = parse_date_value(None)
        assert result is None

    def test_invalid_date(self):
        result = parse_date_value("not a date")
        assert result is None


class TestNumericParsing:
    """Tests for numeric parsing."""

    def test_integer(self):
        assert parse_numeric_value(42) == 42.0

    def test_float(self):
        assert parse_numeric_value(42.5) == 42.5

    def test_string_integer(self):
        assert parse_numeric_value("42") == 42.0

    def test_string_float(self):
        assert parse_numeric_value("42.5") == 42.5

    def test_string_with_currency(self):
        assert parse_numeric_value("$42.50") == 42.5
        assert parse_numeric_value("£42.50") == 42.5
        assert parse_numeric_value("€42.50") == 42.5

    def test_string_with_comma(self):
        assert parse_numeric_value("1,000.50") == 1000.5

    def test_none(self):
        assert parse_numeric_value(None) is None

    def test_empty_string(self):
        assert parse_numeric_value("") is None

    def test_invalid(self):
        assert parse_numeric_value("not a number") is None


class TestTransformDataframe:
    """Tests for dataframe transformation."""

    def test_transform_with_all_columns(self):
        df = pd.DataFrame({
            "date": ["2024-01-15", "2024-01-16"],
            "store": ["STORE-001", "STORE-002"],
            "product": ["SKU-001", "SKU-002"],
            "sales": [100, 200],
            "inventory": [50, 75],
            "is_promo": [True, False],
        })

        schema = UserSchema(
            date_column="date",
            identifier_columns=["store", "product"],
            metric_columns=["sales", "inventory"],
            attribute_columns=["is_promo"],
        )

        result_df, metadata = transform_dataframe(df, schema)

        assert len(result_df) == 2
        assert metadata["valid_rows"] == 2
        assert metadata["error_rows"] == 0
        assert "_date" in result_df.columns
        assert "store" in result_df.columns
        assert "product" in result_df.columns
        assert "sales" in result_df.columns
        assert "inventory" in result_df.columns

    def test_transform_without_date(self):
        df = pd.DataFrame({
            "store": ["STORE-001", "STORE-002"],
            "sales": [100, 200],
        })

        schema = UserSchema(
            date_column=None,
            identifier_columns=["store"],
            metric_columns=["sales"],
        )

        result_df, metadata = transform_dataframe(df, schema)

        assert len(result_df) == 2
        assert "_date" not in result_df.columns or result_df["_date"].isna().all()

    def test_transform_skips_invalid_rows(self):
        df = pd.DataFrame({
            "date": ["2024-01-15", "invalid_date"],
            "store": ["STORE-001", None],  # Second row has no identifier
            "sales": [100, None],  # Second row has no metric
        })

        schema = UserSchema(
            date_column="date",
            identifier_columns=["store"],
            metric_columns=["sales"],
        )

        result_df, metadata = transform_dataframe(df, schema)

        # Should only have 1 valid row
        assert len(result_df) == 1
        assert metadata["error_rows"] == 1


class TestUserSchema:
    """Tests for UserSchema dataclass."""

    def test_default_values(self):
        schema = UserSchema()
        assert schema.date_column is None
        assert schema.identifier_columns == []
        assert schema.metric_columns == []
        assert schema.attribute_columns == []
        assert schema.ignore_columns == []

    def test_with_values(self):
        schema = UserSchema(
            date_column="date",
            identifier_columns=["store", "product"],
            metric_columns=["sales", "inventory"],
            attribute_columns=["category"],
        )
        assert schema.date_column == "date"
        assert len(schema.identifier_columns) == 2
        assert len(schema.metric_columns) == 2
        assert len(schema.attribute_columns) == 1


class TestColumnAnalysis:
    """Tests for ColumnAnalysis dataclass."""

    def test_numeric_stats(self):
        series = pd.Series([10, 20, 30, 40, 50])
        analysis = analyze_column(series, "value")

        assert analysis.min_value == 10.0
        assert analysis.max_value == 50.0
        assert analysis.mean_value == 30.0
        assert analysis.std_value is not None

    def test_cardinality_ratio(self):
        series = pd.Series(["A", "B", "A", "B", "A"])
        analysis = analyze_column(series, "category")

        # 2 unique values / 5 total = 0.4
        assert analysis.cardinality_ratio == pytest.approx(0.4)
        assert analysis.unique_count == 2
        assert analysis.total_count == 5

    def test_null_count(self):
        series = pd.Series([1, 2, None, 4, None])
        analysis = analyze_column(series, "value")

        assert analysis.null_count == 2
        assert analysis.total_count == 5

    def test_sample_values(self):
        series = pd.Series(["A", "B", "C", "D", "E", "F", "G"])
        analysis = analyze_column(series, "category")

        assert len(analysis.sample_values) <= 5
        assert all(v in ["A", "B", "C", "D", "E", "F", "G"] for v in analysis.sample_values)
