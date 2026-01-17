"""
Detection pipeline orchestrating feature computation, outlier detection, and incident creation.

This is the main entry point for running the complete detection workflow.
"""
import hashlib
from collections import defaultdict
from datetime import date, datetime, timedelta
from typing import Dict, List, Set, Tuple

import numpy as np
import pandas as pd
from sqlalchemy import and_, text
from sqlalchemy.orm import Session

from ..database import (
    RawDailyMetric,
    FeatureDaily,
    DetectionResult,
    Incident,
    IncidentItem,
    IncidentNote,
    IncidentStatus,
    DetectorType,
)
from ..ml import TukeyDetector, IsolationForestDetector
from ..config import get_settings

settings = get_settings()


def run_detection_pipeline(db: Session) -> Dict:
    """
    Run the complete detection pipeline.

    Steps:
    1. Compute daily features from raw metrics
    2. Determine available detection modes based on data
    3. Run Tukey IQR outlier detection (for available metrics)
    4. Run Isolation Forest anomaly detection
    5. Create/update incidents from detection results

    Returns:
        Dictionary with pipeline statistics
    """
    # Determine date range to process
    # Process dates that have raw data but no features, plus recent dates
    date_range = _get_dates_to_process(db)

    if not date_range:
        return {
            "features_created": 0,
            "tukey_outliers": 0,
            "isolation_forest_outliers": 0,
            "incidents_created": 0,
            "incidents_updated": 0,
            "detection_mode": "none",
        }

    # Determine detection mode based on available data
    detection_mode = _determine_detection_mode(db)

    # Step 1: Compute features
    features_created = _compute_features(db, date_range)

    # Step 2: Clear old detection results for these dates (idempotent)
    _clear_detection_results(db, date_range)

    # Step 3: Run detectors
    tukey_detector = TukeyDetector()
    if_detector = IsolationForestDetector()

    tukey_outliers = 0
    if_outliers = 0
    all_outliers: Dict[Tuple[date, str], List[DetectionResult]] = defaultdict(list)

    # Get all store-SKU-date combinations to process
    combinations = _get_combinations_to_process(db, date_range)

    # Determine which Tukey metrics to check based on detection mode
    tukey_metrics = _get_tukey_metrics_for_mode(detection_mode)

    for target_date, store_id, sku_id in combinations:
        # Get features for this combination
        features = _get_features_for_detection(db, target_date, store_id, sku_id)
        if not features:
            continue

        # Run Tukey detection (only for available metrics)
        tukey_results = tukey_detector.detect(
            db, target_date, store_id, sku_id, metrics=tukey_metrics
        )

        for result in tukey_results:
            if result.is_outlier:
                tukey_outliers += 1

        tukey_ids = tukey_detector.save_results(
            db, target_date, store_id, sku_id, tukey_results
        )

        # Run Isolation Forest detection
        if_result = if_detector.detect(
            db, target_date, store_id, sku_id, features
        )

        if_id = None
        if if_result:
            if if_result.is_anomaly:
                if_outliers += 1
            if_id = if_detector.save_result(
                db, target_date, store_id, sku_id, if_result
            )

        # Collect outliers for incident creation
        is_tukey_outlier = any(r.is_outlier for r in tukey_results)
        is_if_outlier = if_result and if_result.is_anomaly

        if is_tukey_outlier or is_if_outlier:
            key = (target_date, store_id)
            outlier_data = {
                "sku_id": sku_id,
                "tukey_results": tukey_results,
                "if_result": if_result,
                "tukey_ids": tukey_ids,
                "if_id": if_id,
                "features": features,
            }
            all_outliers[key].append(outlier_data)

    db.commit()

    # Step 4: Create/update incidents
    incidents_created, incidents_updated = _create_incidents(db, all_outliers)

    # Clear detector caches
    if_detector.clear_cache()

    return {
        "features_created": features_created,
        "tukey_outliers": tukey_outliers,
        "isolation_forest_outliers": if_outliers,
        "incidents_created": incidents_created,
        "incidents_updated": incidents_updated,
        "detection_mode": detection_mode,
    }


def _get_dates_to_process(db: Session) -> List[date]:
    """Get dates that need processing."""
    # Get min and max dates from raw data
    result = db.execute(
        text("SELECT MIN(date), MAX(date) FROM raw_daily_metrics")
    ).fetchone()

    if not result or not result[0]:
        return []

    min_date, max_date = result

    # Process all dates in range
    dates = []
    current = min_date
    while current <= max_date:
        dates.append(current)
        current += timedelta(days=1)

    return dates


def _determine_detection_mode(db: Session) -> str:
    """
    Determine which detection mode to use based on available data.

    Returns:
        'full' - both on_hand and sold are available
        'sold_only' - only sold data is meaningful
        'on_hand_only' - only on_hand data is meaningful
        'none' - insufficient data for detection
    """
    # Check what data we have
    result = db.execute(
        text("""
            SELECT
                COUNT(*) as total,
                SUM(CASE WHEN on_hand > 0 OR on_hand IS NOT NULL THEN 1 ELSE 0 END) as has_on_hand,
                SUM(CASE WHEN sold > 0 THEN 1 ELSE 0 END) as has_sold,
                SUM(CASE WHEN on_hand = 0 AND sold = 0 THEN 1 ELSE 0 END) as zeros
            FROM raw_daily_metrics
        """)
    ).fetchone()

    if not result or result[0] == 0:
        return "none"

    total = result[0]
    has_on_hand = result[1] or 0
    has_sold = result[2] or 0

    # Determine mode based on data availability
    # Consider data "meaningful" if at least 10% of records have non-zero values
    on_hand_meaningful = has_on_hand > (total * 0.1)
    sold_meaningful = has_sold > (total * 0.1)

    if on_hand_meaningful and sold_meaningful:
        return "full"
    elif sold_meaningful:
        return "sold_only"
    elif on_hand_meaningful:
        return "on_hand_only"
    else:
        return "none"


def _get_tukey_metrics_for_mode(detection_mode: str) -> List[str]:
    """
    Get the list of Tukey metrics to check based on detection mode.

    Args:
        detection_mode: 'full', 'sold_only', 'on_hand_only', or 'none'

    Returns:
        List of metric names to check
    """
    if detection_mode == "full":
        return ["sold", "delta_on_hand"]
    elif detection_mode == "sold_only":
        return ["sold"]
    elif detection_mode == "on_hand_only":
        return ["delta_on_hand"]
    else:
        return []


def _compute_features(db: Session, date_range: List[date]) -> int:
    """Compute daily features from raw metrics."""
    if not date_range:
        return 0

    # Clear existing features for these dates
    db.execute(
        text(
            "DELETE FROM features_daily WHERE date >= :min_date AND date <= :max_date"
        ),
        {"min_date": min(date_range), "max_date": max(date_range)},
    )

    features_created = 0

    # Get all raw metrics in date range
    metrics = (
        db.query(RawDailyMetric)
        .filter(
            RawDailyMetric.date >= min(date_range),
            RawDailyMetric.date <= max(date_range),
        )
        .order_by(RawDailyMetric.store_id, RawDailyMetric.sku_id, RawDailyMetric.date)
        .all()
    )

    # Group by store-SKU for delta calculation
    by_store_sku = defaultdict(list)
    for m in metrics:
        by_store_sku[(m.store_id, m.sku_id)].append(m)

    features_to_insert = []

    for (store_id, sku_id), store_sku_metrics in by_store_sku.items():
        prev_on_hand = None

        for m in store_sku_metrics:
            delta_on_hand = None
            if prev_on_hand is not None:
                delta_on_hand = m.on_hand - prev_on_hand

            # Calculate rolling statistics (simplified - would be more efficient with window functions)
            window_start = m.date - timedelta(days=28)
            rolling_data = [
                x for x in store_sku_metrics
                if window_start <= x.date < m.date
            ]

            sold_rolling_mean = None
            sold_rolling_std = None
            on_hand_rolling_mean = None
            on_hand_rolling_std = None

            if rolling_data:
                sold_values = [x.sold for x in rolling_data]
                on_hand_values = [x.on_hand for x in rolling_data]

                if sold_values:
                    import numpy as np
                    sold_rolling_mean = float(np.mean(sold_values))
                    sold_rolling_std = float(np.std(sold_values)) if len(sold_values) > 1 else 0

                if on_hand_values:
                    import numpy as np
                    on_hand_rolling_mean = float(np.mean(on_hand_values))
                    on_hand_rolling_std = float(np.std(on_hand_values)) if len(on_hand_values) > 1 else 0

            feature = FeatureDaily(
                date=m.date,
                store_id=m.store_id,
                sku_id=m.sku_id,
                on_hand=m.on_hand,
                sold=m.sold,
                delivered=m.delivered,
                returned=m.returned,
                price=m.price,
                promo_flag=m.promo_flag,
                delta_on_hand=delta_on_hand,
                day_of_week=m.date.weekday(),
                sold_rolling_mean=sold_rolling_mean,
                sold_rolling_std=sold_rolling_std,
                on_hand_rolling_mean=on_hand_rolling_mean,
                on_hand_rolling_std=on_hand_rolling_std,
            )
            features_to_insert.append(feature)
            features_created += 1
            prev_on_hand = m.on_hand

            # Batch insert
            if len(features_to_insert) >= 5000:
                db.bulk_save_objects(features_to_insert)
                db.commit()
                features_to_insert = []

    if features_to_insert:
        db.bulk_save_objects(features_to_insert)
        db.commit()

    return features_created


def _clear_detection_results(db: Session, date_range: List[date]) -> None:
    """Clear existing detection results for the date range."""
    if not date_range:
        return

    db.execute(
        text(
            "DELETE FROM detection_results WHERE date >= :min_date AND date <= :max_date"
        ),
        {"min_date": min(date_range), "max_date": max(date_range)},
    )
    db.commit()


def _get_combinations_to_process(
    db: Session, date_range: List[date]
) -> List[Tuple[date, str, str]]:
    """Get all (date, store_id, sku_id) combinations to process."""
    if not date_range:
        return []

    results = (
        db.query(
            FeatureDaily.date,
            FeatureDaily.store_id,
            FeatureDaily.sku_id,
        )
        .filter(
            FeatureDaily.date >= min(date_range),
            FeatureDaily.date <= max(date_range),
        )
        .all()
    )

    return [(r[0], r[1], r[2]) for r in results]


def _get_features_for_detection(
    db: Session,
    target_date: date,
    store_id: str,
    sku_id: str,
) -> Dict:
    """Get features for a specific detection."""
    feature = (
        db.query(FeatureDaily)
        .filter(
            FeatureDaily.date == target_date,
            FeatureDaily.store_id == store_id,
            FeatureDaily.sku_id == sku_id,
        )
        .first()
    )

    if not feature:
        return {}

    return {
        "sold": feature.sold,
        "delivered": feature.delivered,
        "returned": feature.returned,
        "delta_on_hand": feature.delta_on_hand,
        "price": feature.price,
        "promo_flag": feature.promo_flag,
        "day_of_week": feature.day_of_week,
    }


def _create_incidents(
    db: Session,
    all_outliers: Dict[Tuple[date, str], List[Dict]],
) -> Tuple[int, int]:
    """
    Create or update incidents from detection results.

    Groups outliers by date + store into incidents.
    """
    created = 0
    updated = 0

    for (incident_date, store_id), outlier_list in all_outliers.items():
        if not outlier_list:
            continue

        # Calculate severity and gather info
        sku_ids = [o["sku_id"] for o in outlier_list]
        detectors_triggered = set()
        total_impact = 0
        headlines = []

        for outlier in outlier_list:
            features = outlier["features"]

            # Check which detectors triggered
            if any(r.is_outlier for r in outlier["tukey_results"]):
                detectors_triggered.add("tukey")
                for r in outlier["tukey_results"]:
                    if r.is_outlier:
                        if r.metric_name == "sold":
                            headlines.append(f"Unusual sales for {outlier['sku_id']}")
                        elif r.metric_name == "delta_on_hand":
                            headlines.append(f"Inventory anomaly for {outlier['sku_id']}")

            if outlier["if_result"] and outlier["if_result"].is_anomaly:
                detectors_triggered.add("isolation_forest")
                if outlier["if_result"].reasons:
                    top_reason = outlier["if_result"].reasons[0]
                    headlines.append(
                        f"{top_reason['feature']} anomaly for {outlier['sku_id']}"
                    )

            # Estimate impact
            price = features.get("price", 0) or 0
            sold = features.get("sold", 0) or 0
            delta = features.get("delta_on_hand", 0) or 0

            # Impact proxy: either unusual sales value or inventory loss
            impact = max(abs(sold * price), abs(delta * price))
            total_impact += impact

        # Calculate severity score (0-100)
        severity = _calculate_severity(
            outlier_list, total_impact, len(detectors_triggered)
        )

        # Create headline
        if len(sku_ids) == 1:
            headline = headlines[0] if headlines else f"Anomaly detected for {sku_ids[0]}"
        else:
            headline = f"{len(sku_ids)} SKUs with anomalies in store {store_id}"

        # Create deduplication key
        dedup_key = _create_dedup_key(incident_date, store_id, sorted(sku_ids))

        # Check for existing incident
        existing = (
            db.query(Incident)
            .filter(Incident.dedup_key == dedup_key)
            .first()
        )

        if existing:
            # Update existing incident
            if existing.status != IncidentStatus.RESOLVED:
                existing.severity_score = severity
                existing.estimated_impact = total_impact
                existing.sku_count = len(sku_ids)
                existing.detectors_triggered = list(detectors_triggered)
                existing.updated_at = datetime.utcnow()
                updated += 1
        else:
            # Create new incident
            incident = Incident(
                date=incident_date,
                store_id=store_id,
                status=IncidentStatus.OPEN,
                severity_score=severity,
                headline=headline,
                description=_create_description(outlier_list),
                sku_count=len(sku_ids),
                estimated_impact=total_impact,
                detectors_triggered=list(detectors_triggered),
                dedup_key=dedup_key,
            )
            db.add(incident)
            db.flush()

            # Add incident items
            for outlier in outlier_list:
                detection_ids = list(outlier["tukey_ids"])
                if outlier["if_id"]:
                    detection_ids.append(outlier["if_id"])

                item = IncidentItem(
                    incident_id=incident.id,
                    sku_id=outlier["sku_id"],
                    detection_result_ids=detection_ids,
                    contribution_score=_calculate_item_contribution(outlier, total_impact),
                )
                db.add(item)

            # Add creation note
            note = IncidentNote(
                incident_id=incident.id,
                author="system",
                content=f"Incident created by automated detection. Triggered by: {', '.join(detectors_triggered)}",
                note_type="status_change",
            )
            db.add(note)

            created += 1

    db.commit()
    return created, updated


def _calculate_severity(
    outlier_list: List[Dict],
    total_impact: float,
    detector_count: int,
) -> float:
    """
    Calculate incident severity score (0-100).

    Factors:
    - Financial impact (estimated)
    - Number of SKUs affected
    - Number of detectors triggered
    - Confidence of detections
    """
    # Base score from detector count
    detector_score = detector_count * 20  # Max 40 for both detectors

    # SKU count contribution
    sku_count = len(outlier_list)
    sku_score = min(sku_count * 5, 20)  # Max 20

    # Impact contribution (log scale)
    import math
    if total_impact > 0:
        impact_score = min(math.log10(total_impact + 1) * 5, 30)  # Max 30
    else:
        impact_score = 0

    # Confidence contribution (average outlier distance / anomaly scores)
    confidence_score = 0
    outlier_distances = []
    for outlier in outlier_list:
        for r in outlier["tukey_results"]:
            if r.is_outlier and r.outlier_distance != 0:
                # Normalise outlier distance to 0-1 range
                outlier_distances.append(min(abs(r.outlier_distance) / 50, 1))

    if outlier_distances:
        confidence_score = sum(outlier_distances) / len(outlier_distances) * 10  # Max 10

    total = detector_score + sku_score + impact_score + confidence_score
    return min(max(total, 0), 100)


def _create_dedup_key(
    incident_date: date,
    store_id: str,
    sku_ids: List[str],
) -> str:
    """Create a deduplication key for the incident."""
    # Include date window (7 days) to catch recurring issues
    week_start = incident_date - timedelta(days=incident_date.weekday())
    content = f"{week_start.isoformat()}_{store_id}_{'_'.join(sku_ids[:10])}"
    return hashlib.sha256(content.encode()).hexdigest()[:32]


def _create_description(outlier_list: List[Dict]) -> str:
    """Create a description summarising the anomalies."""
    lines = []

    for outlier in outlier_list[:5]:  # Limit to first 5 SKUs
        sku_id = outlier["sku_id"]
        reasons = []

        for r in outlier["tukey_results"]:
            if r.is_outlier:
                direction = "above" if r.outlier_distance > 0 else "below"
                reasons.append(
                    f"{r.metric_name} {abs(r.outlier_distance):.1f} {direction} normal range"
                )

        if outlier["if_result"] and outlier["if_result"].is_anomaly:
            if outlier["if_result"].reasons:
                for reason in outlier["if_result"].reasons[:2]:
                    reasons.append(reason["message"])

        if reasons:
            lines.append(f"• {sku_id}: {'; '.join(reasons)}")

    if len(outlier_list) > 5:
        lines.append(f"• ...and {len(outlier_list) - 5} more SKUs")

    return "\n".join(lines) if lines else "Multiple anomalies detected"


def _calculate_item_contribution(outlier: Dict, total_impact: float) -> float:
    """Calculate how much this item contributes to the incident severity."""
    if total_impact == 0:
        return 0

    features = outlier["features"]
    price = features.get("price", 0) or 0
    sold = features.get("sold", 0) or 0
    delta = features.get("delta_on_hand", 0) or 0

    item_impact = max(abs(sold * price), abs(delta * price))
    return item_impact / total_impact if total_impact > 0 else 0


# =============================================================================
# FLEXIBLE DATASET DETECTION
# =============================================================================

def run_flexible_detection(db: Session, dataset_id: int) -> Dict:
    """
    Run anomaly detection on a flexible dataset.

    Uses the dataset's configured metric columns for detection.
    Groups data by identifier columns and optionally by date.

    Args:
        db: Database session
        dataset_id: ID of the dataset to analyze

    Returns:
        Dictionary with detection statistics
    """
    from ..database import Dataset, DataRow

    # Get dataset
    dataset = db.query(Dataset).filter(Dataset.id == dataset_id).first()
    if not dataset:
        return {"error": "Dataset not found", "dataset_id": dataset_id}

    if not dataset.metric_columns:
        return {
            "error": "No metric columns configured",
            "dataset_id": dataset_id,
            "message": "Please configure at least one metric column for anomaly detection",
        }

    # Get all data rows
    rows = (
        db.query(DataRow)
        .filter(DataRow.dataset_id == dataset_id)
        .order_by(DataRow.date, DataRow.identifier_key)
        .all()
    )

    if not rows:
        return {
            "error": "No data rows found",
            "dataset_id": dataset_id,
            "row_count": 0,
        }

    # Convert to DataFrame for analysis
    data = []
    for row in rows:
        row_data = {
            "row_id": row.id,
            "date": row.date,
            "identifier_key": row.identifier_key,
        }
        # Add identifiers
        for col in dataset.identifier_columns:
            row_data[f"id_{col}"] = row.identifiers.get(col)
        # Add metrics
        for col in dataset.metric_columns:
            row_data[col] = row.metrics.get(col)
        data.append(row_data)

    df = pd.DataFrame(data)

    # Run detection for each metric column
    detection_results = []
    total_outliers = 0

    for metric_col in dataset.metric_columns:
        metric_data = df[metric_col].dropna()
        if len(metric_data) < 10:  # Need minimum data for detection
            continue

        # Run Tukey IQR detection
        outliers = _detect_tukey_flexible(df, metric_col)
        detection_results.extend(outliers)
        total_outliers += len([o for o in outliers if o["is_outlier"]])

    # Run Isolation Forest if we have multiple metrics
    if len(dataset.metric_columns) >= 2:
        if_outliers = _detect_isolation_forest_flexible(df, dataset.metric_columns)
        detection_results.extend(if_outliers)
        total_outliers += len([o for o in if_outliers if o["is_outlier"]])

    # Group outliers by identifier for incident creation
    outliers_by_identifier = defaultdict(list)
    for result in detection_results:
        if result["is_outlier"]:
            key = result["identifier_key"]
            outliers_by_identifier[key].append(result)

    # Create incidents for each identifier with outliers
    incidents_created = 0
    for identifier_key, outliers in outliers_by_identifier.items():
        if outliers:
            incident = _create_flexible_incident(
                db, dataset, identifier_key, outliers, df
            )
            if incident:
                incidents_created += 1

    db.commit()

    return {
        "dataset_id": dataset_id,
        "dataset_name": dataset.name,
        "rows_analyzed": len(rows),
        "metrics_analyzed": dataset.metric_columns,
        "total_outliers": total_outliers,
        "incidents_created": incidents_created,
        "detection_results": detection_results[:100],  # Limit returned results
    }


def _detect_tukey_flexible(df: pd.DataFrame, metric_col: str) -> List[Dict]:
    """
    Run Tukey IQR detection on a single metric column.

    Args:
        df: DataFrame with the data
        metric_col: Name of the metric column to analyze

    Returns:
        List of detection results
    """
    results = []
    values = df[metric_col].dropna()

    if len(values) < 4:  # Need minimum data for IQR
        return results

    # Calculate Tukey statistics
    q1 = values.quantile(0.25)
    q3 = values.quantile(0.75)
    iqr = q3 - q1
    lower_fence = q1 - 1.5 * iqr
    upper_fence = q3 + 1.5 * iqr

    # Check each row
    for idx, row in df.iterrows():
        val = row.get(metric_col)
        if val is None or np.isnan(val):
            continue

        is_outlier = val < lower_fence or val > upper_fence
        outlier_distance = 0
        if val < lower_fence:
            outlier_distance = val - lower_fence
        elif val > upper_fence:
            outlier_distance = val - upper_fence

        results.append({
            "row_id": row["row_id"],
            "identifier_key": row["identifier_key"],
            "date": row["date"],
            "detector_type": "tukey",
            "metric_name": metric_col,
            "actual_value": float(val),
            "q1": float(q1),
            "q3": float(q3),
            "iqr": float(iqr),
            "lower_fence": float(lower_fence),
            "upper_fence": float(upper_fence),
            "outlier_distance": float(outlier_distance),
            "is_outlier": is_outlier,
            "reason": f"{metric_col} value {val:.2f} is {'below' if val < lower_fence else 'above'} normal range [{lower_fence:.2f}, {upper_fence:.2f}]" if is_outlier else None,
        })

    return results


def _detect_isolation_forest_flexible(
    df: pd.DataFrame,
    metric_cols: List[str],
) -> List[Dict]:
    """
    Run Isolation Forest detection on multiple metric columns.

    Args:
        df: DataFrame with the data
        metric_cols: List of metric column names

    Returns:
        List of detection results
    """
    from sklearn.ensemble import IsolationForest
    from sklearn.preprocessing import StandardScaler

    results = []

    # Prepare feature matrix
    feature_df = df[metric_cols].copy()

    # Drop rows with any missing values
    valid_mask = ~feature_df.isna().any(axis=1)
    feature_df = feature_df[valid_mask]
    valid_indices = df.index[valid_mask]

    if len(feature_df) < 20:  # Need minimum data
        return results

    # Scale features
    scaler = StandardScaler()
    X = scaler.fit_transform(feature_df)

    # Fit Isolation Forest
    clf = IsolationForest(
        contamination=0.05,  # Expect 5% anomalies
        random_state=42,
        n_estimators=100,
    )
    predictions = clf.fit_predict(X)
    scores = clf.decision_function(X)

    # Calculate feature importances (z-scores)
    z_scores = np.abs((X - X.mean(axis=0)) / (X.std(axis=0) + 1e-8))

    # Generate results
    for i, (idx, pred, score) in enumerate(zip(valid_indices, predictions, scores)):
        is_outlier = pred == -1
        row = df.loc[idx]

        # Get top contributing features
        feature_z = z_scores[i]
        sorted_features = sorted(
            zip(metric_cols, feature_z),
            key=lambda x: x[1],
            reverse=True
        )
        top_features = sorted_features[:3]

        reasons = []
        if is_outlier:
            for feat, z in top_features:
                if z > 2:  # More than 2 standard deviations
                    reasons.append({
                        "feature": feat,
                        "z_score": float(z),
                        "message": f"{feat} is {z:.1f} standard deviations from mean",
                    })

        results.append({
            "row_id": row["row_id"],
            "identifier_key": row["identifier_key"],
            "date": row["date"],
            "detector_type": "isolation_forest",
            "anomaly_score": float(score),
            "is_outlier": is_outlier,
            "reasons": reasons,
        })

    return results


def _create_flexible_incident(
    db: Session,
    dataset,
    identifier_key: str,
    outliers: List[Dict],
    df: pd.DataFrame,
) -> Incident:
    """
    Create an incident for a set of outliers in a flexible dataset.

    Args:
        db: Database session
        dataset: Dataset object
        identifier_key: Identifier key for this group
        outliers: List of outlier detection results
        df: DataFrame with the data

    Returns:
        Created Incident object
    """
    from ..database import DataRow

    # Get identifier values from the first outlier
    first_row = df[df["identifier_key"] == identifier_key].iloc[0] if len(df[df["identifier_key"] == identifier_key]) > 0 else None
    if first_row is None:
        return None

    # Build identifier string
    id_parts = []
    for col in dataset.identifier_columns:
        val = first_row.get(f"id_{col}")
        if val:
            id_parts.append(f"{col}={val}")
    identifier_str = ", ".join(id_parts) if id_parts else identifier_key[:8]

    # Collect unique dates
    dates = set()
    for outlier in outliers:
        if outlier.get("date"):
            dates.add(outlier["date"])

    incident_date = min(dates) if dates else date.today()

    # Count outliers by detector
    tukey_count = len([o for o in outliers if o["detector_type"] == "tukey"])
    if_count = len([o for o in outliers if o["detector_type"] == "isolation_forest"])

    detectors = []
    if tukey_count > 0:
        detectors.append("tukey")
    if if_count > 0:
        detectors.append("isolation_forest")

    # Create headline
    metric_names = set(o.get("metric_name", "") for o in outliers if o.get("metric_name"))
    if metric_names:
        headline = f"Anomalies in {', '.join(metric_names)} for {identifier_str}"
    else:
        headline = f"Anomalies detected for {identifier_str}"

    # Create description
    descriptions = []
    for outlier in outliers[:5]:
        if outlier.get("reason"):
            descriptions.append(f"• {outlier['reason']}")
        elif outlier.get("reasons"):
            for r in outlier["reasons"][:2]:
                descriptions.append(f"• {r['message']}")

    description = "\n".join(descriptions) if descriptions else "Multiple anomalies detected"

    # Calculate severity
    severity = min(20 + len(outliers) * 5 + len(detectors) * 15, 100)

    # Create dedup key
    dedup_key = hashlib.sha256(
        f"{dataset.id}_{identifier_key}_{incident_date.isoformat()}".encode()
    ).hexdigest()[:32]

    # Check for existing incident
    existing = db.query(Incident).filter(Incident.dedup_key == dedup_key).first()
    if existing:
        # Update existing
        if existing.status != IncidentStatus.RESOLVED:
            existing.severity_score = severity
            existing.detectors_triggered = detectors
            existing.updated_at = datetime.utcnow()
        return existing

    # Create new incident
    # Note: Using identifier_key as store_id for compatibility with existing model
    incident = Incident(
        date=incident_date,
        store_id=identifier_str[:50],  # Use identifier string as store_id
        status=IncidentStatus.OPEN,
        severity_score=severity,
        headline=headline[:500],
        description=description,
        sku_count=1,  # Each identifier is treated as one "unit"
        estimated_impact=0,
        detectors_triggered=detectors,
        dedup_key=dedup_key,
        dataset_id=dataset.id,  # Link to dataset
    )
    db.add(incident)
    db.flush()

    # Add creation note
    note = IncidentNote(
        incident_id=incident.id,
        author="system",
        content=f"Incident created from flexible dataset '{dataset.name}'. Analyzed metrics: {', '.join(dataset.metric_columns)}",
        note_type="status_change",
    )
    db.add(note)

    return incident
