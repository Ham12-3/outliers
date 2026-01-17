"""
Tukey IQR-based outlier detection.

Uses the Tukey method (also known as box plot method) to identify outliers:
- Q1 = 25th percentile
- Q3 = 75th percentile
- IQR = Q3 - Q1
- Lower fence = Q1 - 1.5 * IQR
- Upper fence = Q3 + 1.5 * IQR
- Outliers are points outside the fences
"""
from dataclasses import dataclass
from datetime import date, timedelta
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sqlalchemy import and_
from sqlalchemy.orm import Session

from ..database import RawDailyMetric, DetectionResult, DetectorType
from ..config import get_settings

settings = get_settings()


@dataclass
class TukeyResult:
    """Result from Tukey outlier detection for a single metric."""

    metric_name: str
    actual_value: float
    q1: float
    q3: float
    iqr: float
    lower_fence: float
    upper_fence: float
    is_outlier: bool
    outlier_distance: float  # Positive = above upper, negative = below lower, 0 = not outlier
    sample_size: int
    fallback_used: Optional[str] = None


class TukeyDetector:
    """
    Tukey IQR outlier detector with fallback strategies for sparse data.

    Detection hierarchy for data sparsity:
    1. Store + SKU level (most specific)
    2. Store level (fallback if SKU has < min_samples)
    3. SKU level (fallback if store has < min_samples)
    4. Global level (final fallback)
    5. Insufficient data (if global also has < min_samples)
    """

    def __init__(
        self,
        rolling_window_days: int = None,
        min_samples: int = None,
        k: float = 1.5,
    ):
        """
        Initialise the Tukey detector.

        Args:
            rolling_window_days: Number of days for rolling baseline window
            min_samples: Minimum samples required for reliable statistics
            k: IQR multiplier for fence calculation (default 1.5)
        """
        self.rolling_window_days = rolling_window_days or settings.tukey_rolling_window_days
        self.min_samples = min_samples or settings.tukey_min_samples
        self.k = k

    def detect(
        self,
        db: Session,
        target_date: date,
        store_id: str,
        sku_id: str,
        metrics: List[str] = None,
    ) -> List[TukeyResult]:
        """
        Run Tukey outlier detection for a specific date, store, and SKU.

        Args:
            db: Database session
            target_date: Date to check for outliers
            store_id: Store identifier
            sku_id: SKU identifier
            metrics: List of metrics to check (default: ['sold', 'delta_on_hand'])

        Returns:
            List of TukeyResult for each metric
        """
        if metrics is None:
            metrics = ["sold", "delta_on_hand"]

        results = []

        for metric in metrics:
            result = self._detect_single_metric(
                db, target_date, store_id, sku_id, metric
            )
            if result:
                results.append(result)

        return results

    def _detect_single_metric(
        self,
        db: Session,
        target_date: date,
        store_id: str,
        sku_id: str,
        metric_name: str,
    ) -> Optional[TukeyResult]:
        """Detect outliers for a single metric with fallback strategy."""
        window_start = target_date - timedelta(days=self.rolling_window_days)
        window_end = target_date - timedelta(days=1)  # Exclude target date from baseline

        # Get target value
        target_value = self._get_metric_value(db, target_date, store_id, sku_id, metric_name)
        if target_value is None:
            return None

        # Try store+SKU level first
        baseline, fallback = self._get_baseline_values(
            db, window_start, window_end, store_id, sku_id, metric_name
        )

        if len(baseline) < self.min_samples:
            return TukeyResult(
                metric_name=metric_name,
                actual_value=target_value,
                q1=0,
                q3=0,
                iqr=0,
                lower_fence=0,
                upper_fence=0,
                is_outlier=False,
                outlier_distance=0,
                sample_size=len(baseline),
                fallback_used="insufficient_data",
            )

        # Calculate statistics
        q1, q3, iqr, lower_fence, upper_fence = self._calculate_fences(baseline)

        # Determine if outlier
        is_outlier = target_value < lower_fence or target_value > upper_fence
        outlier_distance = 0.0
        if target_value < lower_fence:
            outlier_distance = target_value - lower_fence  # Negative
        elif target_value > upper_fence:
            outlier_distance = target_value - upper_fence  # Positive

        return TukeyResult(
            metric_name=metric_name,
            actual_value=target_value,
            q1=q1,
            q3=q3,
            iqr=iqr,
            lower_fence=lower_fence,
            upper_fence=upper_fence,
            is_outlier=is_outlier,
            outlier_distance=outlier_distance,
            sample_size=len(baseline),
            fallback_used=fallback,
        )

    def _get_metric_value(
        self,
        db: Session,
        target_date: date,
        store_id: str,
        sku_id: str,
        metric_name: str,
    ) -> Optional[float]:
        """Get the metric value for the target date."""
        if metric_name == "delta_on_hand":
            # Need current and previous day
            prev_date = target_date - timedelta(days=1)
            current = (
                db.query(RawDailyMetric.on_hand)
                .filter(
                    RawDailyMetric.date == target_date,
                    RawDailyMetric.store_id == store_id,
                    RawDailyMetric.sku_id == sku_id,
                )
                .first()
            )
            previous = (
                db.query(RawDailyMetric.on_hand)
                .filter(
                    RawDailyMetric.date == prev_date,
                    RawDailyMetric.store_id == store_id,
                    RawDailyMetric.sku_id == sku_id,
                )
                .first()
            )
            if current and previous:
                return float(current[0] - previous[0])
            return None

        # Standard metric
        result = (
            db.query(getattr(RawDailyMetric, metric_name))
            .filter(
                RawDailyMetric.date == target_date,
                RawDailyMetric.store_id == store_id,
                RawDailyMetric.sku_id == sku_id,
            )
            .first()
        )
        return float(result[0]) if result else None

    def _get_baseline_values(
        self,
        db: Session,
        window_start: date,
        window_end: date,
        store_id: str,
        sku_id: str,
        metric_name: str,
    ) -> Tuple[List[float], Optional[str]]:
        """
        Get baseline values with fallback strategy.

        Returns tuple of (values, fallback_level_used).
        """
        # Level 1: Store + SKU
        values = self._query_baseline(db, window_start, window_end, store_id, sku_id, metric_name)
        if len(values) >= self.min_samples:
            return values, None

        # Level 2: Store only (aggregate all SKUs in store)
        values = self._query_baseline(db, window_start, window_end, store_id, None, metric_name)
        if len(values) >= self.min_samples:
            return values, "store"

        # Level 3: SKU only (aggregate all stores for this SKU)
        values = self._query_baseline(db, window_start, window_end, None, sku_id, metric_name)
        if len(values) >= self.min_samples:
            return values, "sku"

        # Level 4: Global (all data in window)
        values = self._query_baseline(db, window_start, window_end, None, None, metric_name)
        if len(values) >= self.min_samples:
            return values, "global"

        # Insufficient data at all levels
        return values, "insufficient_data"

    def _query_baseline(
        self,
        db: Session,
        window_start: date,
        window_end: date,
        store_id: Optional[str],
        sku_id: Optional[str],
        metric_name: str,
    ) -> List[float]:
        """Query baseline values from database."""
        if metric_name == "delta_on_hand":
            return self._query_delta_on_hand_baseline(
                db, window_start, window_end, store_id, sku_id
            )

        query = db.query(getattr(RawDailyMetric, metric_name)).filter(
            RawDailyMetric.date >= window_start,
            RawDailyMetric.date <= window_end,
        )

        if store_id:
            query = query.filter(RawDailyMetric.store_id == store_id)
        if sku_id:
            query = query.filter(RawDailyMetric.sku_id == sku_id)

        results = query.all()
        return [float(r[0]) for r in results if r[0] is not None]

    def _query_delta_on_hand_baseline(
        self,
        db: Session,
        window_start: date,
        window_end: date,
        store_id: Optional[str],
        sku_id: Optional[str],
    ) -> List[float]:
        """Query delta_on_hand baseline by computing day-over-day changes."""
        # We need one extra day before window_start for the first delta
        extended_start = window_start - timedelta(days=1)

        query = db.query(
            RawDailyMetric.date,
            RawDailyMetric.store_id,
            RawDailyMetric.sku_id,
            RawDailyMetric.on_hand,
        ).filter(
            RawDailyMetric.date >= extended_start,
            RawDailyMetric.date <= window_end,
        )

        if store_id:
            query = query.filter(RawDailyMetric.store_id == store_id)
        if sku_id:
            query = query.filter(RawDailyMetric.sku_id == sku_id)

        query = query.order_by(
            RawDailyMetric.store_id,
            RawDailyMetric.sku_id,
            RawDailyMetric.date,
        )

        results = query.all()

        # Compute deltas
        deltas = []
        prev = {}
        for row in results:
            key = (row.store_id, row.sku_id)
            if key in prev and row.date >= window_start:
                delta = row.on_hand - prev[key]
                deltas.append(float(delta))
            prev[key] = row.on_hand

        return deltas

    def _calculate_fences(
        self, values: List[float]
    ) -> Tuple[float, float, float, float, float]:
        """
        Calculate Tukey fences from values.

        Returns: (q1, q3, iqr, lower_fence, upper_fence)
        """
        arr = np.array(values)
        q1 = float(np.percentile(arr, 25))
        q3 = float(np.percentile(arr, 75))
        iqr = q3 - q1
        lower_fence = q1 - self.k * iqr
        upper_fence = q3 + self.k * iqr
        return q1, q3, iqr, lower_fence, upper_fence

    def save_results(
        self,
        db: Session,
        target_date: date,
        store_id: str,
        sku_id: str,
        results: List[TukeyResult],
    ) -> List[int]:
        """Save Tukey detection results to database."""
        saved_ids = []

        for result in results:
            # Generate explanation reasons
            reasons = []
            if result.is_outlier:
                if result.outlier_distance > 0:
                    reasons.append({
                        "type": "above_upper_fence",
                        "message": f"{result.metric_name} value ({result.actual_value:.1f}) is {result.outlier_distance:.1f} above upper fence ({result.upper_fence:.1f})",
                        "distance": result.outlier_distance,
                    })
                else:
                    reasons.append({
                        "type": "below_lower_fence",
                        "message": f"{result.metric_name} value ({result.actual_value:.1f}) is {abs(result.outlier_distance):.1f} below lower fence ({result.lower_fence:.1f})",
                        "distance": result.outlier_distance,
                    })

            detection = DetectionResult(
                date=target_date,
                store_id=store_id,
                sku_id=sku_id,
                detector_type=DetectorType.TUKEY,
                metric_name=result.metric_name,
                q1=result.q1,
                q3=result.q3,
                iqr=result.iqr,
                lower_fence=result.lower_fence,
                upper_fence=result.upper_fence,
                actual_value=result.actual_value,
                outlier_distance=result.outlier_distance,
                is_outlier=result.is_outlier,
                reasons=reasons if reasons else None,
                sample_size=result.sample_size,
                fallback_used=result.fallback_used,
            )
            db.add(detection)
            db.flush()
            saved_ids.append(detection.id)

        return saved_ids
