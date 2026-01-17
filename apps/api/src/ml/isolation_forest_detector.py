"""
Isolation Forest anomaly detection.

Uses sklearn's IsolationForest algorithm to detect anomalies based on
multiple features. The algorithm isolates observations by randomly
selecting a feature and then randomly selecting a split value between
the maximum and minimum values of the selected feature.
"""
import hashlib
import pickle
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sqlalchemy import and_
from sqlalchemy.orm import Session

from ..database import RawDailyMetric, FeatureDaily, DetectionResult, DetectorType
from ..config import get_settings

settings = get_settings()

# Features used for Isolation Forest
FEATURE_COLUMNS = [
    "sold",
    "delivered",
    "returned",
    "delta_on_hand",
    "price",
    "promo_flag",
    "day_of_week",
]


@dataclass
class IsolationForestResult:
    """Result from Isolation Forest anomaly detection."""

    anomaly_score: float  # -1 to 1, where -1 is most anomalous
    is_anomaly: bool
    threshold_used: float
    reasons: List[Dict]  # Top contributing features
    fallback_used: Optional[str] = None


class IsolationForestDetector:
    """
    Isolation Forest anomaly detector with training strategy fallbacks.

    Training hierarchy:
    1. Per-store model (if enough data)
    2. Global model (fallback)
    """

    def __init__(
        self,
        contamination: float = None,
        min_samples: int = None,
        n_estimators: int = 100,
        cache_dir: str = ".cache/models",
    ):
        """
        Initialise the Isolation Forest detector.

        Args:
            contamination: Expected proportion of outliers (default from settings)
            min_samples: Minimum samples required for training (default from settings)
            n_estimators: Number of trees in the forest
            cache_dir: Directory to cache trained models
        """
        self.contamination = contamination or settings.isolation_forest_contamination
        self.min_samples = min_samples or settings.isolation_forest_min_samples
        self.n_estimators = n_estimators
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # In-memory cache for models
        self._models: Dict[str, Tuple[IsolationForest, StandardScaler]] = {}
        self._rolling_stats: Dict[str, Dict] = {}

    def detect(
        self,
        db: Session,
        target_date: date,
        store_id: str,
        sku_id: str,
        features: Dict,
    ) -> Optional[IsolationForestResult]:
        """
        Run Isolation Forest anomaly detection for a specific observation.

        Args:
            db: Database session
            target_date: Date to check
            store_id: Store identifier
            sku_id: SKU identifier
            features: Dictionary of feature values for this observation

        Returns:
            IsolationForestResult or None if insufficient data
        """
        # Get or train model
        model, scaler, fallback = self._get_model(db, store_id, target_date)

        if model is None:
            return IsolationForestResult(
                anomaly_score=0,
                is_anomaly=False,
                threshold_used=self.contamination,
                reasons=[],
                fallback_used="insufficient_data",
            )

        # Prepare feature vector
        feature_vector = self._prepare_features(features)
        if feature_vector is None:
            return None

        # Scale features (convert to DataFrame to match training format)
        feature_df = pd.DataFrame([feature_vector], columns=FEATURE_COLUMNS)
        feature_scaled = scaler.transform(feature_df)

        # Get anomaly score
        # IsolationForest: decision_function returns negative for anomalies
        anomaly_score = float(model.decision_function(feature_scaled)[0])
        is_anomaly = model.predict(feature_scaled)[0] == -1

        # Calculate feature contributions (z-scores compared to rolling baseline)
        reasons = self._calculate_feature_reasons(
            db, store_id, sku_id, target_date, features
        )

        return IsolationForestResult(
            anomaly_score=anomaly_score,
            is_anomaly=is_anomaly,
            threshold_used=self.contamination,
            reasons=reasons,
            fallback_used=fallback,
        )

    def _get_model(
        self,
        db: Session,
        store_id: str,
        target_date: date,
    ) -> Tuple[Optional[IsolationForest], Optional[StandardScaler], Optional[str]]:
        """
        Get or train a model for the given store.

        Returns: (model, scaler, fallback_level)
        """
        # Try store-specific model first
        model_key = f"store_{store_id}"
        if model_key in self._models:
            return (*self._models[model_key], None)

        # Try to train store-specific model
        train_data = self._get_training_data(db, store_id, target_date)

        if len(train_data) >= self.min_samples:
            model, scaler = self._train_model(train_data)
            self._models[model_key] = (model, scaler)
            return model, scaler, None

        # Fall back to global model
        model_key = "global"
        if model_key in self._models:
            return (*self._models[model_key], "global")

        train_data = self._get_training_data(db, None, target_date)

        if len(train_data) >= self.min_samples:
            model, scaler = self._train_model(train_data)
            self._models[model_key] = (model, scaler)
            return model, scaler, "global"

        return None, None, "insufficient_data"

    def _get_training_data(
        self,
        db: Session,
        store_id: Optional[str],
        target_date: date,
    ) -> pd.DataFrame:
        """Get training data from features_daily table."""
        window_end = target_date - timedelta(days=1)
        window_start = window_end - timedelta(days=90)  # Use 90 days for training

        query = db.query(FeatureDaily).filter(
            FeatureDaily.date >= window_start,
            FeatureDaily.date <= window_end,
        )

        if store_id:
            query = query.filter(FeatureDaily.store_id == store_id)

        results = query.all()

        if not results:
            return pd.DataFrame()

        # Convert to DataFrame
        data = []
        for r in results:
            data.append({
                "sold": r.sold,
                "delivered": r.delivered,
                "returned": r.returned,
                "delta_on_hand": r.delta_on_hand if r.delta_on_hand is not None else 0,
                "price": r.price,
                "promo_flag": 1 if r.promo_flag else 0,
                "day_of_week": r.day_of_week,
            })

        return pd.DataFrame(data)

    def _train_model(
        self, train_data: pd.DataFrame
    ) -> Tuple[IsolationForest, StandardScaler]:
        """Train Isolation Forest model."""
        # Handle missing values
        train_data = train_data.fillna(0)

        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(train_data)

        # Train model
        model = IsolationForest(
            n_estimators=self.n_estimators,
            contamination=self.contamination,
            random_state=42,
            n_jobs=-1,
        )
        model.fit(X_scaled)

        return model, scaler

    def _prepare_features(self, features: Dict) -> Optional[List[float]]:
        """Prepare feature vector from feature dictionary."""
        try:
            return [
                float(features.get("sold", 0)),
                float(features.get("delivered", 0)),
                float(features.get("returned", 0)),
                float(features.get("delta_on_hand", 0) or 0),
                float(features.get("price", 0)),
                1.0 if features.get("promo_flag", False) else 0.0,
                float(features.get("day_of_week", 0)),
            ]
        except (ValueError, TypeError):
            return None

    def _calculate_feature_reasons(
        self,
        db: Session,
        store_id: str,
        sku_id: str,
        target_date: date,
        features: Dict,
    ) -> List[Dict]:
        """
        Calculate which features contributed most to the anomaly.

        Uses z-score comparison against rolling statistics.
        """
        # Get rolling statistics for this store+SKU
        stats = self._get_rolling_stats(db, store_id, sku_id, target_date)

        if not stats:
            return []

        # Calculate z-scores for numeric features
        z_scores = []
        feature_names = ["sold", "delivered", "returned", "delta_on_hand", "price"]

        for feature in feature_names:
            value = features.get(feature, 0)
            if value is None:
                value = 0

            mean = stats.get(f"{feature}_mean", 0)
            std = stats.get(f"{feature}_std", 1)

            if std > 0:
                z = (value - mean) / std
            else:
                z = 0

            z_scores.append({
                "feature": feature,
                "value": float(value),
                "mean": float(mean),
                "std": float(std),
                "z_score": float(z),
            })

        # Sort by absolute z-score and take top 3
        z_scores.sort(key=lambda x: abs(x["z_score"]), reverse=True)

        # Format as reasons, filtering out small z-scores
        reasons = []
        for item in z_scores[:3]:
            if abs(item["z_score"]) > 1.5:  # Only include if notably different
                direction = "above" if item["z_score"] > 0 else "below"
                reasons.append({
                    "feature": item["feature"],
                    "z_score": round(item["z_score"], 2),
                    "message": f"{item['feature']} is {abs(item['z_score']):.1f} std {direction} average",
                    "value": item["value"],
                    "mean": round(item["mean"], 2),
                })

        return reasons

    def _get_rolling_stats(
        self,
        db: Session,
        store_id: str,
        sku_id: str,
        target_date: date,
    ) -> Dict:
        """Get rolling statistics for a store+SKU combination."""
        cache_key = f"{store_id}_{sku_id}_{target_date}"
        if cache_key in self._rolling_stats:
            return self._rolling_stats[cache_key]

        window_start = target_date - timedelta(days=28)
        window_end = target_date - timedelta(days=1)

        features = (
            db.query(FeatureDaily)
            .filter(
                FeatureDaily.store_id == store_id,
                FeatureDaily.sku_id == sku_id,
                FeatureDaily.date >= window_start,
                FeatureDaily.date <= window_end,
            )
            .all()
        )

        if len(features) < 5:
            # Fall back to store level
            features = (
                db.query(FeatureDaily)
                .filter(
                    FeatureDaily.store_id == store_id,
                    FeatureDaily.date >= window_start,
                    FeatureDaily.date <= window_end,
                )
                .all()
            )

        if not features:
            return {}

        # Calculate statistics
        stats = {}
        for col in ["sold", "delivered", "returned", "delta_on_hand", "price"]:
            values = []
            for f in features:
                v = getattr(f, col)
                if v is not None:
                    values.append(v)

            if values:
                stats[f"{col}_mean"] = np.mean(values)
                stats[f"{col}_std"] = np.std(values) if len(values) > 1 else 1.0
            else:
                stats[f"{col}_mean"] = 0
                stats[f"{col}_std"] = 1.0

        self._rolling_stats[cache_key] = stats
        return stats

    def save_result(
        self,
        db: Session,
        target_date: date,
        store_id: str,
        sku_id: str,
        result: IsolationForestResult,
    ) -> int:
        """Save Isolation Forest detection result to database."""
        detection = DetectionResult(
            date=target_date,
            store_id=store_id,
            sku_id=sku_id,
            detector_type=DetectorType.ISOLATION_FOREST,
            anomaly_score=result.anomaly_score,
            threshold_used=result.threshold_used,
            is_outlier=result.is_anomaly,
            reasons=result.reasons if result.reasons else None,
            fallback_used=result.fallback_used,
        )
        db.add(detection)
        db.flush()
        return detection.id

    def clear_cache(self):
        """Clear model cache."""
        self._models.clear()
        self._rolling_stats.clear()
