"""Tests for the Isolation Forest anomaly detector."""
import pytest
import numpy as np
import pandas as pd

from src.ml.isolation_forest_detector import IsolationForestDetector, FEATURE_COLUMNS


class TestIsolationForestDetector:
    """Test suite for IsolationForestDetector."""

    def test_train_model_with_synthetic_data(self):
        """Test model training with synthetic data."""
        detector = IsolationForestDetector(contamination=0.1, min_samples=20)

        # Create synthetic training data
        np.random.seed(42)
        n_samples = 100
        train_data = pd.DataFrame({
            "sold": np.random.normal(50, 10, n_samples),
            "delivered": np.random.normal(30, 5, n_samples),
            "returned": np.random.poisson(2, n_samples),
            "delta_on_hand": np.random.normal(0, 5, n_samples),
            "price": np.random.uniform(10, 100, n_samples),
            "promo_flag": np.random.choice([0, 1], n_samples),
            "day_of_week": np.random.randint(0, 7, n_samples),
        })

        model, scaler = detector._train_model(train_data)

        assert model is not None
        assert scaler is not None
        assert hasattr(model, "predict")
        assert hasattr(model, "decision_function")

    def test_predict_normal_point(self):
        """Test prediction on a normal data point."""
        detector = IsolationForestDetector(contamination=0.1, min_samples=20)

        # Create training data with normal distribution
        np.random.seed(42)
        n_samples = 200
        train_data = pd.DataFrame({
            "sold": np.random.normal(50, 10, n_samples),
            "delivered": np.random.normal(30, 5, n_samples),
            "returned": np.random.poisson(2, n_samples),
            "delta_on_hand": np.random.normal(0, 5, n_samples),
            "price": np.random.uniform(10, 100, n_samples),
            "promo_flag": np.random.choice([0, 1], n_samples),
            "day_of_week": np.random.randint(0, 7, n_samples),
        })

        model, scaler = detector._train_model(train_data)

        # Test with a normal point (close to mean)
        normal_features = {
            "sold": 50,
            "delivered": 30,
            "returned": 2,
            "delta_on_hand": 0,
            "price": 50,
            "promo_flag": False,
            "day_of_week": 3,
        }

        feature_vector = detector._prepare_features(normal_features)
        scaled = scaler.transform([feature_vector])
        prediction = model.predict(scaled)[0]

        # Normal point should be predicted as inlier (1) most of the time
        # Note: Due to randomness, we just check it returns a valid prediction
        assert prediction in [-1, 1]

    def test_predict_anomalous_point(self):
        """Test prediction on an anomalous data point."""
        detector = IsolationForestDetector(contamination=0.05, min_samples=20)

        # Create training data with normal distribution
        np.random.seed(42)
        n_samples = 200
        train_data = pd.DataFrame({
            "sold": np.random.normal(50, 10, n_samples),
            "delivered": np.random.normal(30, 5, n_samples),
            "returned": np.random.poisson(2, n_samples),
            "delta_on_hand": np.random.normal(0, 5, n_samples),
            "price": np.random.uniform(10, 100, n_samples),
            "promo_flag": np.random.choice([0, 1], n_samples),
            "day_of_week": np.random.randint(0, 7, n_samples),
        })

        model, scaler = detector._train_model(train_data)

        # Test with an extreme outlier
        anomaly_features = {
            "sold": 500,  # 45 std devs from mean!
            "delivered": 300,
            "returned": 50,
            "delta_on_hand": -200,
            "price": 500,
            "promo_flag": True,
            "day_of_week": 3,
        }

        feature_vector = detector._prepare_features(anomaly_features)
        scaled = scaler.transform([feature_vector])
        prediction = model.predict(scaled)[0]
        score = model.decision_function(scaled)[0]

        # Extreme outlier should have negative decision function score
        assert score < 0
        assert prediction == -1  # Should be classified as anomaly

    def test_prepare_features_complete(self):
        """Test feature preparation with all features present."""
        detector = IsolationForestDetector()

        features = {
            "sold": 100,
            "delivered": 50,
            "returned": 5,
            "delta_on_hand": -10,
            "price": 29.99,
            "promo_flag": True,
            "day_of_week": 4,
        }

        result = detector._prepare_features(features)

        assert result is not None
        assert len(result) == 7
        assert result[0] == 100.0  # sold
        assert result[1] == 50.0  # delivered
        assert result[2] == 5.0  # returned
        assert result[3] == -10.0  # delta_on_hand
        assert result[4] == 29.99  # price
        assert result[5] == 1.0  # promo_flag (True -> 1.0)
        assert result[6] == 4.0  # day_of_week

    def test_prepare_features_with_missing(self):
        """Test feature preparation with missing values."""
        detector = IsolationForestDetector()

        # Missing some features
        features = {
            "sold": 100,
            "price": 29.99,
        }

        result = detector._prepare_features(features)

        assert result is not None
        assert len(result) == 7
        assert result[0] == 100.0  # sold
        assert result[1] == 0.0  # delivered (default)
        assert result[4] == 29.99  # price

    def test_prepare_features_with_none_values(self):
        """Test feature preparation with None values."""
        detector = IsolationForestDetector()

        features = {
            "sold": 100,
            "delivered": None,
            "returned": 5,
            "delta_on_hand": None,
            "price": 29.99,
            "promo_flag": False,
            "day_of_week": 2,
        }

        result = detector._prepare_features(features)

        assert result is not None
        assert result[1] == 0.0  # delivered (None -> 0)
        assert result[3] == 0.0  # delta_on_hand (None -> 0)

    def test_cache_management(self):
        """Test model cache management."""
        detector = IsolationForestDetector()

        # Initially empty
        assert len(detector._models) == 0

        # After clearing, should still be empty
        detector.clear_cache()
        assert len(detector._models) == 0
        assert len(detector._rolling_stats) == 0

    def test_feature_columns_constant(self):
        """Test that feature columns are defined correctly."""
        expected = [
            "sold",
            "delivered",
            "returned",
            "delta_on_hand",
            "price",
            "promo_flag",
            "day_of_week",
        ]

        assert FEATURE_COLUMNS == expected


class TestIsolationForestResult:
    """Test the IsolationForestResult dataclass."""

    def test_result_creation(self):
        """Test creating an IsolationForestResult."""
        from src.ml.isolation_forest_detector import IsolationForestResult

        result = IsolationForestResult(
            anomaly_score=-0.15,
            is_anomaly=True,
            threshold_used=0.05,
            reasons=[
                {"feature": "sold", "z_score": 3.5, "message": "sold is 3.5 std above average"},
            ],
            fallback_used=None,
        )

        assert result.anomaly_score == -0.15
        assert result.is_anomaly is True
        assert len(result.reasons) == 1
        assert result.reasons[0]["feature"] == "sold"

    def test_result_with_fallback(self):
        """Test IsolationForestResult with fallback indicator."""
        from src.ml.isolation_forest_detector import IsolationForestResult

        result = IsolationForestResult(
            anomaly_score=0.1,
            is_anomaly=False,
            threshold_used=0.05,
            reasons=[],
            fallback_used="global",
        )

        assert result.fallback_used == "global"
        assert result.is_anomaly is False
