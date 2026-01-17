"""Tests for the Tukey IQR outlier detector."""
import pytest
import numpy as np
from datetime import date, timedelta

from src.ml.tukey_detector import TukeyDetector


class TestTukeyDetector:
    """Test suite for TukeyDetector."""

    def test_calculate_fences_basic(self):
        """Test fence calculation with known values."""
        detector = TukeyDetector(rolling_window_days=28, min_samples=5)

        # Known dataset: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        # Q1 = 3.25, Q3 = 7.75, IQR = 4.5
        # Lower fence = 3.25 - 1.5 * 4.5 = -3.5
        # Upper fence = 7.75 + 1.5 * 4.5 = 14.5
        values = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        q1, q3, iqr, lower_fence, upper_fence = detector._calculate_fences(values)

        assert np.isclose(q1, 3.25)
        assert np.isclose(q3, 7.75)
        assert np.isclose(iqr, 4.5)
        assert np.isclose(lower_fence, -3.5)
        assert np.isclose(upper_fence, 14.5)

    def test_calculate_fences_with_outliers(self):
        """Test fence calculation with outliers present in data."""
        detector = TukeyDetector()

        # Dataset with outliers
        values = [10, 12, 11, 13, 9, 15, 14, 100, 11, 12]

        q1, q3, iqr, lower_fence, upper_fence = detector._calculate_fences(values)

        # Verify the outlier (100) would be detected
        assert 100 > upper_fence
        assert all(v >= lower_fence for v in [10, 12, 11, 13, 9, 15, 14, 11, 12])

    def test_detect_outlier_above_upper_fence(self, sample_metrics):
        """Test detection of value above upper fence."""
        detector = TukeyDetector(min_samples=5)

        # Calculate fences using normal values
        normal_values = [v for v in sample_metrics["sold"] if v < 50]
        q1, q3, iqr, lower_fence, upper_fence = detector._calculate_fences(normal_values)

        # The value 100 should be above the upper fence
        assert 100 > upper_fence
        outlier_distance = 100 - upper_fence
        assert outlier_distance > 0

    def test_detect_outlier_below_lower_fence(self, sample_metrics):
        """Test detection of value below lower fence."""
        detector = TukeyDetector(min_samples=5)

        # Calculate fences using normal values
        normal_values = [v for v in sample_metrics["delta_on_hand"] if v > -20]
        q1, q3, iqr, lower_fence, upper_fence = detector._calculate_fences(normal_values)

        # The value -50 should be below the lower fence
        assert -50 < lower_fence
        outlier_distance = -50 - lower_fence
        assert outlier_distance < 0

    def test_insufficient_data_handling(self):
        """Test handling of insufficient data."""
        detector = TukeyDetector(min_samples=10)

        # Only 5 samples, should be insufficient
        values = [1, 2, 3, 4, 5]

        # The detector should still calculate fences
        q1, q3, iqr, lower_fence, upper_fence = detector._calculate_fences(values)

        # Fences should be calculated even with small sample
        assert iqr > 0

    def test_constant_values(self):
        """Test handling of constant values (IQR = 0)."""
        detector = TukeyDetector()

        # All values are the same
        values = [5, 5, 5, 5, 5, 5, 5, 5, 5, 5]

        q1, q3, iqr, lower_fence, upper_fence = detector._calculate_fences(values)

        assert q1 == 5
        assert q3 == 5
        assert iqr == 0
        assert lower_fence == 5
        assert upper_fence == 5

    def test_negative_values(self):
        """Test handling of negative values."""
        detector = TukeyDetector()

        values = [-10, -8, -9, -7, -11, -6, -12, -5]
        q1, q3, iqr, lower_fence, upper_fence = detector._calculate_fences(values)

        # All fences should be negative for this dataset
        assert lower_fence < 0
        assert upper_fence < 0
        assert q1 < q3

    def test_k_parameter(self):
        """Test that k parameter affects fence width."""
        detector_normal = TukeyDetector(k=1.5)
        detector_wide = TukeyDetector(k=3.0)

        values = list(range(1, 21))

        _, _, _, lower_normal, upper_normal = detector_normal._calculate_fences(values)
        _, _, _, lower_wide, upper_wide = detector_wide._calculate_fences(values)

        # Wider k should produce wider fences
        assert lower_wide < lower_normal
        assert upper_wide > upper_normal

    def test_single_outlier_detection(self):
        """Test that a single extreme value is detected as outlier."""
        detector = TukeyDetector(k=1.5)

        # Normal distribution with one extreme value
        normal_values = list(range(10, 21))  # 10 to 20
        all_values = normal_values + [1000]  # Add extreme outlier

        q1, q3, iqr, lower_fence, upper_fence = detector._calculate_fences(all_values)

        # 1000 should definitely be an outlier
        assert 1000 > upper_fence
        # Normal values should not be outliers
        for v in normal_values:
            assert lower_fence <= v <= upper_fence


class TestTukeyResultDataclass:
    """Test the TukeyResult dataclass."""

    def test_result_creation(self):
        """Test creating a TukeyResult."""
        from src.ml.tukey_detector import TukeyResult

        result = TukeyResult(
            metric_name="sold",
            actual_value=100,
            q1=10,
            q3=20,
            iqr=10,
            lower_fence=-5,
            upper_fence=35,
            is_outlier=True,
            outlier_distance=65,
            sample_size=30,
            fallback_used=None,
        )

        assert result.metric_name == "sold"
        assert result.is_outlier is True
        assert result.outlier_distance == 65

    def test_result_with_fallback(self):
        """Test TukeyResult with fallback indicator."""
        from src.ml.tukey_detector import TukeyResult

        result = TukeyResult(
            metric_name="delta_on_hand",
            actual_value=-50,
            q1=-5,
            q3=5,
            iqr=10,
            lower_fence=-20,
            upper_fence=20,
            is_outlier=True,
            outlier_distance=-30,
            sample_size=8,
            fallback_used="store",
        )

        assert result.fallback_used == "store"
        assert result.outlier_distance < 0  # Below lower fence
