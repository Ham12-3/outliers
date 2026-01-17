"""Application configuration."""
from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    database_url: str = "postgresql://outliers:outliers_dev@localhost:5432/outliers"
    environment: str = "development"

    # Scheduler settings
    scheduler_enabled: bool = True
    detection_schedule_hour: int = 2
    detection_schedule_minute: int = 0

    # Detection settings
    tukey_rolling_window_days: int = 28
    tukey_min_samples: int = 10
    isolation_forest_contamination: float = 0.05
    isolation_forest_min_samples: int = 100

    class Config:
        env_file = ".env"
        extra = "ignore"


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()
