"""Test configuration and fixtures."""
import pytest
from datetime import date, timedelta
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from src.database.models import Base


@pytest.fixture(scope="function")
def db_session():
    """Create an in-memory SQLite database for testing."""
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    session = Session()
    try:
        yield session
    finally:
        session.close()
        Base.metadata.drop_all(engine)


@pytest.fixture
def sample_dates():
    """Generate sample dates for testing."""
    end_date = date.today()
    return [end_date - timedelta(days=i) for i in range(30)]


@pytest.fixture
def sample_metrics():
    """Generate sample metrics data for testing."""
    return {
        "sold": [10, 12, 11, 13, 9, 15, 14, 100, 11, 12],  # 100 is an outlier
        "delta_on_hand": [-5, -3, -4, -2, -6, -50, -4, -3, -5, -4],  # -50 is an outlier
    }
