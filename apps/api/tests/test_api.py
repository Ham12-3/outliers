"""Tests for the FastAPI endpoints."""
import pytest
from datetime import date, datetime
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from src.main import app
from src.database.models import Base, Incident, IncidentStatus
from src.database.connection import get_db


# Create test database
engine = create_engine(
    "sqlite:///:memory:",
    connect_args={"check_same_thread": False},
    poolclass=StaticPool,
)
TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def override_get_db():
    """Override database dependency for testing."""
    db = TestingSessionLocal()
    try:
        yield db
    finally:
        db.close()


app.dependency_overrides[get_db] = override_get_db


@pytest.fixture(scope="function")
def client():
    """Create test client with fresh database."""
    Base.metadata.create_all(bind=engine)
    yield TestClient(app)
    Base.metadata.drop_all(bind=engine)


@pytest.fixture
def sample_incident(client):
    """Create a sample incident for testing."""
    db = TestingSessionLocal()
    incident = Incident(
        date=date.today(),
        store_id="LONDON-001",
        status=IncidentStatus.OPEN,
        severity_score=75.5,
        headline="Test incident",
        description="Test description",
        sku_count=3,
        estimated_impact=1500.50,
        detectors_triggered=["tukey", "isolation_forest"],
        dedup_key="test_dedup_key_001",
    )
    db.add(incident)
    db.commit()
    db.refresh(incident)
    incident_id = incident.id
    db.close()
    return incident_id


class TestHealthEndpoint:
    """Test health check endpoint."""

    def test_health_check(self, client):
        """Test that health endpoint returns healthy status."""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert data["service"] == "inventory-anomaly-api"

    def test_root_endpoint(self, client):
        """Test root endpoint."""
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert "message" in data
        assert "docs" in data


class TestIncidentsEndpoints:
    """Test incidents API endpoints."""

    def test_list_incidents_empty(self, client):
        """Test listing incidents when none exist."""
        response = client.get("/incidents")
        assert response.status_code == 200
        data = response.json()
        assert data["incidents"] == []
        assert data["total"] == 0

    def test_list_incidents_with_data(self, client, sample_incident):
        """Test listing incidents with data."""
        response = client.get("/incidents")
        assert response.status_code == 200
        data = response.json()
        assert len(data["incidents"]) == 1
        assert data["total"] == 1
        assert data["incidents"][0]["headline"] == "Test incident"

    def test_list_incidents_filter_by_status(self, client, sample_incident):
        """Test filtering incidents by status."""
        # Filter for open incidents
        response = client.get("/incidents?status=open")
        assert response.status_code == 200
        data = response.json()
        assert len(data["incidents"]) == 1

        # Filter for resolved incidents (should be empty)
        response = client.get("/incidents?status=resolved")
        assert response.status_code == 200
        data = response.json()
        assert len(data["incidents"]) == 0

    def test_list_incidents_filter_by_store(self, client, sample_incident):
        """Test filtering incidents by store."""
        response = client.get("/incidents?store_id=LONDON-001")
        assert response.status_code == 200
        data = response.json()
        assert len(data["incidents"]) == 1

        response = client.get("/incidents?store_id=MANCHESTER-001")
        assert response.status_code == 200
        data = response.json()
        assert len(data["incidents"]) == 0

    def test_list_incidents_pagination(self, client, sample_incident):
        """Test incidents pagination."""
        response = client.get("/incidents?page=1&page_size=10")
        assert response.status_code == 200
        data = response.json()
        assert data["page"] == 1
        assert data["page_size"] == 10

    def test_get_incident_detail(self, client, sample_incident):
        """Test getting a single incident detail."""
        response = client.get(f"/incidents/{sample_incident}")
        assert response.status_code == 200
        data = response.json()
        assert data["id"] == sample_incident
        assert data["headline"] == "Test incident"
        assert data["severity_score"] == 75.5
        assert data["store_id"] == "LONDON-001"

    def test_get_incident_not_found(self, client):
        """Test getting a non-existent incident."""
        response = client.get("/incidents/99999")
        assert response.status_code == 404

    def test_update_incident_status(self, client, sample_incident):
        """Test updating incident status."""
        response = client.patch(
            f"/incidents/{sample_incident}",
            json={"status": "investigating"}
        )
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "investigating"

        # Verify the change persisted
        response = client.get(f"/incidents/{sample_incident}")
        assert response.json()["status"] == "investigating"

    def test_update_incident_resolved(self, client, sample_incident):
        """Test resolving an incident."""
        response = client.patch(
            f"/incidents/{sample_incident}",
            json={
                "status": "resolved",
                "resolution_reason": "False positive"
            }
        )
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "resolved"
        assert data["resolution_reason"] == "False positive"
        assert data["resolved_at"] is not None

    def test_update_incident_assignee(self, client, sample_incident):
        """Test assigning an incident."""
        response = client.patch(
            f"/incidents/{sample_incident}",
            json={"assignee": "john.smith"}
        )
        assert response.status_code == 200
        data = response.json()
        assert data["assignee"] == "john.smith"

    def test_update_incident_invalid_status(self, client, sample_incident):
        """Test updating with invalid status."""
        response = client.patch(
            f"/incidents/{sample_incident}",
            json={"status": "invalid_status"}
        )
        assert response.status_code == 400

    def test_add_incident_note(self, client, sample_incident):
        """Test adding a note to an incident."""
        response = client.post(
            f"/incidents/{sample_incident}/notes",
            json={
                "content": "This is a test note",
                "author": "test_user"
            }
        )
        assert response.status_code == 200
        data = response.json()
        assert data["content"] == "This is a test note"
        assert data["author"] == "test_user"
        assert data["note_type"] == "comment"

    def test_incidents_summary(self, client, sample_incident):
        """Test incidents summary endpoint."""
        response = client.get("/incidents/summary")
        assert response.status_code == 200
        data = response.json()
        assert "status_counts" in data
        assert "high_severity_count" in data
        assert "affected_stores" in data

    def test_incidents_summary_with_multiple_statuses(self, client):
        """Test incidents summary with incidents in different statuses."""
        db = TestingSessionLocal()

        # Create incidents with different statuses
        incidents = [
            Incident(
                date=date.today(),
                store_id="LONDON-001",
                status=IncidentStatus.OPEN,
                severity_score=80.0,
                headline="Open incident 1",
                detectors_triggered=["tukey"],
                dedup_key="key_open_1",
            ),
            Incident(
                date=date.today(),
                store_id="LONDON-002",
                status=IncidentStatus.OPEN,
                severity_score=50.0,
                headline="Open incident 2",
                detectors_triggered=["tukey"],
                dedup_key="key_open_2",
            ),
            Incident(
                date=date.today(),
                store_id="MANCHESTER-001",
                status=IncidentStatus.INVESTIGATING,
                severity_score=75.0,
                headline="Investigating incident",
                detectors_triggered=["isolation_forest"],
                dedup_key="key_investigating_1",
            ),
            Incident(
                date=date.today(),
                store_id="BIRMINGHAM-001",
                status=IncidentStatus.RESOLVED,
                severity_score=90.0,
                headline="Resolved incident",
                detectors_triggered=["tukey"],
                dedup_key="key_resolved_1",
            ),
        ]
        for incident in incidents:
            db.add(incident)
        db.commit()
        db.close()

        response = client.get("/incidents/summary")
        assert response.status_code == 200
        data = response.json()

        # Verify status counts
        assert data["status_counts"]["open"] == 2
        assert data["status_counts"]["investigating"] == 1
        assert data["status_counts"]["resolved"] == 1

        # High severity (>=70) and NOT resolved: 80.0 (open) + 75.0 (investigating) = 2
        assert data["high_severity_count"] == 2

        # Affected stores should be 4 (unique)
        assert data["affected_stores"] == 4

    def test_incidents_summary_high_severity_excludes_resolved(self, client):
        """Test that high severity count excludes resolved incidents."""
        db = TestingSessionLocal()

        # Create a high severity resolved incident
        incident = Incident(
            date=date.today(),
            store_id="LONDON-001",
            status=IncidentStatus.RESOLVED,
            severity_score=95.0,  # High severity but resolved
            headline="Resolved high severity",
            detectors_triggered=["tukey"],
            dedup_key="key_high_resolved",
        )
        db.add(incident)
        db.commit()
        db.close()

        response = client.get("/incidents/summary")
        assert response.status_code == 200
        data = response.json()

        # Should be 0 because the only high severity incident is resolved
        assert data["high_severity_count"] == 0


class TestDataEndpoints:
    """Test data management endpoints."""

    def test_data_stats_empty(self, client):
        """Test data stats when no data exists."""
        response = client.get("/data/stats")
        assert response.status_code == 200
        data = response.json()
        assert data["total_rows"] == 0

    def test_upload_csv_invalid_file(self, client):
        """Test uploading a non-CSV file."""
        response = client.post(
            "/data/upload-csv",
            files={"file": ("test.txt", b"not a csv", "text/plain")}
        )
        assert response.status_code == 400


class TestMetricsEndpoints:
    """Test metrics endpoints."""

    def test_list_stores_empty(self, client):
        """Test listing stores when no data exists."""
        response = client.get("/metrics/stores")
        assert response.status_code == 200
        data = response.json()
        assert data["stores"] == []

    def test_list_skus_empty(self, client):
        """Test listing SKUs when no data exists."""
        response = client.get("/metrics/skus")
        assert response.status_code == 200
        data = response.json()
        assert data["skus"] == []

    def test_incidents_over_time(self, client):
        """Test incidents over time endpoint."""
        response = client.get("/metrics/incidents-over-time?days=30")
        assert response.status_code == 200
        data = response.json()
        assert "data" in data

    def test_severity_distribution(self, client):
        """Test severity distribution endpoint."""
        response = client.get("/metrics/severity-distribution")
        assert response.status_code == 200
        data = response.json()
        assert "distribution" in data

    def test_top_stores(self, client):
        """Test top stores endpoint."""
        response = client.get("/metrics/top-stores?limit=5")
        assert response.status_code == 200
        data = response.json()
        assert "stores" in data


class TestDetectionEndpoints:
    """Test detection endpoints."""

    def test_detection_status(self, client):
        """Test detection status endpoint."""
        response = client.get("/detect/status")
        assert response.status_code == 200
        data = response.json()
        assert "running" in data
