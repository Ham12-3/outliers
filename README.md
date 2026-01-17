# Anomaly Detection Platform

A flexible, full-stack application for detecting anomalies in any tabular data. Upload any CSV, configure which columns to analyze, and run sophisticated anomaly detection algorithms.

## Features

- **Flexible Data Ingestion**: Upload ANY CSV structure - you choose which columns are identifiers, metrics, and attributes
- **Automatic Schema Detection**: Smart column type detection (dates, numbers, categories) with suggested roles
- **Dual Detection Methods**:
  - **Tukey IQR**: Statistical outlier detection using boxplot method
  - **Isolation Forest**: ML-based anomaly detection with feature importance explanations
- **Dataset Management**: Organize data into named datasets with configurable schemas
- **Incident Management**: Grouped alerts with severity scoring, assignment, and status workflow
- **Explainability**: Clear explanations for why each anomaly was flagged
- **Dashboard**: KPIs, time series charts, severity distribution, and top affected identifiers
- **Scheduled Detection**: Daily automated runs via APScheduler

## Tech Stack

- **Frontend**: Next.js 14 (App Router) + Tailwind CSS + Recharts
- **Backend**: FastAPI (Python)
- **ML**: scikit-learn + NumPy + Pandas
- **Database**: PostgreSQL
- **Job Scheduling**: APScheduler
- **Containerisation**: Docker Compose

## Quick Start

### Prerequisites

- Docker and Docker Compose
- Git

### Running the Application

```bash
# Clone the repository
git clone <repository-url>
cd outliers

# Start all services
docker compose up --build

# The application will be available at:
# - Frontend: http://localhost:3000
# - API: http://localhost:8000
# - API Docs: http://localhost:8000/docs
```

### First Run Setup

1. Open the frontend at http://localhost:3000
2. Navigate to the **Data** page
3. Click **Generate Demo Data** to create synthetic data with realistic patterns and anomalies
4. Click **Run Detection Now** to execute the anomaly detection pipeline
5. View results on the **Dashboard** and **Incidents** pages

## Project Structure

```
outliers/
├── apps/
│   ├── api/                    # FastAPI backend
│   │   ├── alembic/           # Database migrations
│   │   ├── src/
│   │   │   ├── database/      # Models and connection
│   │   │   ├── ml/            # Detection algorithms
│   │   │   ├── routers/       # API endpoints
│   │   │   ├── services/      # Business logic
│   │   │   ├── config.py      # Settings
│   │   │   ├── main.py        # FastAPI app
│   │   │   └── scheduler.py   # APScheduler jobs
│   │   └── tests/             # Unit and API tests
│   │
│   └── web/                    # Next.js frontend
│       └── src/
│           ├── app/           # Pages (App Router)
│           ├── components/    # React components
│           └── lib/           # API client
│
├── docker-compose.yml
└── README.md
```

## API Endpoints

### Data Management
- `POST /data/upload-csv` - Upload CSV with daily metrics
- `POST /data/generate-demo` - Generate synthetic demo data
- `GET /data/stats` - Get data statistics

### Detection
- `POST /detect/run` - Run detection pipeline
- `GET /detect/status` - Get detection status

### Incidents
- `GET /incidents` - List incidents with filters
- `GET /incidents/{id}` - Get incident details
- `PATCH /incidents/{id}` - Update incident status/assignee
- `POST /incidents/{id}/notes` - Add note to incident
- `GET /incidents/summary` - Get incident statistics
- `GET /incidents/{id}/detection-details` - Get detection explanations

### Metrics
- `GET /metrics/timeseries` - Get time series data for store/SKU
- `GET /metrics/boxplot` - Get boxplot statistics
- `GET /metrics/stores` - List all stores
- `GET /metrics/skus` - List all SKUs
- `GET /metrics/incidents-over-time` - Incidents over time chart data
- `GET /metrics/severity-distribution` - Severity distribution data
- `GET /metrics/top-stores` - Top stores by incident count

## Flexible Data Model

The system accepts **any CSV structure**. You define which columns are:

- **Date Column** (optional): For time-series analysis
- **Identifier Columns**: Columns that group your data (e.g., store, product, region, customer)
- **Metric Columns**: Numeric columns to run anomaly detection on
- **Attribute Columns**: Additional data to store but not analyze

### How It Works

1. **Upload Any CSV** - The system analyzes your file and detects column types:
   - `date` / `datetime` - Date columns
   - `integer` / `float` - Numeric columns (candidates for metrics)
   - `categorical` - Low-cardinality strings (candidates for identifiers)
   - `text` - High-cardinality strings
   - `boolean` - True/false values

2. **Configure Schema** - Choose roles for each column:
   - **Identifiers** (purple): Used for grouping data
   - **Metrics** (green): Numeric columns for anomaly detection
   - **Attributes** (orange): Extra data to store

3. **Run Detection** - The system runs:
   - Tukey IQR on each metric column
   - Isolation Forest on combined metrics (if 2+ metrics)

### Example: Sales Data

```csv
date,store,product,category,units_sold,revenue,is_promo
2024-01-15,STORE-001,SKU-001,Electronics,25,499.75,false
2024-01-15,STORE-001,SKU-002,Food,150,450.00,true
```

You might configure this as:
- **Date**: `date`
- **Identifiers**: `store`, `product`
- **Metrics**: `units_sold`, `revenue`
- **Attributes**: `category`, `is_promo`

### Example: Sensor Data

```csv
timestamp,sensor_id,location,temperature,humidity,pressure
2024-01-15 10:00:00,SENSOR-A,Building-1,22.5,45.2,1013.25
2024-01-15 10:00:00,SENSOR-B,Building-2,23.1,42.8,1012.98
```

You might configure this as:
- **Date**: `timestamp`
- **Identifiers**: `sensor_id`, `location`
- **Metrics**: `temperature`, `humidity`, `pressure`
- **Attributes**: (none)

### Date Format Support

The system handles various date formats automatically:

| Format | Example | Notes |
|--------|---------|-------|
| ISO 8601 | `2024-01-15` | Preferred format |
| UK Format | `15/01/2024` | DD/MM/YYYY |
| Text | `January 15, 2024` | Various text formats |
| Unix timestamp | `1705276800` | Seconds since epoch |

### API Endpoints

**Analyze CSV** (detect column types):
```bash
curl -X POST http://localhost:8000/data/analyze-csv -F "file=@data.csv"
```

**Create Dataset** (define schema):
```bash
curl -X POST http://localhost:8000/data/datasets \
  -H "Content-Type: application/json" \
  -d '{
    "name": "Sales Data",
    "date_column": "date",
    "identifier_columns": ["store", "product"],
    "metric_columns": ["units_sold", "revenue"],
    "attribute_columns": ["category"]
  }'
```

**Upload to Dataset**:
```bash
curl -X POST http://localhost:8000/data/datasets/1/upload -F "file=@data.csv"
```

**Run Detection**:
```bash
curl -X POST http://localhost:8000/detect/run/1
```

## Detection Algorithms

### Tukey IQR Method
- Computes Q1, Q3, IQR over a rolling 28-day window
- Fences: Q1 - 1.5×IQR (lower), Q3 + 1.5×IQR (upper)
- Detects outliers for `sold` and `delta_on_hand` metrics
- Fallback hierarchy for sparse data: store+SKU → store → SKU → global

### Isolation Forest
- Features: sold, delivered, returned, delta_on_hand, price, promo_flag, day_of_week
- Training: per-store model if sufficient data, otherwise global model
- Contamination threshold: 5% (configurable)
- Explanations: z-score comparison for top 3 contributing features

## Environment Variables

### API (apps/api/.env)
```
DATABASE_URL=postgresql://outliers:outliers_dev@localhost:5432/outliers
ENVIRONMENT=development
SCHEDULER_ENABLED=true
DETECTION_SCHEDULE_HOUR=2
DETECTION_SCHEDULE_MINUTE=0
```

### Web (apps/web/.env)
```
API_URL=http://localhost:8000
NEXT_PUBLIC_API_URL=http://localhost:8000
```

## Running Tests

```bash
# API tests
cd apps/api
pip install -r requirements.txt
pytest tests/ -v

# Run specific test file
pytest tests/test_tukey_detector.py -v
pytest tests/test_isolation_forest.py -v
pytest tests/test_api.py -v
```

## Database Migrations

**Automatic migrations**: The API container automatically runs `alembic upgrade head` on startup before serving requests. You should see this in the logs:
```
=== Running database migrations ===
... (migration output)
=== Migrations complete, starting API ===
```

### Manual migration commands

```bash
cd apps/api

# Create a new migration (after changing models.py)
alembic revision --autogenerate -m "description"

# Run migrations manually
alembic upgrade head

# Rollback one migration
alembic downgrade -1

# Check current migration version
alembic current
```

### Reset database

To completely reset the database (deletes all data):

```bash
# Stop all containers and remove the database volume
docker compose down -v

# Start fresh
docker compose up --build
```

## Troubleshooting

### Migrations stuck or failing
If migrations fail with errors like "type already exists" or "relation already exists", reset the database:
```bash
docker compose down -v
docker compose up --build
```
This deletes the database volume and starts fresh.

### "relation does not exist" errors
This means migrations haven't run. Check the API logs for migration output. If needed, reset the database:
```bash
docker compose down -v
docker compose up --build
```

### Containers not starting
```bash
# Check container logs
docker compose logs api
docker compose logs web
docker compose logs postgres

# Rebuild containers
docker compose down
docker compose up --build
```

### Database connection issues
```bash
# Verify Postgres is healthy
docker compose exec postgres pg_isready -U outliers

# Check tables exist
docker compose exec postgres psql -U outliers -d outliers -c "\dt"
```

### Frontend not connecting to API
- Ensure API is running and healthy: http://localhost:8000/health
- Check CORS settings in API
- Verify environment variables in web container

### Detection running slowly
- Large datasets may take time to process
- Consider running in background mode: `POST /detect/run?background=true`
- Check database indexes are created

## Future Improvements

- **Streaming Ingestion**: Real-time data ingestion via Kafka or similar
- **RBAC**: Role-based access control with user authentication
- **Enhanced Explainability**: SHAP values for Isolation Forest, more detailed Tukey explanations
- **Drift Monitoring**: Detect concept drift in detection models over time
- **Alerting**: Email/Slack notifications for high-severity incidents
- **Multi-tenancy**: Support for multiple organisations with isolated data
- **Model Versioning**: Track and compare different detection model versions
- **A/B Testing**: Compare detection methods in production
- **Historical Analysis**: Long-term trend analysis and reporting
- **Export**: PDF/CSV export of incidents and reports

## Licence

MIT
