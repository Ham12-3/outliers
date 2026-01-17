FROM python:3.11-slim

WORKDIR /app

# Install system dependencies including postgresql-client for migrations
RUN apt-get update && apt-get install -y \
    curl \
    postgresql-client \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create cache directory
RUN mkdir -p /app/.cache

EXPOSE 8000

# Default command (overridden by docker-compose to run migrations first)
CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]
