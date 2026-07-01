FROM python:3.11-slim

WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy dependency metadata and install
COPY pyproject.toml README.md ./
RUN pip install --no-cache-dir -e .

# Copy application code
COPY src/ ./src/

# Create data directory for SQLite mount
RUN mkdir -p /data

ENV PYTHONPATH=/app
ENV DATABASE_URL=sqlite+aiosqlite:////data/classifications.db

EXPOSE 7860

CMD ["python", "-m", "uvicorn", "src.api.app:app", "--host", "0.0.0.0", "--port", "7860"]
