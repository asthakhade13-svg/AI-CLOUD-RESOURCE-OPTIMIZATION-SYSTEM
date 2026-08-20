# --- Stage 1: Build dependencies ---
FROM python:3.11-slim AS builder

WORKDIR /build

COPY requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

# --- Stage 2: Final lightweight image ---
FROM python:3.11-slim

# Set system environment settings
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app

WORKDIR /app

# Copy dependency packages from builder
COPY --from=builder /root/.local /root/.local
ENV PATH=/root/.local/bin:$PATH

# Copy ML application assets
COPY src/ /app/src/
COPY artifacts/ /app/artifacts/
COPY data/ /app/data/
COPY ml_service/ /app/ml_service/
COPY rl/ /app/rl/
COPY simulation/ /app/simulation/

EXPOSE 8050

# Run uvicorn on port 8050
CMD ["uvicorn", "ml_service.main:app", "--host", "0.0.0.0", "--port", "8050"]
