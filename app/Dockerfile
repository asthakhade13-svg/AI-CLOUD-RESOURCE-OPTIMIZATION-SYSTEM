# --- Stage 1: Build dependencies ---
FROM python:3.11-slim AS builder

WORKDIR /build

COPY requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

# --- Stage 2: Final lightweight image ---
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app

WORKDIR /app

# Copy dependencies
COPY --from=builder /root/.local /root/.local
ENV PATH=/root/.local/bin:$PATH

# Copy source codes
COPY src/ /app/src/
COPY app/ /app/app/
COPY rl/ /app/rl/
COPY k8s/ /app/k8s/
COPY index.html /app/

EXPOSE 8000

# Start FastAPI Gateway app
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
